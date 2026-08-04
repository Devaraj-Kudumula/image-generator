"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Copy, Loader2, Send, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Alert } from "@/components/ui/alert";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { api, CHAT_ASPECT_RATIOS } from "@/lib/api";
import { buildHistoryForApi, themeKickoffUserMessage } from "@/lib/chat-theme-utils";
import { stripThemeChatFiller } from "@/lib/chat-response-utils";
import { generateId } from "@/lib/utils";
import { useChatStore, useGalleryStore } from "@/lib/store/generation-store";

interface StreamingChatProps {
  sessionId: string;
  systemPromptOverride?: string;
}

/** Small proportional rectangle previewing an aspect ratio (e.g. "16:9"). */
function RatioGlyph({ ratio }: { ratio: string }) {
  const [w, h] = ratio.split(":").map(Number);
  const max = 20;
  const width = w >= h ? max : Math.round((w / h) * max);
  const height = h >= w ? max : Math.round((h / w) * max);
  return (
    <span className="flex h-5 w-5 items-center justify-center">
      <span
        className="rounded-[2px] border-[1.5px] border-current"
        style={{ width, height }}
      />
    </span>
  );
}

function getMessageText(message: { parts: Array<{ type: string; text?: string }> }) {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("");
}

function createAssistantUiMessage(content: string): UIMessage {
  return {
    id: generateId("msg"),
    role: "assistant",
    parts: [{ type: "text", text: content }],
  };
}

export function StreamingChat({ sessionId, systemPromptOverride }: StreamingChatProps) {
  const addImage = useChatStore((s) => s.addImage);
  const setUiMessages = useChatStore((s) => s.setUiMessages);
  const markThemeKickoff = useChatStore((s) => s.markThemeKickoff);
  const markKickoffAssistantMessage = useChatStore((s) => s.markKickoffAssistantMessage);
  const chatAspectRatio = useChatStore((s) => s.chatAspectRatio);
  const setChatAspectRatio = useChatStore((s) => s.setChatAspectRatio);
  const session = useChatStore((s) =>
    s.sessions.find((entry) => entry.id === sessionId)
  );
  const themeKickoffIds = session?.themeKickoffIds ?? [];
  const kickoffAssistantMessageIds = session?.kickoffAssistantMessageIds ?? [];
  const themeMessages = useMemo(
    () => session?.messages.filter((m) => m.role === "theme") ?? [],
    [session?.messages]
  );
  const addToGallery = useGalleryStore((s) => s.addImage);
  const [input, setInput] = useState("");
  const [generatingFor, setGeneratingFor] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [themeKickoffLoading, setThemeKickoffLoading] = useState(false);
  const [themeKickoffError, setThemeKickoffError] = useState<string | null>(null);
  const kickoffInFlightRef = useRef<string | null>(null);

  const initialMessages = useMemo(
    () => session?.uiMessages ?? [],
    [session?.uiMessages]
  );

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
        body: { systemPromptOverride },
      }),
    [systemPromptOverride]
  );

  const { messages, sendMessage, setMessages, status, error } = useChat({
    id: sessionId,
    messages: initialMessages,
    transport,
  });

  const isLoading = status === "submitted" || status === "streaming";

  useEffect(() => {
    setInput("");
    setGenerateError(null);
    setGeneratingFor(null);
    setThemeKickoffError(null);
  }, [sessionId]);

  useEffect(() => {
    setUiMessages(sessionId, messages as UIMessage[]);
  }, [messages, sessionId, setUiMessages]);

  useEffect(() => {
    const latestTheme = themeMessages[themeMessages.length - 1];
    if (!latestTheme) return;
    if (themeKickoffIds.includes(latestTheme.id)) return;
    if (kickoffInFlightRef.current === latestTheme.id) return;

    let cancelled = false;
    kickoffInFlightRef.current = latestTheme.id;

    const runThemeKickoff = async () => {
      const prompt = latestTheme.content.trim();
      if (!prompt) {
        setThemeKickoffError(
          "This theme has no prompt text yet. Edit lib/chat-themes.json."
        );
        return;
      }

      setThemeKickoffLoading(true);
      setThemeKickoffError(null);

      try {
        const history = buildHistoryForApi(
          messages.map((msg) => ({
            role: msg.role,
            content: getMessageText(msg),
          }))
        );

        const label = latestTheme.themeLabel || latestTheme.themeId || "Custom";
        const data = await api.chatMessage({
          user_message: themeKickoffUserMessage(label),
          history,
          system_prompt_override: prompt,
        });

        if (cancelled) return;

        const answer = data.answer?.trim() || "No answer generated.";
        const kickoffMessage = createAssistantUiMessage(answer);
        setMessages((current) => [...current, kickoffMessage]);
        markThemeKickoff(sessionId, latestTheme.id);
        markKickoffAssistantMessage(sessionId, kickoffMessage.id);
      } catch (err) {
        if (cancelled) return;
        const message =
          err instanceof Error ? err.message : "Failed to get theme acknowledgment";
        setThemeKickoffError(message);
        setMessages((current) => [
          ...current,
          createAssistantUiMessage(`Error: ${message}`),
        ]);
      } finally {
        if (!cancelled) setThemeKickoffLoading(false);
        if (kickoffInFlightRef.current === latestTheme.id) {
          kickoffInFlightRef.current = null;
        }
      }
    };

    void runThemeKickoff();

    return () => {
      cancelled = true;
    };
  }, [
    themeMessages,
    themeKickoffIds,
    messages,
    setMessages,
    markThemeKickoff,
    markKickoffAssistantMessage,
    sessionId,
  ]);

  const handleSubmit = (event?: { preventDefault?: () => void }) => {
    event?.preventDefault?.();
    const text = input.trim();
    if (!text || isLoading || themeKickoffLoading) return;
    void sendMessage({ text });
    setInput("");
  };

  const copyMessage = (content: string) => {
    if (navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(content);
    }
  };

  const generateFromMessage = async (messageId: string, content: string) => {
    setGeneratingFor(messageId);
    setGenerateError(null);
    try {
      const result = await api.generateImage({
        prompt: content,
        aspect_ratio: chatAspectRatio,
        session_id: `ai_chat_${sessionId}`,
      });
      const image = {
        id: generateId("img"),
        filename: result.filename,
        imageUrl: result.image_url,
        imageDataUrl: result.image_data_url,
        prompt: result.image_prompt?.trim() || content,
        aspectRatio: result.aspect_ratio || chatAspectRatio,
        createdAt: new Date().toISOString(),
        kind: "generated" as const,
      };
      addImage(sessionId, image);
      addToGallery(image);
    } catch (err) {
      setGenerateError(
        err instanceof Error ? err.message : "Image generation failed"
      );
    } finally {
      setGeneratingFor(null);
    }
  };

  const hasContent =
    themeMessages.length > 0 || messages.length > 0 || themeKickoffLoading;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ScrollArea className="flex-1 px-4 py-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {!hasContent && (
            <Alert className="border-dashed">
              Your conversation will appear here. Type a question below to start.
            </Alert>
          )}

          {themeMessages.map((themeMsg) => (
            <Card
              key={themeMsg.id}
              className="border-amber-200/60 bg-amber-50/40 p-4 dark:border-amber-900/40 dark:bg-amber-950/20"
            >
              <p className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                Theme · {themeMsg.themeLabel || themeMsg.themeId || "Custom"}
              </p>
              <p className="whitespace-pre-wrap text-sm">{themeMsg.content}</p>
              <p className="mt-3 text-xs text-muted-foreground">
                This text is sent to the model as the system prompt for this chat until you
                pick another theme or clear messages.
              </p>
            </Card>
          ))}

          {messages.map((msg) => {
            const rawContent = getMessageText(msg);
            const content =
              msg.role === "assistant" &&
              systemPromptOverride &&
              !kickoffAssistantMessageIds.includes(msg.id)
                ? stripThemeChatFiller(rawContent)
                : rawContent;
            const isKickoffReply = kickoffAssistantMessageIds.includes(msg.id);
            const isGeneratingThis = generatingFor === msg.id;

            return (
              <Card
                key={msg.id}
                className={`p-4 ${msg.role === "user" ? "ml-8 bg-primary/5" : "mr-8"}`}
              >
                <p className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                  {msg.role === "user" ? "You" : "Assistant"}
                </p>
                <p className="whitespace-pre-wrap text-sm">{content}</p>
                {msg.role === "assistant" && content && !isKickoffReply && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={isLoading || !!generatingFor}
                      onClick={() => void generateFromMessage(msg.id, content)}
                    >
                      {isGeneratingThis ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Sparkles className="h-4 w-4" />
                      )}
                      {isGeneratingThis ? "Generating…" : "Generate image from this"}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => copyMessage(content)}
                    >
                      <Copy className="h-4 w-4" />
                      Copy
                    </Button>
                  </div>
                )}
              </Card>
            );
          })}

          {(isLoading || themeKickoffLoading) && (
            <p className="text-sm text-muted-foreground">
              {themeKickoffLoading ? "Applying theme…" : "Streaming…"}
            </p>
          )}
        </div>
      </ScrollArea>

      {error && (
        <Alert className="mx-4 mb-2 border-destructive/50 text-destructive">
          {error.message}
        </Alert>
      )}

      {themeKickoffError && (
        <Alert className="mx-4 mb-2 border-destructive/50 text-destructive">
          Error: {themeKickoffError}
        </Alert>
      )}

      {generateError && (
        <Alert className="mx-4 mb-2 border-destructive/50 text-destructive">
          {generateError}
        </Alert>
      )}

      <div className="border-t px-4 py-3">
        <form className="mx-auto flex max-w-3xl items-end gap-2" onSubmit={handleSubmit}>
          <Popover>
            <PopoverTrigger asChild>
              <button
                type="button"
                title="Aspect ratio"
                className="flex h-[52px] shrink-0 items-center gap-1.5 rounded-md border px-2.5 text-xs text-muted-foreground transition hover:bg-muted/60 hover:text-foreground"
              >
                <RatioGlyph ratio={chatAspectRatio} />
                <span className="font-medium text-foreground">{chatAspectRatio}</span>
                <ChevronDown className="h-3.5 w-3.5 opacity-60" />
              </button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-64 p-2">
              <p className="px-1 pb-2 text-[11px] text-muted-foreground">
                Used for generation, edits, accurate &amp; refined-prompt passes.
              </p>
              <div className="grid grid-cols-2 gap-1">
                {CHAT_ASPECT_RATIOS.map((option) => {
                  const active = option.value === chatAspectRatio;
                  const name = option.label.split(" — ")[1] ?? "";
                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setChatAspectRatio(option.value)}
                      aria-pressed={active}
                      title={option.label}
                      className={`flex items-center gap-2 rounded-md border px-2 py-1.5 text-xs transition ${
                        active
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-transparent text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                      }`}
                    >
                      <RatioGlyph ratio={option.value} />
                      <span className="flex min-w-0 flex-col items-start leading-tight">
                        <span className="font-medium">{option.value}</span>
                        <span className="truncate text-[10px] opacity-70">{name}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </PopoverContent>
          </Popover>

          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about anatomy, imaging, or illustration ideas…"
            className="min-h-[52px] resize-none"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e as unknown as React.FormEvent);
              }
            }}
          />
          <Button
            type="submit"
            size="icon"
            className="h-[52px] w-[52px] shrink-0"
            disabled={isLoading || themeKickoffLoading}
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}
