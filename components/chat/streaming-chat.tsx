"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useEffect, useMemo, useState } from "react";
import { Loader2, Send, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Alert } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { ImageCard } from "@/components/generation/result-grid";
import { api } from "@/lib/api";
import { generateId } from "@/lib/utils";
import { useChatStore, useGalleryStore } from "@/lib/store/generation-store";

interface StreamingChatProps {
  sessionId: string;
  systemPromptOverride?: string;
}

function getMessageText(message: { parts: Array<{ type: string; text?: string }> }) {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("");
}

export function StreamingChat({ sessionId, systemPromptOverride }: StreamingChatProps) {
  const addImage = useChatStore((s) => s.addImage);
  const setUiMessages = useChatStore((s) => s.setUiMessages);
  const sessionImages = useChatStore(
    (s) => s.sessions.find((session) => session.id === sessionId)?.images ?? []
  );
  const addToGallery = useGalleryStore((s) => s.addImage);
  const [input, setInput] = useState("");
  const [generatingFor, setGeneratingFor] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const imagesByMessage = useMemo(() => {
    const map = new Map<string, typeof sessionImages>();
    const orphans: typeof sessionImages = [];

    for (const image of sessionImages) {
      if (image.messageId) {
        const existing = map.get(image.messageId) ?? [];
        map.set(image.messageId, [...existing, image]);
      } else {
        orphans.push(image);
      }
    }

    return { map, orphans };
  }, [sessionImages]);

  const initialMessages = useMemo(
    () =>
      useChatStore
        .getState()
        .sessions.find((session) => session.id === sessionId)?.uiMessages ?? [],
    [sessionId]
  );

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
        body: { systemPromptOverride },
      }),
    [systemPromptOverride]
  );

  const { messages, sendMessage, status, error } = useChat({
    id: sessionId,
    messages: initialMessages,
    transport,
  });

  const isLoading = status === "submitted" || status === "streaming";

  useEffect(() => {
    setInput("");
    setGenerateError(null);
    setGeneratingFor(null);
  }, [sessionId]);

  useEffect(() => {
    setUiMessages(sessionId, messages as UIMessage[]);
  }, [messages, sessionId, setUiMessages]);

  const handleSubmit = (event?: { preventDefault?: () => void }) => {
    event?.preventDefault?.();
    const text = input.trim();
    if (!text || isLoading) return;
    void sendMessage({ text });
    setInput("");
  };

  const generateFromMessage = async (messageId: string, content: string) => {
    setGeneratingFor(messageId);
    setGenerateError(null);
    try {
      const result = await api.generateImage({ prompt: content });
      const image = {
        id: generateId("img"),
        filename: result.filename,
        imageUrl: result.image_url,
        imageDataUrl: result.image_data_url,
        prompt: content,
        createdAt: new Date().toISOString(),
        kind: "generated" as const,
      };
      addImage(sessionId, { ...image, messageId });
      addToGallery(image);
    } catch (err) {
      setGenerateError(
        err instanceof Error ? err.message : "Image generation failed"
      );
    } finally {
      setGeneratingFor(null);
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ScrollArea className="flex-1 px-4 py-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {!messages.length && (
            <Alert className="border-dashed">
              Start a conversation to brainstorm medical illustration prompts. Responses stream in real time.
            </Alert>
          )}
          {messages.map((msg) => {
            const content = getMessageText(msg);
            const isGeneratingThis = generatingFor === msg.id;
            const messageImages = imagesByMessage.map.get(msg.id) ?? [];

            return (
              <div key={msg.id} className="space-y-3">
                <Card
                  className={`p-4 ${msg.role === "user" ? "ml-8 bg-primary/5" : "mr-8"}`}
                >
                  <p className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                    {msg.role === "user" ? "You" : "Assistant"}
                  </p>
                  <p className="whitespace-pre-wrap text-sm">{content}</p>
                  {msg.role === "assistant" && content && (
                    <Button
                      size="sm"
                      className="mt-3"
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
                  )}
                </Card>

                {msg.role === "assistant" && isGeneratingThis && (
                  <Card className="mr-8 max-w-sm overflow-hidden">
                    <Skeleton className="aspect-square w-full" />
                    <div className="space-y-2 p-3">
                      <Skeleton className="h-3 w-3/4" />
                      <p className="text-xs text-muted-foreground">Generating image…</p>
                    </div>
                  </Card>
                )}

                {msg.role === "assistant" && messageImages.length > 0 && (
                  <div className="mr-8 grid gap-4 sm:grid-cols-2">
                    {messageImages.map((image) => (
                      <ImageCard
                        key={image.id}
                        image={image}
                        className="max-w-sm"
                        inlineActions
                        onImageCreated={(entry) =>
                          addImage(sessionId, { ...entry, messageId: msg.id })
                        }
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {imagesByMessage.orphans.length > 0 && (
            <div className="space-y-3">
              <p className="text-xs font-medium uppercase text-muted-foreground">
                Generated images
              </p>
              <div className="grid gap-4 sm:grid-cols-2">
                {imagesByMessage.orphans.map((image) => (
                  <ImageCard
                    key={image.id}
                    image={image}
                    className="max-w-sm"
                    inlineActions
                  />
                ))}
              </div>
            </div>
          )}
          {isLoading && <p className="text-sm text-muted-foreground">Streaming…</p>}
        </div>
      </ScrollArea>

      {error && (
        <Alert className="mx-4 mb-2 border-destructive/50 text-destructive">
          {error.message}
        </Alert>
      )}

      {generateError && (
        <Alert className="mx-4 mb-2 border-destructive/50 text-destructive">
          {generateError}
        </Alert>
      )}

      <div className="border-t p-4">
        <form
          className="mx-auto flex max-w-3xl gap-2"
          onSubmit={handleSubmit}
        >
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
            disabled={isLoading}
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}
