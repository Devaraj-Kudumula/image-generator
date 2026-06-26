"use client";

import { useEffect, useMemo, useState } from "react";
import { Plus, Pencil, Trash2 } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert } from "@/components/ui/alert";
import { StreamingChat } from "@/components/chat/streaming-chat";
import { ChatImagePanel } from "@/components/chat/chat-image-panel";
import { useChatStore } from "@/lib/store/generation-store";
import { api, type ChatTheme } from "@/lib/api";
import { BUILT_IN_THEMES, mergeChatThemes, THEME_ORDER } from "@/lib/chat-themes";

function getSessionSubtitle(
  uiMessages: Array<{ role: string; parts?: Array<{ type: string; text?: string }> }>
): string {
  for (let i = uiMessages.length - 1; i >= 0; i -= 1) {
    const msg = uiMessages[i];
    if (msg.role !== "user") continue;
    const text = (msg.parts ?? [])
      .filter((part) => part.type === "text")
      .map((part) => part.text ?? "")
      .join("")
      .trim();
    if (text) {
      return text.length > 60 ? `${text.slice(0, 60)}…` : text;
    }
  }
  return "No messages yet";
}

export default function ChatPage() {
  const {
    sessions,
    activeSessionId,
    createSession,
    setActiveSession,
    deleteSession,
    renameSession,
    setTheme,
    clearTheme,
    clearSession,
  } = useChatStore();
  const [themes, setThemes] = useState<Record<string, ChatTheme>>(BUILT_IN_THEMES);
  const [themeError, setThemeError] = useState<string | null>(null);
  const [chatResetKey, setChatResetKey] = useState(0);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");

  const startRename = (id: string, currentName: string) => {
    setEditingId(id);
    setEditingName(currentName);
  };

  const commitRename = () => {
    if (editingId && editingName.trim()) {
      renameSession(editingId, editingName.trim());
    }
    setEditingId(null);
  };

  const cancelRename = () => setEditingId(null);

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeSessionId) ?? sessions[0],
    [sessions, activeSessionId]
  );

  useEffect(() => {
    if (!activeSessionId && sessions[0]) {
      setActiveSession(sessions[0].id);
    }
  }, [activeSessionId, sessions, setActiveSession]);

  useEffect(() => {
    void api
      .getChatThemes()
      .then((res) => setThemes(mergeChatThemes(res.themes)))
      .catch(() => setThemes(BUILT_IN_THEMES));
  }, []);

  const systemOverride = [...(activeSession?.messages ?? [])]
    .reverse()
    .find((m) => m.role === "theme")?.content;

  const orderedThemeIds = useMemo(() => {
    const keys: string[] = [];
    for (const id of THEME_ORDER) {
      if (themes[id]) keys.push(id);
    }
    for (const id of Object.keys(themes).sort()) {
      if (!keys.includes(id)) keys.push(id);
    }
    return keys;
  }, [themes]);

  const messageCount = activeSession?.uiMessages.length ?? 0;
  const imageCount = activeSession?.images.length ?? 0;

  const handleClearChat = () => {
    if (!activeSession) return;
    if (!messageCount && !imageCount) return;
    if (
      !confirm("Clear all messages and images in this chat?")
    ) {
      return;
    }
    clearSession(activeSession.id);
    setChatResetKey((key) => key + 1);
  };

  return (
    <AppShell>
      <div className="flex h-[calc(100vh-3.5rem)]">
        <aside className="hidden w-64 shrink-0 border-r bg-sidebar md:flex md:flex-col">
          <div className="flex items-center justify-between border-b p-3">
            <p className="text-sm font-medium">
              Chats
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                {sessions.length === 1 ? "1 session" : `${sessions.length} sessions`}
              </span>
            </p>
            <Button size="icon" variant="ghost" className="h-8 w-8" onClick={createSession}>
              <Plus className="h-4 w-4" />
            </Button>
          </div>
          <ScrollArea className="flex-1 p-2">
            {sessions.map((session) => {
              const subtitle = getSessionSubtitle(session.uiMessages);
              const isActive = session.id === activeSession?.id;
              const isEditing = editingId === session.id;
              return (
                <div
                  key={session.id}
                  className={`group mb-1 flex items-center gap-1 rounded-lg pr-1 ${
                    isActive ? "bg-sidebar-accent" : "hover:bg-sidebar-accent"
                  }`}
                >
                  {isEditing ? (
                    <input
                      autoFocus
                      value={editingName}
                      onChange={(e) => setEditingName(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitRename();
                        if (e.key === "Escape") cancelRename();
                      }}
                      className="m-1 w-full rounded-md border bg-background px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-ring"
                    />
                  ) : (
                    <>
                      <button
                        type="button"
                        onClick={() => setActiveSession(session.id)}
                        className={`min-w-0 flex-1 rounded-lg px-3 py-2 text-left text-sm ${
                          isActive ? "font-medium" : "text-muted-foreground"
                        }`}
                      >
                        <span className="block truncate">{session.name}</span>
                        <span className="block truncate text-xs text-muted-foreground">
                          {subtitle}
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          startRename(session.id, session.name);
                        }}
                        aria-label="Rename chat"
                        title="Rename chat"
                        className="shrink-0 rounded-md p-1.5 text-muted-foreground opacity-0 transition hover:bg-background hover:text-foreground group-hover:opacity-100"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm(`Delete "${session.name}"?`)) {
                            deleteSession(session.id);
                          }
                        }}
                        aria-label="Delete chat"
                        title="Delete chat"
                        className="shrink-0 rounded-md p-1.5 text-muted-foreground opacity-0 transition hover:bg-background hover:text-destructive group-hover:opacity-100"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </>
                  )}
                </div>
              );
            })}
          </ScrollArea>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col lg:flex-row">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col lg:border-r">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b px-4 py-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">
                  {activeSession?.name ?? "Chat"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {messageCount === 0 && imageCount === 0
                    ? "Type a question below to start."
                    : `${messageCount} message${messageCount !== 1 ? "s" : ""} · ${imageCount} image${imageCount !== 1 ? "s" : ""}`}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Select
                  value={activeSession?.themeId ?? "none"}
                  onValueChange={(themeId) => {
                    if (!activeSession) return;
                    setThemeError(null);
                    if (themeId === "none") {
                      clearTheme(activeSession.id);
                      return;
                    }
                    const theme = themes[themeId];
                    if (!theme) return;
                    const prompt = (theme.prompt || "").trim();
                    if (!prompt) {
                      setThemeError(
                        "This theme has no prompt text yet. Edit lib/chat-themes.json."
                      );
                      return;
                    }
                    setTheme(activeSession.id, themeId, theme.label, prompt);
                  }}
                >
                  <SelectTrigger className="w-[220px]">
                    <SelectValue placeholder="Select theme" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">No theme</SelectItem>
                    {orderedThemeIds.map((id) => (
                      <SelectItem key={id} value={id}>
                        {themes[id].label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {activeSession && (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleClearChat}
                    >
                      Clear messages
                    </Button>
                  </>
                )}
              </div>
            </div>

            {themeError && (
              <Alert className="mx-4 mt-3 border-destructive/50 text-destructive">
                {themeError}
              </Alert>
            )}

            <div className="flex min-h-0 flex-1 flex-col">
              {activeSession && (
                <StreamingChat
                  key={`${activeSession.id}-${chatResetKey}`}
                  sessionId={activeSession.id}
                  systemPromptOverride={systemOverride}
                />
              )}
            </div>
          </div>

          {activeSession && <ChatImagePanel sessionId={activeSession.id} />}
        </div>
      </div>
    </AppShell>
  );
}
