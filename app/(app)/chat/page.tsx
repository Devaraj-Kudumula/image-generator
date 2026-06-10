"use client";

import { useEffect, useMemo, useState } from "react";
import { Plus } from "lucide-react";
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
import { StreamingChat } from "@/components/chat/streaming-chat";
import { useChatStore } from "@/lib/store/generation-store";
import { api, type ChatTheme } from "@/lib/api";
import { BUILT_IN_THEMES, mergeChatThemes } from "@/lib/chat-themes";

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
  } = useChatStore();
  const [themes, setThemes] = useState<Record<string, ChatTheme>>(BUILT_IN_THEMES);

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

  const systemOverride = activeSession?.messages.find((m) => m.role === "theme")?.content;

  return (
    <AppShell>
      <div className="flex h-[calc(100vh-3.5rem)]">
        <aside className="hidden w-64 shrink-0 border-r bg-sidebar md:flex md:flex-col">
          <div className="flex items-center justify-between border-b p-3">
            <p className="text-sm font-medium">Chats</p>
            <Button size="icon" variant="ghost" className="h-8 w-8" onClick={createSession}>
              <Plus className="h-4 w-4" />
            </Button>
          </div>
          <ScrollArea className="flex-1 p-2">
            {sessions.map((session) => (
              <button
                key={session.id}
                type="button"
                onClick={() => setActiveSession(session.id)}
                className={`mb-1 w-full rounded-lg px-3 py-2 text-left text-sm ${
                  session.id === activeSession?.id
                    ? "bg-sidebar-accent font-medium"
                    : "text-muted-foreground hover:bg-sidebar-accent"
                }`}
              >
                {session.name}
              </button>
            ))}
          </ScrollArea>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3">
            <Select
              value={activeSession?.themeId ?? "none"}
              onValueChange={(themeId) => {
                if (!activeSession) return;
                if (themeId === "none") {
                  clearTheme(activeSession.id);
                  return;
                }
                const theme = themes[themeId];
                if (!theme) return;
                setTheme(activeSession.id, themeId, theme.label, theme.prompt);
              }}
            >
              <SelectTrigger className="w-[220px]">
                <SelectValue placeholder="Select theme" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">No theme</SelectItem>
                {Object.entries(themes).map(([id, theme]) => (
                  <SelectItem key={id} value={id}>
                    {theme.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {activeSession && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const name = prompt("Rename chat", activeSession.name);
                    if (name?.trim()) renameSession(activeSession.id, name.trim());
                  }}
                >
                  Rename
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => deleteSession(activeSession.id)}
                >
                  Delete
                </Button>
              </>
            )}
          </div>

          <div className="flex min-h-0 flex-1 flex-col">
            {activeSession && (
              <StreamingChat
                key={activeSession.id}
                sessionId={activeSession.id}
                systemPromptOverride={systemOverride}
              />
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
