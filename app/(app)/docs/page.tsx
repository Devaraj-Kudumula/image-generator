"use client";

import { useEffect, useRef, useState } from "react";
import { Upload, Send, RotateCcw } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { getClientSessionId } from "@/lib/utils";

const NO_RAG_OPTION_VALUE = "NO_RAG";
const WEB_RETRIEVAL_OPTION_VALUE = "WEB_RETRIEVAL";

interface DocMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export default function DocsPage() {
  const [sessionId, setSessionId] = useState("");
  const [docNames, setDocNames] = useState<string[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [messages, setMessages] = useState<DocMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [disabled, setDisabled] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const id = getClientSessionId();
    setSessionId(id);
    void refreshDocs(id);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const refreshDocs = async (id: string) => {
    try {
      const res = await api.getDocNames(id);
      setDocNames(res.doc_names ?? []);
      setDisabled(!!res.disabled);
      if (res.doc_names?.length) {
        setSelectedDocs(res.doc_names);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !sessionId) return;
    setLoading(true);
    setError(null);
    try {
      await api.uploadDoc(file, sessionId);
      await refreshDocs(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const sendQuestion = async () => {
    const question = input.trim();
    if (!question || !sessionId) return;
    setInput("");
    setLoading(true);
    setError(null);

    const userMsg: DocMessage = {
      id: `u_${Date.now()}`,
      role: "user",
      content: question,
    };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const chatHistory = messages
        .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
        .join("\n");

      const res = await api.chatWithDocs({
        user_question: question,
        selected_doc_names: selectedDocs,
        chat_history: chatHistory,
        session_id: sessionId,
      });

      setMessages((prev) => [
        ...prev,
        { id: `a_${Date.now()}`, role: "assistant", content: res.answer },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat failed");
    } finally {
      setLoading(false);
    }
  };

  const toggleDoc = (name: string) => {
    if (name === NO_RAG_OPTION_VALUE) {
      setSelectedDocs((prev) =>
        prev.includes(NO_RAG_OPTION_VALUE) ? [] : [NO_RAG_OPTION_VALUE]
      );
      return;
    }

    setSelectedDocs((prev) => {
      const withoutNoRag = prev.filter((d) => d !== NO_RAG_OPTION_VALUE);
      if (name === WEB_RETRIEVAL_OPTION_VALUE) {
        return withoutNoRag.includes(WEB_RETRIEVAL_OPTION_VALUE)
          ? withoutNoRag.filter((d) => d !== WEB_RETRIEVAL_OPTION_VALUE)
          : [...withoutNoRag, WEB_RETRIEVAL_OPTION_VALUE];
      }
      return withoutNoRag.includes(name)
        ? withoutNoRag.filter((d) => d !== name)
        : [...withoutNoRag, name];
    });
  };

  const handleResetSession = async () => {
    if (!sessionId || disabled) return;
    setLoading(true);
    setError(null);
    try {
      await api.resetSession(sessionId);
      setMessages([]);
      await refreshDocs(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset session");
    } finally {
      setLoading(false);
    }
  };

  const noRagSelected = selectedDocs.includes(NO_RAG_OPTION_VALUE);
  const webRetrievalSelected = selectedDocs.includes(WEB_RETRIEVAL_OPTION_VALUE);

  return (
    <AppShell>
      <div className="mx-auto grid max-w-6xl gap-6 px-4 py-8 lg:grid-cols-[280px_1fr] md:px-6">
        <aside className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold">Docs Q&A</h2>
            <p className="text-sm text-muted-foreground">
              Upload PDFs and ask questions grounded in your documents
            </p>
          </div>

          <Card className="space-y-3 p-4">
            <input
              ref={fileRef}
              type="file"
              accept="application/pdf"
              className="hidden"
              onChange={(e) => void handleUpload(e)}
            />
            <Button
              variant="outline"
              className="w-full"
              onClick={() => fileRef.current?.click()}
              disabled={loading || disabled}
            >
              <Upload className="h-4 w-4" />
              Upload PDF
            </Button>

            <Button
              variant="outline"
              className="w-full"
              onClick={() => void handleResetSession()}
              disabled={loading || disabled}
            >
              <RotateCcw className="h-4 w-4" />
              Reset session
            </Button>

            {disabled && (
              <Alert className="text-xs">
                RAG is disabled. Configure MONGODB_URI and OPENAI_API_KEY to enable document chat.
              </Alert>
            )}

            <div className="max-h-64 space-y-2 overflow-auto pb-1">
              <div className="flex flex-wrap gap-2">
                <Badge
                  variant={noRagSelected ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => toggleDoc(NO_RAG_OPTION_VALUE)}
                >
                  Don&apos;t use my documents
                </Badge>
                <Badge
                  variant={webRetrievalSelected ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => toggleDoc(WEB_RETRIEVAL_OPTION_VALUE)}
                >
                  Include web search
                </Badge>
              </div>
              <div className="flex flex-col items-start gap-2">
                {docNames.map((name) => (
                  <Badge
                    key={name}
                    variant={
                      !noRagSelected && selectedDocs.includes(name)
                        ? "default"
                        : "outline"
                    }
                    title={name}
                    className={`max-w-none cursor-pointer whitespace-nowrap ${noRagSelected ? "opacity-50" : ""}`}
                    onClick={() => !noRagSelected && toggleDoc(name)}
                  >
                    {name}
                  </Badge>
                ))}
                {!docNames.length && !disabled && (
                  <p className="text-xs text-muted-foreground">No documents loaded</p>
                )}
              </div>
            </div>
          </Card>
        </aside>

        <div className="flex min-h-[70vh] flex-col rounded-xl border">
          <ScrollArea className="flex-1 p-4">
            <div className="mx-auto max-w-3xl space-y-4">
              {!messages.length && (
                <Alert className="border-dashed">
                  Select documents and ask a clinical or scientific question.
                </Alert>
              )}
              {messages.map((msg) => (
                <Card
                  key={msg.id}
                  className={`p-4 ${msg.role === "user" ? "ml-8 bg-primary/5" : "mr-8"}`}
                >
                  <p className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                    {msg.role === "user" ? "You" : "Assistant"}
                  </p>
                  <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
                </Card>
              ))}
              {loading && <p className="text-sm text-muted-foreground">Searching documents…</p>}
              <div ref={bottomRef} />
            </div>
          </ScrollArea>

          {error && (
            <Alert className="mx-4 mb-2 border-destructive/50 text-destructive">{error}</Alert>
          )}

          <div className="border-t p-4">
            <div className="flex gap-2">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about your documents…"
                className="min-h-[52px] resize-none"
                disabled={disabled}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void sendQuestion();
                  }
                }}
              />
              <Button
                size="icon"
                className="h-[52px] w-[52px] shrink-0"
                onClick={() => void sendQuestion()}
                disabled={loading || disabled}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
