import { convertToModelMessages, streamText, type UIMessage } from "ai";
import { openai } from "@ai-sdk/openai";
import {
  AI_CHAT_SYSTEM,
  buildThemeSystemPrompt,
  conversationMaxContextTokens,
  conversationModel,
  conversationTemperature,
  normalizeHistory,
  trimHistoryForContextCap,
  type ChatHistoryEntry,
} from "@/lib/server/ai-chat";

export const maxDuration = 180;

function getMessageText(message: UIMessage): string {
  return (message.parts ?? [])
    .filter((part) => part.type === "text")
    .map((part) => ("text" in part ? part.text : "") ?? "")
    .join("");
}

function trimUiMessages(
  messages: UIMessage[],
  systemText: string
): UIMessage[] {
  if (!messages.length) return messages;

  const historyEntries: ChatHistoryEntry[] = [];
  for (let i = 0; i < messages.length - 1; i += 1) {
    const msg = messages[i];
    if (msg.role !== "user" && msg.role !== "assistant") continue;
    const content = getMessageText(msg).trim();
    if (!content) continue;
    historyEntries.push({ role: msg.role, content });
  }

  const last = messages[messages.length - 1];
  const lastContent = getMessageText(last).trim();
  if (!lastContent || last.role !== "user") {
    return messages;
  }

  const { history } = trimHistoryForContextCap(
    systemText,
    historyEntries,
    lastContent,
    conversationMaxContextTokens()
  );

  const trimmedMessages: UIMessage[] = [];
  let historyIndex = 0;
  for (let i = 0; i < messages.length - 1; i += 1) {
    const msg = messages[i];
    if (msg.role !== "user" && msg.role !== "assistant") {
      trimmedMessages.push(msg);
      continue;
    }
    const content = getMessageText(msg).trim();
    if (!content) continue;
    const nextHistory = history[historyIndex];
    if (
      nextHistory &&
      nextHistory.role === msg.role &&
      nextHistory.content === content
    ) {
      trimmedMessages.push(msg);
      historyIndex += 1;
    }
  }

  trimmedMessages.push(last);
  return trimmedMessages;
}

export async function POST(req: Request) {
  const { messages, systemPromptOverride } = await req.json();
  const override =
    typeof systemPromptOverride === "string" ? systemPromptOverride.trim() : "";
  const systemText = override ? buildThemeSystemPrompt(override) : AI_CHAT_SYSTEM;

  const uiMessages = Array.isArray(messages) ? (messages as UIMessage[]) : [];
  const trimmedMessages = trimUiMessages(uiMessages, systemText);

  // Validate history shape (same filter as non-streaming route)
  normalizeHistory(
    trimmedMessages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: getMessageText(m) }))
  );

  const result = streamText({
    model: openai(conversationModel()),
    system: systemText,
    temperature: conversationTemperature(),
    messages: convertToModelMessages(trimmedMessages),
  });

  return result.toUIMessageStreamResponse();
}
