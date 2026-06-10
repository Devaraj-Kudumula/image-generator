/** Mirrors prompts.py → AI_CHAT_SYSTEM (word-for-word). */
export const AI_CHAT_SYSTEM =
  "You are a careful, expert assistant helping users think through medical and " +
  "scientific illustration, anatomy, imaging, and related topics.\n\n" +
  "Behavior:\n" +
  "• Use the full conversation so far: resolve pronouns, follow up on earlier " +
  "constraints, and do not contradict what the user already established unless " +
  "you flag a correction clearly.\n" +
  "• Prefer accuracy over brevity. Give thorough answers with clear structure " +
  "(short sections, bullet lists where helpful, numbered steps when describing " +
  "a process). Default to substantive detail; avoid empty filler.\n" +
  "• When the topic is clinical or anatomical, be precise about terminology, " +
  "laterality, orientation, and common imaging/plane conventions. If something is " +
  "uncertain or guideline-dependent, say so and outline reasonable options.\n" +
  "• Use GitHub-flavored Markdown when it improves readability (headings, lists, " +
  "`code` for short literals). Do not wrap the entire reply in one code block.\n" +
  "• You only output text. Do not claim to have generated or attached images; " +
  "the product may generate images separately from your text.\n" +
  "• Do not invent citations, paper titles, or guideline quotes. If retrieval " +
  "would be needed for a definitive answer, explain what to verify and where.\n" +
  "• Stay helpful and direct. Match the user's tone; be concise in short " +
  'exchanges and expansive when they ask for depth or "explain in detail".';

export const AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS = 12000;

/** Appended when a theme override replaces AI_CHAT_SYSTEM (prompt-engineer mode). */
export const AI_CHAT_THEME_REPLY_RULES =
  "\n\nRESPONSE RULES (CRITICAL):\n" +
  "When the user gives a topic, condition, organ, pathology, or imaging study, your entire reply " +
  "must be ONLY the image-generation prompt in the OUTPUT FORMAT specified above.\n" +
  "Do not add preamble, meta-commentary, closing remarks, follow-up questions, or offers of " +
  "further help (e.g. never ask if they want help with something else).\n" +
  "The reply is sent directly to an image generator — any conversational text breaks the pipeline.";

export function buildThemeSystemPrompt(themePrompt: string): string {
  const base = themePrompt.trim();
  if (!base) return base;
  const combined = `${base}${AI_CHAT_THEME_REPLY_RULES}`;
  if (combined.length > AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS) {
    return combined.slice(0, AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS);
  }
  return combined;
}

export function conversationModel(): string {
  return process.env.OPENAI_CONVERSATION_MODEL || "gpt-5.5";
}

export function conversationTemperature(): number {
  const raw = process.env.OPENAI_CONVERSATION_TEMPERATURE;
  const parsed = raw ? Number.parseFloat(raw) : 0.7;
  return Number.isFinite(parsed) ? parsed : 0.7;
}

export type ChatHistoryEntry = { role: "user" | "assistant"; content: string };

export function normalizeHistory(history: unknown): ChatHistoryEntry[] {
  if (!Array.isArray(history)) return [];
  const out: ChatHistoryEntry[] = [];
  for (const entry of history) {
    if (!entry || typeof entry !== "object") continue;
    const role = (entry as { role?: string }).role;
    const content = (entry as { content?: string }).content;
    if (
      (role === "user" || role === "assistant") &&
      typeof content === "string" &&
      content.trim()
    ) {
      out.push({ role, content });
    }
  }
  return out;
}

export function conversationMaxContextTokens(): number {
  const raw = process.env.OPENAI_CONVERSATION_MAX_CONTEXT_TOKENS;
  const parsed = raw ? Number.parseInt(raw, 10) : 200_000;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 200_000;
}

function estimateMessageTokens(
  messages: Array<{ role: string; content: string }>
): number {
  const perMessageOverhead = 4;
  let total = 0;
  for (const msg of messages) {
    const text = msg.content || "";
    total += Math.max(1, Math.ceil(text.length / 4)) + perMessageOverhead;
  }
  return total;
}

/** Mirrors Flask ai_chat_routes._build_messages_with_context_cap history trimming. */
export function trimHistoryForContextCap(
  systemText: string,
  historyEntries: ChatHistoryEntry[],
  userMessage: string,
  maxContextTokens = conversationMaxContextTokens()
): { history: ChatHistoryEntry[]; trimmedPairs: number; estimatedInputTokens: number } {
  const trimmedUser = userMessage.trim();
  if (!trimmedUser) {
    return { history: [], trimmedPairs: historyEntries.length, estimatedInputTokens: 0 };
  }

  const systemMsg = { role: "system", content: systemText };
  const userMsg = { role: "user", content: trimmedUser };
  const fixedTokens = estimateMessageTokens([systemMsg, userMsg]);
  let historyBudget = maxContextTokens - fixedTokens - 16;
  if (historyBudget < 0) historyBudget = 0;

  let hist = [...historyEntries];
  let trimmedPairs = 0;

  const historyTokenCount = () => estimateMessageTokens(hist);

  while (historyTokenCount() > historyBudget) {
    if (hist.length >= 2) {
      hist = hist.slice(2);
      trimmedPairs += 1;
      continue;
    }
    if (hist.length > 0) {
      hist = hist.slice(1);
      continue;
    }
    break;
  }

  const messages = [systemMsg, ...hist, userMsg];
  return {
    history: hist,
    trimmedPairs,
    estimatedInputTokens: estimateMessageTokens(messages),
  };
}
