/** Mirrors static/ai_chat.js → themeKickoffUserMessage (word-for-word). */
export function themeKickoffUserMessage(label: string): string {
  return (
    'The user selected the "' +
    label +
    '" theme; the full system instructions were shown in the chat. ' +
    "Reply in one or two short sentences that you will follow those instructions for this conversation, " +
    "then invite their next message."
  );
}

export function buildHistoryForApi(
  messages: Array<{ role: string; content: string }>
): Array<{ role: "user" | "assistant"; content: string }> {
  return messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role as "user" | "assistant", content: m.content }));
}
