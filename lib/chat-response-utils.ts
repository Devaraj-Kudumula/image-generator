const CONVERSATIONAL_TAIL_PATTERNS = [
  /\n+---[\s\S]*$/,
  /\n+(?:Would you like|Let me know|Feel free to|If you(?:'d| would) like|I can also|Happy to help|Anything else|Do you want|Shall I|Need anything)[\s\S]*$/i,
];

/** Strip trailing conversational filler from theme-mode assistant replies. */
export function stripThemeChatFiller(text: string): string {
  let result = text.trim();
  let changed = true;
  while (changed) {
    changed = false;
    for (const pattern of CONVERSATIONAL_TAIL_PATTERNS) {
      const next = result.replace(pattern, "").trim();
      if (next !== result) {
        result = next;
        changed = true;
      }
    }
  }
  return result;
}
