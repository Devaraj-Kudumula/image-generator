import type { ChatTheme } from "@/lib/api";
import themesData from "@/lib/chat-themes.json";

/** Built-in AI chat themes (mirrors prompts.py AI_CHAT_THEME_PROMPTS). */
export const BUILT_IN_THEMES: Record<string, ChatTheme> = themesData;

/** Preferred display order for the theme selector. */
export const THEME_ORDER = [
  "realistic",
  "general",
  "histology",
  "organ_images",
  "radiology",
] as const;

export function mergeChatThemes(
  remote?: Record<string, ChatTheme>
): Record<string, ChatTheme> {
  return { ...BUILT_IN_THEMES, ...remote };
}
