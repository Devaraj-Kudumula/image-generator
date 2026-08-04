export const ALLOWED_ASPECT_RATIOS = [
  "1:1",
  "4:3",
  "3:4",
  "16:9",
  "9:16",
  "3:2",
  "2:3",
  "21:9",
] as const;

export const DEFAULT_ASPECT_RATIO = "16:9";

export function normalizeAspectRatio(value?: string | null): string {
  if (!value) return DEFAULT_ASPECT_RATIO;
  const candidate = String(value).trim();
  return (ALLOWED_ASPECT_RATIOS as readonly string[]).includes(candidate)
    ? candidate
    : DEFAULT_ASPECT_RATIO;
}
