const MAX_IMAGE_PROMPT_CHARS = 8000;

const STRUCTURED_SECTION_LABELS = [
  "Main Clinical Scenario Description",
  "Main illustration description",
  "Clinical study description",
  "Clinical/pathology scenario",
  "Main tissue description",
  "Image Type",
  "Composition",
  "Key Visual Findings",
  "Diagnostic pathology findings",
  "Pathology findings",
  "Environment Details",
  "Realism Constraints",
  "Staining and optical characteristics",
  "Imaging modality and sequence",
  "Realistic imaging characteristics",
  "Output Goal",
  "Output",
  "Style",
] as const;

function truncatePrompt(text: string, max = MAX_IMAGE_PROMPT_CHARS): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}\n\n[Truncated for image generation]`;
}

function stripMarkdown(text: string): string {
  return text
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/^[-*]\s+/gm, "")
    .trim();
}

export function extractStructuredImageSections(text: string): string | null {
  const sections: string[] = [];

  for (const label of STRUCTURED_SECTION_LABELS) {
    const regex = new RegExp(
      `${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}:\\s*([\\s\\S]*?)(?=\\n[A-Za-z][A-Za-z0-9 /()]+:|$)`,
      "i"
    );
    const match = text.match(regex);
    const value = match?.[1]?.trim();
    if (value) sections.push(`${label}: ${value}`);
  }

  if (sections.length >= 2) {
    return sections.join("\n\n");
  }
  return null;
}

export function buildConciseImagePrompt(text: string): string {
  const priorityLabels = [
    "Output Goal",
    "Output",
    "Key Visual Findings",
    "Composition",
    "Main Clinical Scenario Description",
    "Main illustration description",
    "Clinical study description",
  ];

  const parts: string[] = [];
  for (const label of priorityLabels) {
    const regex = new RegExp(
      `${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}:\\s*([\\s\\S]*?)(?=\\n[A-Za-z][A-Za-z0-9 /()]+:|$)`,
      "i"
    );
    const match = text.match(regex);
    const value = match?.[1]?.trim();
    if (value) parts.push(value);
  }

  const body = parts.length ? parts.join(". ") : stripMarkdown(text).slice(0, 3000);
  return (
    "Generate one medical illustration image. " +
    "Create a single clear image matching this description:\n\n" +
    truncatePrompt(body, 4000)
  );
}

/** Turn AI chat assistant text into an explicit Gemini image-generation prompt. */
export function prepareImagePromptFromChat(content: string): string {
  const text = stripMarkdown(content.trim());
  if (!text) return text;

  const structured = extractStructuredImageSections(text);
  const body = structured || text;

  return truncatePrompt(
    "Generate a single medical illustration image. " +
      "Render exactly one image that matches every visual specification below. " +
      "Do not include text labels unless the description explicitly requires them.\n\n" +
      body
  );
}

export function imagePromptVariants(rawPrompt: string): string[] {
  const base = rawPrompt.trim();
  if (!base) return [];

  const variants = [
    base,
    prepareImagePromptFromChat(base),
    buildConciseImagePrompt(base),
  ];

  const seen = new Set<string>();
  return variants.filter((prompt) => {
    if (!prompt || seen.has(prompt)) return false;
    seen.add(prompt);
    return true;
  });
}
