export function imageBytesToDataUrl(imageBytes: Buffer): string {
  return `data:image/png;base64,${imageBytes.toString("base64")}`;
}

export function decodeImageDataUrl(imageDataUrl: string): Buffer {
  if (!imageDataUrl?.trim()) {
    throw new Error("image_data_url is empty");
  }

  let normalized = imageDataUrl.trim();
  if (normalized.startsWith("data:")) {
    const comma = normalized.indexOf(",");
    if (comma === -1) throw new Error("image_data_url is malformed");
    normalized = normalized.slice(comma + 1);
  }

  return Buffer.from(normalized, "base64");
}

type GeminiInlineData = {
  data?: string | Uint8Array | ArrayBuffer;
  mimeType?: string;
  mime_type?: string;
};

type GeminiPart = {
  text?: string;
  inlineData?: GeminiInlineData;
  inline_data?: GeminiInlineData;
};

interface GeminiResponse {
  candidates?: Array<{
    content?: { parts?: GeminiPart[] };
    finishReason?: string;
    finish_reason?: string;
  }>;
  promptFeedback?: {
    blockReason?: string;
    block_reason?: string;
    blockReasonMessage?: string;
    block_reason_message?: string;
  };
}

function getInlineData(part: GeminiPart): GeminiInlineData | null {
  const inline = part.inlineData ?? part.inline_data;
  return inline && typeof inline === "object" ? inline : null;
}

function inlineDataToBuffer(inline: GeminiInlineData): Buffer | null {
  const data = inline.data;
  if (!data) return null;

  if (typeof data === "string") {
    return Buffer.from(data, "base64");
  }
  if (data instanceof ArrayBuffer) {
    return Buffer.from(data);
  }
  if (ArrayBuffer.isView(data)) {
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  }
  return null;
}

export function extractPngBytesFromGeminiResponse(
  response: GeminiResponse
): Buffer | null {
  for (const candidate of response.candidates ?? []) {
    const parts = candidate.content?.parts;
    if (!parts?.length) continue;

    for (const part of parts) {
      const inline = getInlineData(part);
      if (!inline) continue;
      const bytes = inlineDataToBuffer(inline);
      if (bytes?.length) return bytes;
    }
  }

  return null;
}

export function getGeminiFinishReason(response: GeminiResponse): string | undefined {
  const candidate = response.candidates?.[0];
  return candidate?.finishReason ?? candidate?.finish_reason;
}

export function shouldRetryGeminiImageGeneration(
  response: GeminiResponse
): boolean {
  const finishReason = getGeminiFinishReason(response);
  if (!finishReason) return true;
  return finishReason === "NO_IMAGE" || finishReason === "IMAGE_OTHER";
}

export function describeGeminiImageFailure(response: GeminiResponse): string {
  const feedback = response.promptFeedback;
  const blockReason = feedback?.blockReason ?? feedback?.block_reason;
  if (blockReason) {
    const detail =
      feedback?.blockReasonMessage ?? feedback?.block_reason_message;
    return detail
      ? `Image request blocked (${blockReason}): ${detail}`
      : `Image request blocked (${blockReason})`;
  }

  const candidate = response.candidates?.[0];
  if (!candidate) {
    return "No image generated in response (empty candidates)";
  }

  const finishReason = candidate.finishReason ?? candidate.finish_reason;
  if (finishReason && finishReason !== "STOP") {
    return `No image generated in response (finish reason: ${finishReason})`;
  }

  const parts = candidate.content?.parts ?? [];
  const textParts = parts
    .map((part) => part.text?.trim())
    .filter((text): text is string => Boolean(text));
  if (textParts.length) {
    const preview = textParts.join(" ").slice(0, 160);
    return `No image generated in response. Model returned text instead: ${preview}${
      preview.length >= 160 ? "…" : ""
    }`;
  }

  return "No image generated in response";
}

export function timestampFilename(prefix: string): string {
  const stamp = new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\..+/, "")
    .replace("T", "_");
  return `${prefix}_${stamp}.png`;
}
