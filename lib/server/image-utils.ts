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

interface GeminiPart {
  inlineData?: { data?: string | Uint8Array; mimeType?: string };
}

interface GeminiResponse {
  candidates?: Array<{ content?: { parts?: GeminiPart[] } }>;
}

export function extractPngBytesFromGeminiResponse(
  response: GeminiResponse
): Buffer | null {
  const parts = response.candidates?.[0]?.content?.parts;
  if (!parts?.length) return null;

  for (const part of parts) {
    const data = part.inlineData?.data;
    if (!data) continue;
    if (typeof data === "string") {
      return Buffer.from(data, "base64");
    }
    return Buffer.from(data);
  }

  return null;
}

export function timestampFilename(prefix: string): string {
  const stamp = new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\..+/, "")
    .replace("T", "_");
  return `${prefix}_${stamp}.png`;
}
