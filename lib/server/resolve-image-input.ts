import { decodeImageDataUrl } from "@/lib/server/image-utils";
import { getImageBytes } from "@/lib/server/image-store";

export function resolveImageBytesFromRequest(options: {
  filename?: string;
  imageDataUrl?: string;
}): Buffer {
  const { filename, imageDataUrl } = options;

  if (imageDataUrl?.trim()) {
    const trimmed = imageDataUrl.trim();
    if (trimmed.startsWith("data:")) {
      return decodeImageDataUrl(trimmed);
    }
    const apiMatch = trimmed.match(/\/api\/images\/([^/?#]+)/i);
    if (apiMatch?.[1]) {
      const stored = getImageBytes(apiMatch[1]);
      if (stored) return stored;
    }
  }

  if (filename) {
    const stored = getImageBytes(filename);
    if (stored) return stored;
  }

  throw new Error(
    "Image not found. Regenerate the image or pass image_data_url."
  );
}
