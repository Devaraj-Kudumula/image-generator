import { GoogleGenAI } from "@google/genai";

import { normalizeAspectRatio } from "@/lib/server/aspect-ratio";
import {
  decodeImageDataUrl,
  extractPngBytesFromGeminiResponse,
  imageBytesToDataUrl,
  timestampFilename,
} from "@/lib/server/image-utils";
import { getImageBytes, storeImage } from "@/lib/server/image-store";
import { EDIT_IMAGE_USER_PREFIX } from "@/lib/server/prompts";

const DEFAULT_MODEL = "gemini-3-pro-image-preview";

let client: GoogleGenAI | null = null;

export function getGoogleApiKey(): string | undefined {
  return process.env.GOOGLE_GENERATIVE_AI_API_KEY;
}

export function getGeminiClient(): GoogleGenAI {
  const apiKey = getGoogleApiKey();
  if (!apiKey) {
    throw new Error(
      "Google Generative AI API key not configured. Set GOOGLE_GENERATIVE_AI_API_KEY in .env"
    );
  }
  if (!client) {
    client = new GoogleGenAI({ apiKey });
  }
  return client;
}

export function isGeminiReady(): boolean {
  return Boolean(getGoogleApiKey());
}

function persistImage(bytes: Buffer, prefix: string) {
  const filename = timestampFilename(prefix);
  storeImage(filename, bytes);
  return {
    filename,
    imageBytes: bytes,
    imageDataUrl: imageBytesToDataUrl(bytes),
  };
}

export async function generateImageWithGemini(options: {
  prompt: string;
  aspectRatio?: string | null;
  model?: string | null;
}) {
  const gemini = getGeminiClient();
  const ratio = normalizeAspectRatio(options.aspectRatio);
  const model = options.model || DEFAULT_MODEL;

  const response = await gemini.models.generateContent({
    model,
    contents: options.prompt,
    config: {
      responseModalities: ["TEXT", "IMAGE"],
      imageConfig: { aspectRatio: ratio },
    },
  });

  const imageBytes = extractPngBytesFromGeminiResponse(response);
  if (!imageBytes?.length) {
    throw new Error("No image generated in response");
  }

  const saved = persistImage(imageBytes, "image");
  return { ...saved, aspectRatio: ratio };
}

function loadImageForEdit(filename: string, imageDataUrl?: string): Buffer {
  if (imageDataUrl) {
    return decodeImageDataUrl(imageDataUrl);
  }
  const stored = filename ? getImageBytes(filename) : undefined;
  if (stored) return stored;
  throw new Error(
    `File not found: ${filename}. Pass image_data_url for stateless edits.`
  );
}

export async function editImageWithGemini(options: {
  filename?: string;
  imageDataUrl?: string;
  changes: string;
  aspectRatio?: string | null;
}) {
  const gemini = getGeminiClient();
  const ratio = normalizeAspectRatio(options.aspectRatio);
  const imageBytes = loadImageForEdit(
    options.filename || "",
    options.imageDataUrl
  );
  const prompt = `${EDIT_IMAGE_USER_PREFIX}Changes: ${options.changes}`;

  const response = await gemini.models.generateContent({
    model: DEFAULT_MODEL,
    contents: [
      {
        role: "user",
        parts: [
          { text: prompt },
          {
            inlineData: {
              mimeType: "image/png",
              data: imageBytes.toString("base64"),
            },
          },
        ],
      },
    ],
    config: {
      responseModalities: ["TEXT", "IMAGE"],
      imageConfig: { aspectRatio: ratio },
    },
  });

  const editedBytes = extractPngBytesFromGeminiResponse(response);
  if (!editedBytes?.length) {
    throw new Error("No edited image generated in response");
  }

  const saved = persistImage(editedBytes, "edited");
  return { ...saved, aspectRatio: ratio };
}
