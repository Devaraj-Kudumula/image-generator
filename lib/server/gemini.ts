import { GoogleGenAI } from "@google/genai";

import { normalizeAspectRatio } from "@/lib/server/aspect-ratio";
import {
  decodeImageDataUrl,
  describeGeminiImageFailure,
  extractPngBytesFromGeminiResponse,
  imageBytesToDataUrl,
  shouldRetryGeminiImageGeneration,
  timestampFilename,
} from "@/lib/server/image-utils";
import { imagePromptVariants } from "@/lib/server/prepare-image-prompt";
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

async function requestGeminiImage(
  prompt: string,
  aspectRatio: string,
  model: string
) {
  const gemini = getGeminiClient();
  const response = await gemini.models.generateContent({
    model,
    contents: prompt,
    config: {
      responseModalities: ["IMAGE"],
      imageConfig: { aspectRatio },
    },
  });

  return {
    response,
    imageBytes: extractPngBytesFromGeminiResponse(response),
  };
}

export async function generateImageWithGemini(options: {
  prompt: string;
  aspectRatio?: string | null;
  model?: string | null;
}) {
  const ratio = normalizeAspectRatio(options.aspectRatio);
  const model = options.model || DEFAULT_MODEL;
  const prompts = imagePromptVariants(options.prompt);

  let lastError = "No image generated in response";

  for (let index = 0; index < prompts.length; index += 1) {
    const prompt = prompts[index];
    const { response, imageBytes } = await requestGeminiImage(prompt, ratio, model);

    if (imageBytes?.length) {
      const saved = persistImage(imageBytes, "image");
      return { ...saved, aspectRatio: ratio, imagePrompt: prompt };
    }

    lastError = describeGeminiImageFailure(response);
    const hasAnotherVariant = index < prompts.length - 1;
    if (!hasAnotherVariant || !shouldRetryGeminiImageGeneration(response)) {
      break;
    }
  }

  throw new Error(lastError);
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
      responseModalities: ["IMAGE"],
      imageConfig: { aspectRatio: ratio },
    },
  });

  const editedBytes = extractPngBytesFromGeminiResponse(response);
  if (!editedBytes?.length) {
    throw new Error(describeGeminiImageFailure(response));
  }

  const saved = persistImage(editedBytes, "edited");
  return { ...saved, aspectRatio: ratio };
}
