export type AspectRatio = "auto" | "1:1" | "16:9" | "9:16" | "4:3" | "3:4";

/** Aspect ratios used by AI Chat (mirrors static/ai_chat.js). */
export type ChatAspectRatio =
  | "1:1"
  | "4:3"
  | "3:4"
  | "16:9"
  | "9:16"
  | "3:2"
  | "2:3"
  | "21:9";

export const CHAT_ASPECT_RATIOS: Array<{
  value: ChatAspectRatio;
  label: string;
}> = [
  { value: "16:9", label: "16:9 — Widescreen" },
  { value: "9:16", label: "9:16 — Portrait" },
  { value: "1:1", label: "1:1 — Square" },
  { value: "4:3", label: "4:3 — Standard" },
  { value: "3:4", label: "3:4 — Tall" },
  { value: "3:2", label: "3:2 — Photo" },
  { value: "2:3", label: "2:3 — Photo tall" },
  { value: "21:9", label: "21:9 — Ultrawide" },
];

export const DEFAULT_CHAT_ASPECT_RATIO: ChatAspectRatio = "16:9";

export type ImageKind =
  | "generated"
  | "edited"
  | "refined_prompt"
  | "accurate"
  | "canvas_edited";

export interface GeneratedImage {
  id: string;
  filename: string;
  imageUrl: string;
  imageDataUrl?: string;
  prompt: string;
  aspectRatio?: string;
  createdAt: string;
  kind?: ImageKind;
  meta?: string;
  parentImageId?: string;
  accuracyTrace?: unknown;
}

export interface GenerateImageBody {
  prompt: string;
  aspect_ratio?: AspectRatio | ChatAspectRatio;
  model?: string;
  session_id?: string;
}

export interface EditImageBody {
  filename?: string;
  image_data_url?: string;
  changes: string;
  aspect_ratio?: ChatAspectRatio;
  session_id?: string;
}

export interface GetAccurateBody {
  filename?: string;
  image_data_url?: string;
  original_prompt?: string;
  prompt?: string;
  include_trace?: boolean;
  aspect_ratio?: ChatAspectRatio;
  session_id?: string;
}

export interface VectorizeBody {
  filename?: string;
  image_data_url?: string;
  include_meta?: boolean;
  prompt?: string;
}

export interface RefineSvgBody {
  filename?: string;
  image_data_url?: string;
  instructions?: string;
  max_iterations?: number;
  include_trace?: boolean;
}

export interface ChatTheme {
  label: string;
  prompt: string;
}

export interface ChatMessageBody {
  user_message: string;
  history?: Array<{ role: "user" | "assistant"; content: string }>;
  system_prompt_override?: string;
}

export interface DocChatBody {
  user_question: string;
  selected_doc_names?: string[];
  chat_history?: string;
  session_id?: string;
}

async function parseJsonResponse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let data: unknown = null;

  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      if (!res.ok) {
        throw new Error(
          res.status >= 500
            ? "Image API request failed. Check that GOOGLE_GENERATIVE_AI_API_KEY is set in .env and restart the dev server."
            : text || `Request failed: ${res.status}`
        );
      }
    }
  }

  if (!res.ok) {
    const err = (data ?? {}) as { error?: string };
    throw new Error(err.error || `Request failed: ${res.status}`);
  }

  return data as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Network error";
    throw new Error(
      detail === "fetch failed" || detail === "Failed to fetch"
        ? "Request failed — the server may still be processing, or Python dependencies are missing. Run: pip install -r requirements.txt"
        : detail
    );
  }
  return parseJsonResponse<T>(res);
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  return parseJsonResponse<T>(res);
}

export const api = {
  health: () => get<{ status: string; gemini_client_ready: boolean }>("/api/health"),

  generateImage: (body: GenerateImageBody) =>
    post<{
      success: boolean;
      image_url: string;
      filename: string;
      image_data_url?: string;
      aspect_ratio?: string;
      image_prompt?: string;
    }>("/api/generate-image", body),

  editImage: (body: EditImageBody) =>
    post<{
      success: boolean;
      image_url: string;
      filename: string;
      image_data_url?: string;
      aspect_ratio?: string;
    }>("/api/edit-image", body),

  getAccurate: (body: GetAccurateBody) =>
    post<{
      success: boolean;
      image_url: string;
      filename: string;
      image_data_url?: string;
      aspect_ratio?: string;
      flaws_detected?: number;
      iterations?: number;
      accuracy_trace?: unknown;
    }>("/api/get-accurate", body),

  refinedPromptImage: (body: GetAccurateBody) =>
    post<{
      success: boolean;
      image_url: string;
      filename: string;
      image_data_url?: string;
      aspect_ratio?: string;
      refined_prompt?: string;
      vision_analysis?: string;
      refined_regen_trace?: unknown;
    }>("/api/refined-prompt-image", body),

  vectorizeImage: (body: VectorizeBody) =>
    post<{ success: boolean; svg: string; svg_filename: string }>(
      "/api/vectorize-image",
      body
    ),

  refineSvgCodegen: (body: RefineSvgBody) =>
    post<{
      success: boolean;
      svg: string;
      svg_filename: string;
      iterations?: number;
      png_data_url?: string;
    }>("/api/refine-svg-codegen", body),

  getChatThemes: () =>
    get<{ themes: Record<string, ChatTheme> }>("/api/ai-chat-themes"),

  chatMessage: (body: ChatMessageBody) =>
    post<{ answer: string; usage?: Record<string, unknown> }>(
      "/api/ai-chat-message",
      body
    ),

  getDocNames: (sessionId: string) =>
    get<{
      doc_names: string[];
      base_doc_names: string[];
      session_doc_names: string[];
      disabled?: boolean;
    }>(`/api/doc-names?session_id=${encodeURIComponent(sessionId)}`),

  chatWithDocs: (body: DocChatBody) =>
    post<{ answer: string; sources?: string[] }>("/api/chat-with-docs", body),

  uploadDoc: async (file: File, sessionId: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("session_id", sessionId);
    const res = await fetch("/api/upload-doc", { method: "POST", body: form });
    return parseJsonResponse(res);
  },

  resetSession: (sessionId: string) =>
    post<{ success: boolean }>("/api/session/reset", { session_id: sessionId }),
};

export function resolveImageSrc(image: Pick<GeneratedImage, "imageUrl" | "imageDataUrl">) {
  return image.imageDataUrl || image.imageUrl;
}
