export type AspectRatio = "auto" | "1:1" | "16:9" | "9:16" | "4:3" | "3:4";

export interface GeneratedImage {
  id: string;
  filename: string;
  imageUrl: string;
  imageDataUrl?: string;
  prompt: string;
  aspectRatio?: string;
  createdAt: string;
  kind?: "generated" | "edited" | "refined" | "accurate";
}

export interface GenerateImageBody {
  prompt: string;
  aspect_ratio?: AspectRatio;
  session_id?: string;
}

export interface EditImageBody {
  filename?: string;
  image_data_url?: string;
  change_instructions: string;
  session_id?: string;
}

export interface GetAccurateBody {
  filename?: string;
  image_data_url?: string;
  prompt?: string;
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
            ? "Backend unavailable. Start Flask in another terminal: conda activate img && python server.py"
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
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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
    }>("/api/generate-image", body),

  editImage: (body: EditImageBody) =>
    post<{
      success: boolean;
      image_url: string;
      filename: string;
      image_data_url?: string;
    }>("/api/edit-image", body),

  getAccurate: (body: GetAccurateBody) =>
    post<{
      success: boolean;
      image_url: string;
      filename: string;
      image_data_url?: string;
    }>("/api/get-accurate", body),

  refinedPromptImage: (body: GetAccurateBody) =>
    post<{
      success: boolean;
      image_url: string;
      filename: string;
      image_data_url?: string;
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
    post<{ reply: string; usage?: Record<string, unknown> }>(
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
