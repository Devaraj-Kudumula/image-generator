import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { UIMessage } from "ai";
import type { AspectRatio, GeneratedImage } from "@/lib/api";
import { generateId } from "@/lib/utils";

export type GenerationMode = "illustration" | "flowchart";
export type InputMode = "generate" | "enhance" | "sketch" | "reference";

interface GenerationState {
  prompt: string;
  aspectRatio: AspectRatio;
  style: string;
  model: string;
  mode: GenerationMode;
  inputMode: InputMode;
  category: string | null;
  isGenerating: boolean;
  error: string | null;
  results: GeneratedImage[];
  referenceFile: File | null;
  referenceDataUrl: string | null;
  setPrompt: (prompt: string) => void;
  setAspectRatio: (ratio: AspectRatio) => void;
  setStyle: (style: string) => void;
  setModel: (model: string) => void;
  setMode: (mode: GenerationMode) => void;
  setInputMode: (mode: InputMode) => void;
  setCategory: (category: string | null) => void;
  setReferenceFile: (file: File | null, dataUrl: string | null) => void;
  setGenerating: (value: boolean) => void;
  setError: (error: string | null) => void;
  addResult: (result: Omit<GeneratedImage, "id" | "createdAt">) => GeneratedImage;
  clearResults: () => void;
}

export const useGenerationStore = create<GenerationState>()((set) => ({
  prompt: "",
  aspectRatio: "auto",
  style: "flat",
  model: "gemini-3-pro-image-preview",
  mode: "illustration",
  inputMode: "generate",
  category: null,
  isGenerating: false,
  error: null,
  results: [],
  referenceFile: null,
  referenceDataUrl: null,
  setPrompt: (prompt) => set({ prompt }),
  setAspectRatio: (aspectRatio) => set({ aspectRatio }),
  setStyle: (style) => set({ style }),
  setModel: (model) => set({ model }),
  setMode: (mode) => set({ mode }),
  setInputMode: (inputMode) => set({ inputMode }),
  setCategory: (category) => set({ category }),
  setReferenceFile: (referenceFile, referenceDataUrl) =>
    set({ referenceFile, referenceDataUrl }),
  setGenerating: (isGenerating) => set({ isGenerating }),
  setError: (error) => set({ error }),
  addResult: (result) => {
    const entry: GeneratedImage = {
      ...result,
      id: generateId("img"),
      createdAt: new Date().toISOString(),
    };
    set((state) => ({ results: [entry, ...state.results] }));
    return entry;
  },
  clearResults: () => set({ results: [] }),
}));

interface GalleryState {
  images: GeneratedImage[];
  addImage: (image: GeneratedImage) => void;
  removeImage: (id: string) => void;
}

const MAX_GALLERY_IMAGES = 100;

function stripDataUrl(image: GeneratedImage): GeneratedImage {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { imageDataUrl: _, ...rest } = image;
  return rest;
}

export const useGalleryStore = create<GalleryState>()(
  persist(
    (set) => ({
      images: [],
      addImage: (image) =>
        set((state) => ({
          images: [image, ...state.images.filter((i) => i.id !== image.id)].slice(
            0,
            MAX_GALLERY_IMAGES
          ),
        })),
      removeImage: (id) =>
        set((state) => ({ images: state.images.filter((i) => i.id !== id) })),
    }),
    {
      name: "gallery-store",
      partialize: (state) => ({
        images: state.images.map(stripDataUrl),
      }),
    }
  )
);

export type ChatImage = GeneratedImage & { messageId?: string };

export interface ChatSession {
  id: string;
  name: string;
  themeId: string | null;
  themeLabel: string | null;
  messages: Array<{
    id: string;
    role: "user" | "assistant" | "theme";
    content: string;
    themeId?: string;
    themeLabel?: string;
  }>;
  uiMessages: UIMessage[];
  images: ChatImage[];
}

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  createSession: () => ChatSession;
  setActiveSession: (id: string) => void;
  deleteSession: (id: string) => void;
  renameSession: (id: string, name: string) => void;
  addMessage: (sessionId: string, message: ChatSession["messages"][0]) => void;
  addImage: (sessionId: string, image: ChatImage) => void;
  setTheme: (sessionId: string, themeId: string, themeLabel: string, prompt: string) => void;
  clearTheme: (sessionId: string) => void;
  setUiMessages: (sessionId: string, messages: UIMessage[]) => void;
}

const defaultSession = (): ChatSession => ({
  id: generateId("chat"),
  name: "New chat",
  themeId: null,
  themeLabel: null,
  messages: [],
  uiMessages: [],
  images: [],
});

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [defaultSession()],
      activeSessionId: null,
      createSession: () => {
        const session = defaultSession();
        set((state) => ({
          sessions: [session, ...state.sessions],
          activeSessionId: session.id,
        }));
        return session;
      },
      setActiveSession: (id) => set({ activeSessionId: id }),
      deleteSession: (id) =>
        set((state) => {
          const sessions = state.sessions.filter((s) => s.id !== id);
          return {
            sessions: sessions.length ? sessions : [defaultSession()],
            activeSessionId:
              state.activeSessionId === id
                ? sessions[0]?.id ?? defaultSession().id
                : state.activeSessionId,
          };
        }),
      renameSession: (id, name) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, name } : s
          ),
        })),
      addMessage: (sessionId, message) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? { ...s, messages: [...s.messages, message] }
              : s
          ),
        })),
      addImage: (sessionId, image) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, images: [...s.images, image] } : s
          ),
        })),
      setTheme: (sessionId, themeId, themeLabel, prompt) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  themeId,
                  themeLabel,
                  messages: [
                    ...s.messages.filter((m) => m.role !== "theme"),
                    {
                      id: generateId("theme"),
                      role: "theme" as const,
                      content: prompt,
                      themeId,
                      themeLabel,
                    },
                  ],
                }
              : s
          ),
        })),
      clearTheme: (sessionId) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  themeId: null,
                  themeLabel: null,
                  messages: s.messages.filter((m) => m.role !== "theme"),
                }
              : s
          ),
        })),
      setUiMessages: (sessionId, uiMessages) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, uiMessages } : s
          ),
        })),
    }),
    {
      name: "chat-store",
      partialize: (state) => ({
        sessions: state.sessions.map((s) => ({
          ...s,
          images: s.images.map(stripDataUrl),
          uiMessages: s.uiMessages.map((m) => ({
            ...m,
            parts: m.parts?.map((p) =>
              p.type === "file" ? { ...p, data: "" } : p
            ) as UIMessage["parts"],
          })),
        })),
        activeSessionId: state.activeSessionId,
      }),
      migrate: (persisted) => {
        const state = persisted as Pick<ChatState, "sessions" | "activeSessionId">;
        if (state?.sessions) {
          state.sessions = state.sessions.map((s) => ({
            ...s,
            uiMessages: s.uiMessages ?? [],
          }));
        }
        return state;
      },
      onRehydrateStorage: () => (state) => {
        if (state && !state.activeSessionId && state.sessions[0]) {
          state.activeSessionId = state.sessions[0].id;
        }
      },
    }
  )
);

interface SidebarState {
  collapsed: boolean;
  mobileOpen: boolean;
  toggleCollapsed: () => void;
  setMobileOpen: (open: boolean) => void;
}

export const useSidebarStore = create<SidebarState>()(
  persist(
    (set) => ({
      collapsed: false,
      mobileOpen: false,
      toggleCollapsed: () => set((s) => ({ collapsed: !s.collapsed })),
      setMobileOpen: (mobileOpen) => set({ mobileOpen }),
    }),
    { name: "sidebar-store" }
  )
);
