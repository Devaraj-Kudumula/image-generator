"use client";

import { useMemo, useState } from "react";
import Image from "next/image";
import {
  Download,
  Expand,
  Layers,
  Loader2,
  Sparkles,
  Target,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Alert } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import dynamic from "next/dynamic";
import { PromptModal } from "@/components/chat/prompt-modal";
import type { ChatImage } from "@/lib/store/generation-store";
import { useChatStore, useGalleryStore } from "@/lib/store/generation-store";
import { api, resolveImageSrc } from "@/lib/api";
import { generateId } from "@/lib/utils";
import { cn } from "@/lib/utils";

const CanvasEditor = dynamic(
  () => import("@/components/canvas/canvas-editor").then((m) => m.CanvasEditor),
  { ssr: false }
);

interface ChatImagePanelProps {
  sessionId: string;
  className?: string;
}

function promptModalTitle(kind: ChatImage["kind"]): string {
  return kind === "refined_prompt" ? "Refined generation prompt" : "Generation prompt";
}

function promptButtonLabel(kind: ChatImage["kind"]): string {
  return kind === "refined_prompt" ? "View refined prompt" : "View prompt";
}

export function ChatImagePanel({ sessionId, className }: ChatImagePanelProps) {
  const session = useChatStore((s) => s.sessions.find((entry) => entry.id === sessionId));
  const chatAspectRatio = useChatStore((s) => s.chatAspectRatio);
  const addImage = useChatStore((s) => s.addImage);
  const addToGallery = useGalleryStore((s) => s.addImage);

  const images = session?.images ?? [];
  const latest = images.length ? images[images.length - 1] : null;

  const [changes, setChanges] = useState("");
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [canvasOpen, setCanvasOpen] = useState(false);
  const [fullscreenSrc, setFullscreenSrc] = useState<string | null>(null);
  const [fullscreenImage, setFullscreenImage] = useState<ChatImage | null>(null);
  const [promptModal, setPromptModal] = useState<{
    title: string;
    body: string;
  } | null>(null);

  const latestSrc = useMemo(
    () => (latest ? resolveImageSrc(latest) : null),
    [latest]
  );

  const pushImage = (entry: Omit<ChatImage, "id" | "createdAt">) => {
    const image: ChatImage = {
      ...entry,
      id: generateId("img"),
      createdAt: new Date().toISOString(),
    };
    addImage(sessionId, image);
    addToGallery(image);
    return image;
  };

  const runWithLoading = async (label: string, fn: () => Promise<void>) => {
    setLoading(label);
    setError(null);
    try {
      await fn();
    } catch (err) {
      setError(err instanceof Error ? err.message : `${label} failed`);
    } finally {
      setLoading(null);
    }
  };

  const applyChanges = async () => {
    if (!latest) return;
    const trimmed = changes.trim();
    if (!trimmed) {
      setError("Please describe the changes you want to apply.");
      return;
    }

    await runWithLoading("Applying changes…", async () => {
      const result = await api.editImage({
        filename: latest.filename,
        image_data_url: latest.imageDataUrl || latest.imageUrl,
        changes: trimmed,
        aspect_ratio: chatAspectRatio,
        session_id: `ai_chat_${sessionId}`,
      });

      pushImage({
        prompt: trimmed,
        filename: result.filename,
        imageUrl: result.image_url,
        imageDataUrl: result.image_data_url,
        aspectRatio: result.aspect_ratio || chatAspectRatio,
        kind: "edited",
        parentImageId: latest.id,
      });
      setChanges("");
    });
  };

  const getAccurate = async (includeTrace: boolean) => {
    if (!latest) return;

    await runWithLoading(
      includeTrace ? "Get Accurate (with log)…" : "Get Accurate — refining…",
      async () => {
        const result = await api.getAccurate({
          filename: latest.filename,
          image_data_url: latest.imageDataUrl || latest.imageUrl,
          original_prompt: latest.prompt || "",
          include_trace: includeTrace,
          aspect_ratio: chatAspectRatio,
          session_id: `ai_chat_${sessionId}`,
        });

        const flaws = result.flaws_detected || 0;
        const iters = result.iterations || 0;
        const usedRatio = result.aspect_ratio || chatAspectRatio;
        const meta =
          (flaws > 0
            ? `Accurate image (${flaws} flaw${flaws !== 1 ? "s" : ""} fixed in ${iters} pass${iters !== 1 ? "es" : ""})`
            : "Accurate image (no flaws detected)") + ` · ${usedRatio}`;

        pushImage({
          prompt: latest.prompt || "",
          filename: result.filename,
          imageUrl: result.image_url,
          imageDataUrl: result.image_data_url,
          aspectRatio: usedRatio,
          kind: "accurate",
          meta,
          accuracyTrace: includeTrace ? result.accuracy_trace : undefined,
          parentImageId: latest.id,
        });
      }
    );
  };

  const refinedPromptImage = async () => {
    if (!latest) return;

    await runWithLoading(
      "Refined prompt — analyzing image, rewriting prompt, generating…",
      async () => {
        const result = await api.refinedPromptImage({
          filename: latest.filename,
          image_data_url: latest.imageDataUrl || latest.imageUrl,
          original_prompt: latest.prompt || "",
          aspect_ratio: chatAspectRatio,
          session_id: `ai_chat_${sessionId}`,
        });

        const refined = String(result.refined_prompt || "").trim();
        const usedRatio = result.aspect_ratio || chatAspectRatio;

        pushImage({
          prompt: refined || latest.prompt || "",
          filename: result.filename,
          imageUrl: result.image_url,
          imageDataUrl: result.image_data_url,
          aspectRatio: usedRatio,
          kind: "refined_prompt",
          meta: `Refined prompt image (vision QA → GPT → Gemini) · ${usedRatio}`,
          parentImageId: latest.id,
        });
      }
    );
  };

  const downloadLatest = () => {
    if (!latest || !latestSrc) return;
    const link = document.createElement("a");
    link.href = latestSrc;
    const filename = latest.filename || `generated-image-${Date.now()}`;
    link.download =
      filename + (/\.(png|jpe?g|webp)$/i.test(filename) ? "" : ".png");
    link.click();
  };

  const saveCanvasEdit = (pngDataUrl: string) => {
    pushImage({
      prompt: "Canvas edit",
      filename: "",
      imageUrl: pngDataUrl,
      imageDataUrl: pngDataUrl,
      aspectRatio: chatAspectRatio,
      kind: "canvas_edited",
    });
    setCanvasOpen(false);
  };

  const openPromptForImage = (image: ChatImage) => {
    const prompt = image.prompt?.trim();
    if (!prompt) return;
    setPromptModal({
      title: promptModalTitle(image.kind),
      body: prompt,
    });
  };

  const openFullscreen = (image: ChatImage) => {
    setFullscreenImage(image);
    setFullscreenSrc(resolveImageSrc(image));
  };

  const actionsDisabled = !!loading || !latest;

  return (
    <aside
      className={cn(
        "flex min-h-0 w-full shrink-0 flex-col border-t bg-sidebar lg:w-96 lg:border-l lg:border-t-0",
        className
      )}
    >
      <div className="border-b px-4 py-3">
        <p className="text-sm font-medium">Latest generated image</p>
        <p className="text-xs text-muted-foreground">
          {latest
            ? `Updated ${new Date(latest.createdAt).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}${latest.aspectRatio ? ` · ${latest.aspectRatio}` : ""}`
            : "No image generated yet"}
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-4 p-4">
          <Card className="overflow-hidden">
            {latestSrc ? (
              <button
                type="button"
                className="relative block aspect-video w-full cursor-zoom-in bg-muted"
                onClick={() => latest && openFullscreen(latest)}
              >
                <Image
                  src={latestSrc}
                  alt="Latest generated image"
                  fill
                  className="object-contain p-2"
                  unoptimized
                />
              </button>
            ) : (
              <div className="flex aspect-video items-center justify-center p-4 text-center text-sm text-muted-foreground">
                Click <strong className="mx-1">Generate image from this</strong> on any
                assistant reply to create an image here.
              </div>
            )}

            {latest?.prompt?.trim() && (
              <div className="border-t px-3 py-2">
                <Button
                  variant="link"
                  size="sm"
                  className="h-auto p-0 text-xs"
                  onClick={() => openPromptForImage(latest)}
                >
                  {promptButtonLabel(latest.kind)}
                </Button>
              </div>
            )}
          </Card>

          {latest && (
            <>
              <Button className="w-full" onClick={() => setCanvasOpen(true)}>
                <Layers className="h-4 w-4" />
                Edit in Canvas
              </Button>

              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground">
                  Suggest changes to this image
                </p>
                <Textarea
                  value={changes}
                  onChange={(e) => setChanges(e.target.value)}
                  rows={2}
                  placeholder="e.g., zoom on lesion, adjust lighting..."
                  className="resize-none text-sm"
                />
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    disabled={actionsDisabled}
                    onClick={() => void applyChanges()}
                  >
                    Apply changes
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={actionsDisabled}
                    onClick={() => setCanvasOpen(true)}
                  >
                    <Layers className="h-3.5 w-3.5" />
                    Edit in Canvas
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={actionsDisabled}
                    onClick={() => void getAccurate(false)}
                  >
                    <Target className="h-3.5 w-3.5" />
                    Get Accurate
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={actionsDisabled}
                    onClick={() => void getAccurate(true)}
                  >
                    <Target className="h-3.5 w-3.5" />
                    Get Accurate + log
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={actionsDisabled}
                    onClick={() => void refinedPromptImage()}
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    Refined prompt image
                  </Button>
                </div>
              </div>

              <Button
                variant="secondary"
                className="w-full"
                disabled={!latestSrc}
                onClick={downloadLatest}
              >
                <Download className="h-4 w-4" />
                Download latest image
              </Button>
            </>
          )}

          {loading && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              {loading}
            </div>
          )}

          {error && (
            <Alert className="border-destructive/50 text-destructive">{error}</Alert>
          )}

          <div className="space-y-2">
            <p className="text-sm font-medium">Images in this chat</p>
            {images.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No images yet for this session.
              </p>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                {images.map((image, idx) => {
                  const src = resolveImageSrc(image);
                  const caption = image.prompt || "";
                  return (
                    <Card key={image.id} className="overflow-hidden">
                      <button
                        type="button"
                        className="relative block aspect-square w-full cursor-zoom-in bg-muted"
                        onClick={() => openFullscreen(image)}
                      >
                        <Image
                          src={src}
                          alt={`Generated image ${idx + 1}`}
                          fill
                          className="object-contain p-1"
                          unoptimized
                        />
                      </button>
                      <div className="space-y-1 p-2">
                        <p className="line-clamp-2 text-[10px] text-muted-foreground">
                          {caption.slice(0, 80)}
                          {caption.length > 80 ? "…" : ""}
                        </p>
                        {image.aspectRatio && (
                          <p className="text-[10px] text-muted-foreground">
                            Aspect: {image.aspectRatio}
                          </p>
                        )}
                        {caption.trim() && (
                          <Button
                            variant="link"
                            size="sm"
                            className="h-auto p-0 text-[10px]"
                            onClick={() => openPromptForImage(image)}
                          >
                            {promptButtonLabel(image.kind)}
                          </Button>
                        )}
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </ScrollArea>

      {latest && (
        <CanvasEditor
          open={canvasOpen}
          onOpenChange={setCanvasOpen}
          image={latest}
          onSaveToChat={saveCanvasEdit}
        />
      )}

      <Dialog
        open={!!fullscreenSrc}
        onOpenChange={(open) => {
          if (!open) {
            setFullscreenSrc(null);
            setFullscreenImage(null);
          }
        }}
      >
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>Generated image</DialogTitle>
          </DialogHeader>
          {fullscreenSrc && (
            <div className="space-y-3">
              <div className="relative aspect-video w-full">
                <Image
                  src={fullscreenSrc}
                  alt="Full-screen generated image"
                  fill
                  className="object-contain"
                  unoptimized
                />
              </div>
              {fullscreenImage?.prompt && (
                <p className="text-sm text-muted-foreground">
                  Prompt:{" "}
                  {fullscreenImage.prompt.slice(0, 140)}
                  {fullscreenImage.prompt.length > 140 ? "…" : ""}
                </p>
              )}
              {fullscreenImage?.prompt?.trim() && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => openPromptForImage(fullscreenImage)}
                >
                  <Expand className="h-3.5 w-3.5" />
                  {promptButtonLabel(fullscreenImage.kind)}
                </Button>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>

      <PromptModal
        open={!!promptModal}
        onOpenChange={(open) => {
          if (!open) setPromptModal(null);
        }}
        title={promptModal?.title ?? "Prompt"}
        body={promptModal?.body ?? ""}
      />
    </aside>
  );
}
