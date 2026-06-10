"use client";

import { useState } from "react";
import Image from "next/image";
import {
  Download,
  Expand,
  PenLine,
  Sparkles,
  Target,
  Layers,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { GeneratedImage } from "@/lib/api";
import { api, resolveImageSrc } from "@/lib/api";
import { useGenerationStore, useGalleryStore } from "@/lib/store/generation-store";
import { cn } from "@/lib/utils";
import dynamic from "next/dynamic";

const CanvasEditor = dynamic(
  () => import("@/components/canvas/canvas-editor").then((m) => m.CanvasEditor),
  { ssr: false }
);

interface ImageCardProps {
  image: GeneratedImage;
  className?: string;
  inlineActions?: boolean;
  onImageCreated?: (image: GeneratedImage) => void;
  onDelete?: () => void;
}

export function ImageCard({
  image,
  className,
  inlineActions = false,
  onImageCreated,
  onDelete,
}: ImageCardProps) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const [canvasOpen, setCanvasOpen] = useState(false);
  const [loading, setLoading] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const addResult = useGenerationStore((s) => s.addResult);
  const addToGallery = useGalleryStore((s) => s.addImage);
  const src = resolveImageSrc(image);

  const runAction = async (
    action: "accurate" | "refined",
    label: string
  ) => {
    setLoading(label);
    setActionError(null);
    try {
      const fn = action === "accurate" ? api.getAccurate : api.refinedPromptImage;
      const result = await fn({
        filename: image.filename,
        image_data_url: image.imageDataUrl,
        prompt: image.prompt,
      });
      const entry = addResult({
        filename: result.filename,
        imageUrl: result.image_url,
        imageDataUrl: result.image_data_url,
        prompt: image.prompt,
        kind: action === "accurate" ? "accurate" : "refined_prompt",
      });
      addToGallery(entry);
      onImageCreated?.(entry);
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : `${label} action failed`
      );
    } finally {
      setLoading(null);
    }
  };

  const download = () => {
    const link = document.createElement("a");
    link.href = src;
    link.download = image.filename || "figure.png";
    link.click();
  };

  return (
    <>
      <Card className={cn("group overflow-hidden", className)}>
        <div className="relative aspect-square bg-muted">
          <button
            type="button"
            className="absolute inset-0 cursor-zoom-in"
            onClick={() => setPreviewOpen(true)}
            aria-label="Preview image"
          />
          <Image
            src={src}
            alt={image.prompt}
            fill
            className="pointer-events-none object-contain p-2"
            unoptimized
          />
          <div className="pointer-events-none absolute inset-0 flex items-end justify-center gap-1 bg-gradient-to-t from-black/70 via-black/20 to-transparent p-3 opacity-0 transition-opacity group-hover:opacity-100">
            <Button size="icon" variant="secondary" className="pointer-events-auto h-8 w-8" onClick={() => setPreviewOpen(true)}>
              <Expand className="h-4 w-4" />
            </Button>
            <Button size="icon" variant="secondary" className="pointer-events-auto h-8 w-8" onClick={download}>
              <Download className="h-4 w-4" />
            </Button>
            <Button size="icon" variant="secondary" className="pointer-events-auto h-8 w-8" onClick={() => setCanvasOpen(true)}>
              <Layers className="h-4 w-4" />
            </Button>
            <Button
              size="icon"
              variant="secondary"
              className="pointer-events-auto h-8 w-8"
              disabled={!!loading}
              onClick={() => void runAction("accurate", "accurate")}
            >
              <Target className="h-4 w-4" />
            </Button>
            <Button
              size="icon"
              variant="secondary"
              className="pointer-events-auto h-8 w-8"
              disabled={!!loading}
              onClick={() => void runAction("refined", "refined")}
            >
              <Sparkles className="h-4 w-4" />
            </Button>
            <Button size="icon" variant="secondary" className="pointer-events-auto h-8 w-8" asChild>
              <a href={`/edit?image=${encodeURIComponent(image.id)}`}>
                <PenLine className="h-4 w-4" />
              </a>
            </Button>
          </div>
        </div>
        <div className="space-y-2 p-3">
          <p className="line-clamp-2 text-xs text-muted-foreground">{image.prompt}</p>
          {inlineActions && (
            <div className="flex flex-wrap gap-1">
              <Button size="sm" variant="outline" onClick={() => setPreviewOpen(true)}>
                <Expand className="h-3.5 w-3.5" />
                Preview
              </Button>
              <Button size="sm" variant="outline" onClick={download}>
                <Download className="h-3.5 w-3.5" />
                Download
              </Button>
              <Button size="sm" variant="outline" onClick={() => setCanvasOpen(true)}>
                <Layers className="h-3.5 w-3.5" />
                Edit in canvas
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={!!loading}
                onClick={() => void runAction("accurate", "accurate")}
              >
                <Target className="h-3.5 w-3.5" />
                Get accurate
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={!!loading}
                onClick={() => void runAction("refined", "refined")}
              >
                <Sparkles className="h-3.5 w-3.5" />
                Refine
              </Button>
              <Button size="sm" variant="outline" asChild>
                <a href={`/edit?image=${encodeURIComponent(image.id)}`}>
                  <PenLine className="h-3.5 w-3.5" />
                  Edit
                </a>
              </Button>
              {onDelete && (
                <Button size="sm" variant="outline" onClick={onDelete}>
                  <Trash2 className="h-3.5 w-3.5" />
                  Delete
                </Button>
              )}
            </div>
          )}
          {loading && <p className="text-xs text-primary">Running {loading}…</p>}
          {actionError && (
            <p className="text-xs text-destructive">{actionError}</p>
          )}
        </div>
      </Card>

      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>Preview</DialogTitle>
          </DialogHeader>
          <div className="relative aspect-video w-full">
            <Image src={src} alt={image.prompt} fill className="object-contain" unoptimized />
          </div>
        </DialogContent>
      </Dialog>

      <CanvasEditor
        open={canvasOpen}
        onOpenChange={setCanvasOpen}
        image={image}
      />
    </>
  );
}

interface ResultGridProps {
  images: GeneratedImage[];
  isGenerating?: boolean;
}

export function ResultGrid({ images, isGenerating }: ResultGridProps) {
  if (!images.length && !isGenerating) {
    return (
      <div className="mx-auto max-w-4xl rounded-xl border border-dashed p-12 text-center text-sm text-muted-foreground">
        Generated figures will appear here. Enter a prompt above to get started.
      </div>
    );
  }

  return (
    <div className="mx-auto grid max-w-6xl gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {isGenerating && (
        <Card className="overflow-hidden">
          <Skeleton className="aspect-square w-full" />
          <div className="space-y-2 p-3">
            <Skeleton className="h-3 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        </Card>
      )}
      {images.map((image) => (
        <ImageCard key={image.id} image={image} />
      ))}
    </div>
  );
}
