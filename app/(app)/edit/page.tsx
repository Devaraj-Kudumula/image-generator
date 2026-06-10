"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import Image from "next/image";
import { Expand, Upload, Wand2 } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Alert } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { api } from "@/lib/api";
import { useGalleryStore, useGenerationStore } from "@/lib/store/generation-store";
import { generateId } from "@/lib/utils";
import dynamic from "next/dynamic";

const CanvasEditor = dynamic(
  () => import("@/components/canvas/canvas-editor").then((m) => m.CanvasEditor),
  { ssr: false }
);

function EditPageContent() {
  const searchParams = useSearchParams();
  const fileRef = useRef<HTMLInputElement>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [filename, setFilename] = useState<string>("");
  const [instructions, setInstructions] = useState("");
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [canvasOpen, setCanvasOpen] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const addToGallery = useGalleryStore((s) => s.addImage);
  const galleryImages = useGalleryStore((s) => s.images);
  const generationResults = useGenerationStore((s) => s.results);

  useEffect(() => {
    const imageId = searchParams.get("image");
    if (!imageId) return;

    const galleryImage = galleryImages.find((img) => img.id === imageId);
    const resultImage = generationResults.find((img) => img.id === imageId);
    const source = galleryImage ?? resultImage;
    if (!source) return;

    setImageDataUrl(source.imageDataUrl ?? null);
    setImageUrl(source.imageUrl);
    setFilename(source.filename);
    setInstructions(source.prompt || "");
    setResultUrl(null);
  }, [searchParams, galleryImages, generationResults]);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setImageDataUrl(reader.result as string);
      setImageUrl(null);
      setFilename(file.name);
      setResultUrl(null);
    };
    reader.readAsDataURL(file);
  };

  const handleEdit = async () => {
    if ((!imageDataUrl && !filename) || !instructions.trim()) {
      setError("Upload an image and enter edit instructions");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await api.editImage({
        ...(imageDataUrl ? { image_data_url: imageDataUrl } : { filename }),
        changes: instructions.trim(),
      });
      const url = result.image_data_url || result.image_url;
      setResultUrl(url);
      setFilename(result.filename);
      addToGallery({
        id: generateId("img"),
        filename: result.filename,
        imageUrl: result.image_url,
        imageDataUrl: result.image_data_url,
        prompt: instructions,
        createdAt: new Date().toISOString(),
        kind: "edited",
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Edit failed");
    } finally {
      setLoading(false);
    }
  };

  const activeImage =
    resultUrl || imageDataUrl || imageUrl;

  return (
    <AppShell>
      <div className="mx-auto grid max-w-6xl gap-6 px-4 py-8 md:grid-cols-2 md:px-6">
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold">Upload & Edit</h2>
            <p className="text-sm text-muted-foreground">
              Upload a figure and describe the changes you want
            </p>
          </div>

          <Card
            className="flex min-h-[280px] cursor-pointer flex-col items-center justify-center gap-3 border-dashed p-8 text-center"
            onClick={() => fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleUpload} />
            <Upload className="h-8 w-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Drop an image or click to upload
            </p>
            {filename && <p className="text-xs text-foreground">{filename}</p>}
          </Card>

          <div className="space-y-2">
            <Label htmlFor="instructions">Edit instructions</Label>
            <Textarea
              id="instructions"
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              placeholder="e.g. Add labels for all structures, improve contrast, fix left-right orientation…"
              className="min-h-[120px]"
            />
          </div>

          {error && <Alert className="border-destructive/50 text-destructive">{error}</Alert>}

          <div className="flex flex-wrap gap-2">
            <Button onClick={() => void handleEdit()} disabled={loading}>
              <Wand2 className="h-4 w-4" />
              {loading ? "Editing…" : "Apply edits"}
            </Button>
            {activeImage && (
              <Button variant="outline" onClick={() => setCanvasOpen(true)}>
                Open canvas editor
              </Button>
            )}
          </div>
        </div>

        <Card className="flex min-h-[400px] flex-col overflow-hidden bg-muted/30 p-4">
          {activeImage ? (
            <>
              <button
                type="button"
                className="relative min-h-[360px] w-full flex-1 cursor-zoom-in"
                onClick={() => setPreviewOpen(true)}
                aria-label="Preview image"
              >
                <Image
                  src={activeImage}
                  alt="Preview"
                  fill
                  className="object-contain"
                  unoptimized
                />
              </button>
              <div className="mt-3 flex justify-center">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setPreviewOpen(true)}
                >
                  <Expand className="h-3.5 w-3.5" />
                  Zoom preview
                </Button>
              </div>
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center">
              <p className="text-sm text-muted-foreground">Preview will appear here</p>
            </div>
          )}
        </Card>
      </div>

      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>Preview</DialogTitle>
          </DialogHeader>
          {activeImage && (
            <div className="relative aspect-video w-full">
              <Image
                src={activeImage}
                alt="Preview"
                fill
                className="object-contain"
                unoptimized
              />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {activeImage && (
        <CanvasEditor
          open={canvasOpen}
          onOpenChange={setCanvasOpen}
          image={{
            id: "edit-preview",
            filename,
            imageUrl: activeImage,
            imageDataUrl: activeImage.startsWith("data:") ? activeImage : imageDataUrl ?? undefined,
            prompt: instructions,
            createdAt: new Date().toISOString(),
          }}
        />
      )}
    </AppShell>
  );
}

export default function EditPage() {
  return (
    <Suspense
      fallback={
        <AppShell>
          <div className="mx-auto max-w-6xl px-4 py-8 md:px-6">
            <p className="text-sm text-muted-foreground">Loading editor…</p>
          </div>
        </AppShell>
      }
    >
      <EditPageContent />
    </Suspense>
  );
}
