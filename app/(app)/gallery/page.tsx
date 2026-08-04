"use client";

import { useMemo } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { Alert } from "@/components/ui/alert";
import { ImageCard } from "@/components/generation/result-grid";
import { useGalleryStore } from "@/lib/store/generation-store";

export default function GalleryPage() {
  const { images, removeImage } = useGalleryStore();
  const sorted = useMemo(
    () => [...images].sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
    [images]
  );

  return (
    <AppShell>
      <div className="mx-auto max-w-6xl space-y-6 px-4 py-8 md:px-6">
        <div>
          <h2 className="text-2xl font-semibold">Gallery</h2>
          <p className="text-sm text-muted-foreground">
            Browse and manage your generated figures
          </p>
        </div>

        {!sorted.length ? (
          <Alert className="border-dashed">
            No images yet. Generate figures in Studio or AI Chat to populate your gallery.
          </Alert>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {sorted.map((image) => (
              <ImageCard
                key={image.id}
                image={image}
                inlineActions
                onDelete={() => removeImage(image.id)}
              />
            ))}
          </div>
        )}
      </div>
    </AppShell>
  );
}
