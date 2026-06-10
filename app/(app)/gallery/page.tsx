"use client";

import { useMemo } from "react";
import Image from "next/image";
import { Trash2 } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/alert";
import { useGalleryStore } from "@/lib/store/generation-store";
import { resolveImageSrc } from "@/lib/api";

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
              <Card key={image.id} className="overflow-hidden">
                <div className="relative aspect-square bg-muted">
                  <Image
                    src={resolveImageSrc(image)}
                    alt={image.prompt}
                    fill
                    className="object-contain p-2"
                    unoptimized
                  />
                </div>
                <div className="flex items-start justify-between gap-2 p-3">
                  <p className="line-clamp-2 flex-1 text-xs text-muted-foreground">
                    {image.prompt}
                  </p>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    onClick={() => removeImage(image.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </AppShell>
  );
}
