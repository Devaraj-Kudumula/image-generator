import { useGalleryStore } from "@/lib/store/generation-store";

export function useGallery() {
  const images = useGalleryStore((s) => s.images);
  const addImage = useGalleryStore((s) => s.addImage);
  const removeImage = useGalleryStore((s) => s.removeImage);
  return { images, addImage, removeImage };
}
