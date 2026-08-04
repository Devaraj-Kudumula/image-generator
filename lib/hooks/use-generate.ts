import { useMutation } from "@tanstack/react-query";
import { api, type GenerateImageBody } from "@/lib/api";
import { useGenerationStore, useGalleryStore } from "@/lib/store/generation-store";

export function useGenerate() {
  const addResult = useGenerationStore((s) => s.addResult);
  const addToGallery = useGalleryStore((s) => s.addImage);

  return useMutation({
    mutationFn: (body: GenerateImageBody) => api.generateImage(body),
    onSuccess: (result, variables) => {
      const image = addResult({
        filename: result.filename,
        imageUrl: result.image_url,
        imageDataUrl: result.image_data_url,
        prompt: variables.prompt,
        aspectRatio: result.aspect_ratio ?? variables.aspect_ratio,
        kind: "generated",
      });
      addToGallery(image);
    },
  });
}
