import { useMutation } from "@tanstack/react-query";
import { api, type EditImageBody } from "@/lib/api";

export function useEditImage() {
  return useMutation({
    mutationFn: (body: EditImageBody) => api.editImage(body),
  });
}
