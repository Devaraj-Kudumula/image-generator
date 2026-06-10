const imageStore = new Map<string, Buffer>();

export function storeImage(filename: string, bytes: Buffer): void {
  imageStore.set(filename, bytes);
}

export function getImageBytes(filename: string): Buffer | undefined {
  return imageStore.get(filename);
}
