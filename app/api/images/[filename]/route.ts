import { getImageBytes } from "@/lib/server/image-store";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ filename: string }> }
) {
  const { filename } = await params;
  const bytes = getImageBytes(filename);

  if (!bytes) {
    return Response.json({ error: "Image not found" }, { status: 404 });
  }

  return new Response(new Uint8Array(bytes), {
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "private, max-age=3600",
    },
  });
}
