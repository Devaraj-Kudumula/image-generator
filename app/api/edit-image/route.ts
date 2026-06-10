import { editImageWithGemini } from "@/lib/server/gemini";

export const maxDuration = 120;

export async function POST(req: Request) {
  try {
    const data = await req.json();
    const changes = String(data?.changes || "").trim();
    const filename = data?.filename;
    const imageDataUrl = data?.image_data_url;
    const aspectRatio = data?.aspect_ratio;

    if (!changes) {
      return Response.json({ error: "changes is required" }, { status: 400 });
    }
    if (!filename && !imageDataUrl) {
      return Response.json(
        { error: "filename or image_data_url is required" },
        { status: 400 }
      );
    }

    const result = await editImageWithGemini({
      filename,
      imageDataUrl,
      changes,
      aspectRatio,
    });

    const origin = new URL(req.url).origin;
    const imageUrl = `${origin}/api/images/${result.filename}`;

    return Response.json({
      success: true,
      image_url: imageUrl,
      filename: result.filename,
      image_data_url: result.imageDataUrl,
      aspect_ratio: result.aspectRatio,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Error editing image";
    return Response.json({ error: message }, { status: 500 });
  }
}
