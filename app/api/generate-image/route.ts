import { generateImageWithGemini } from "@/lib/server/gemini";

export const maxDuration = 120;

export async function POST(req: Request) {
  try {
    const data = await req.json();
    const prompt = String(data?.prompt || "").trim();
    const aspectRatio = data?.aspect_ratio;
    const model = data?.model;

    if (!prompt) {
      return Response.json({ error: "Prompt is required" }, { status: 400 });
    }

    const result = await generateImageWithGemini({
      prompt,
      aspectRatio,
      model,
    });

    const origin = new URL(req.url).origin;
    const imageUrl = `${origin}/api/images/${result.filename}`;

    return Response.json({
      success: true,
      image_url: imageUrl,
      filename: result.filename,
      image_data_url: result.imageDataUrl,
      aspect_ratio: result.aspectRatio,
      image_prompt: result.imagePrompt,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Error generating image";
    const status = message.includes("API key") ? 500 : 500;
    return Response.json({ error: message }, { status });
  }
}
