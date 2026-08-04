import { decodeImageDataUrl } from "@/lib/server/image-utils";
import { storeImage } from "@/lib/server/image-store";
import { runPythonImageScript } from "@/lib/server/python-image-pipeline";
import { resolveImageBytesFromRequest } from "@/lib/server/resolve-image-input";

export const maxDuration = 300;

type RefinedPromptResult = Record<string, unknown> & {
  success: boolean;
  filename: string;
  image_data_url?: string;
  aspect_ratio?: string;
  refined_prompt?: string;
  vision_analysis?: string;
  refined_regen_trace?: unknown;
  usage?: Record<string, unknown>;
};

export async function POST(req: Request) {
  try {
    const data = await req.json();
    const filename = data?.filename;
    const imageDataUrl = data?.image_data_url;
    const originalPrompt = data?.original_prompt || data?.prompt || "";
    const includeTrace = Boolean(data?.include_trace);
    const aspectRatio = data?.aspect_ratio;

    if (!filename && !imageDataUrl) {
      return Response.json(
        { error: "Either filename or image_data_url is required" },
        { status: 400 }
      );
    }

    const imageBytes = resolveImageBytesFromRequest({
      filename,
      imageDataUrl,
    });

    const result = await runPythonImageScript<RefinedPromptResult>(
      "refined_prompt_cli.py",
      {
        image_base64: imageBytes.toString("base64"),
        original_prompt: originalPrompt,
        include_trace: includeTrace,
        aspect_ratio: aspectRatio,
      }
    );

    const outputFilename = result.filename;
    let outputBytes = imageBytes;

    if (result.image_data_url?.startsWith("data:")) {
      outputBytes = decodeImageDataUrl(result.image_data_url);
    }

    storeImage(outputFilename, outputBytes);

    const origin = new URL(req.url).origin;
    const imageUrl = `${origin}/api/images/${outputFilename}`;

    return Response.json({
      ...result,
      image_url: imageUrl,
      image_data_url: result.image_data_url,
      success: true,
    });
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Error during refined prompt regeneration";
    return Response.json({ error: message }, { status: 500 });
  }
}
