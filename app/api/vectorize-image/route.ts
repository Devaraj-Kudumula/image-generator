import { vectorizeImage } from "@/lib/server/vectorize";

export const maxDuration = 120;

export async function POST(req: Request) {
  try {
    const data = await req.json();
    const filename = data?.filename as string | undefined;
    const imageDataUrl = data?.image_data_url as string | undefined;

    if (!filename && !imageDataUrl) {
      return Response.json(
        { error: "Either filename or image_data_url is required" },
        { status: 400 }
      );
    }

    const result = await vectorizeImage({
      filename,
      imageDataUrl,
      includeMeta: Boolean(data?.include_meta),
      debugDump: Boolean(data?.debug_dump),
    });

    return Response.json({
      success: result.success,
      svg: result.svg,
      svg_filename: result.svgFilename,
      ...(result.traceMeta ? { trace_meta: result.traceMeta } : {}),
      ...(result.debugSvgFilename
        ? { debug_svg_filename: result.debugSvgFilename }
        : {}),
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Error vectorizing image";
    return Response.json({ error: message }, { status: 500 });
  }
}
