import { proxyToFlaskApi } from "@/lib/server/flask-proxy";

export const maxDuration = 300;

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as Record<string, unknown>;

  const result = await proxyToFlaskApi<{
    success?: boolean;
    svg?: string;
    svg_filename?: string;
    iterations?: number;
    png_data_url?: string;
    error?: string;
  }>("/api/refine-svg-codegen", body);

  if (!result.ok) {
    return Response.json(
      {
        error:
          result.error ||
          "SVG refinement is unavailable. Run the local Flask backend (requirements-local.txt) for the diagram pipeline.",
      },
      { status: result.status }
    );
  }

  return Response.json(result.data);
}
