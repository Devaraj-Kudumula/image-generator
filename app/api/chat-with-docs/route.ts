import { proxyToFlaskApi } from "@/lib/server/flask-proxy";

export const maxDuration = 120;

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as Record<string, unknown>;

  const result = await proxyToFlaskApi<{
    answer?: string;
    sources?: string[];
    error?: string;
  }>("/api/chat-with-docs", body);

  if (!result.ok) {
    return Response.json(
      {
        error:
          result.error ||
          "Document chat is not available. Is the Flask backend running?",
        disabled: result.status === 503,
      },
      { status: result.status }
    );
  }

  return Response.json(result.data);
}
