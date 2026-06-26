import { proxyToFlaskApi } from "@/lib/server/flask-proxy";

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as Record<string, unknown>;

  const result = await proxyToFlaskApi<{
    success?: boolean;
    cleared?: boolean;
    error?: string;
  }>("/api/session/reset", body);

  // If Flask is unavailable, report a non-fatal no-op so the UI stays usable.
  if (!result.ok) {
    return Response.json({ success: true, cleared: false, disabled: true });
  }

  return Response.json(result.data);
}
