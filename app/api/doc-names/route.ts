import { getFromFlaskApi } from "@/lib/server/flask-proxy";

type DocNamesResponse = {
  doc_names: string[];
  base_doc_names: string[];
  session_doc_names: string[];
  count: number;
  session_id: string;
  disabled?: boolean;
} & Record<string, unknown>;

export async function GET(req: Request) {
  const sessionId = new URL(req.url).searchParams.get("session_id")?.trim() || "";

  const result = await getFromFlaskApi<DocNamesResponse>(
    `/api/doc-names?session_id=${encodeURIComponent(sessionId)}`
  );

  // Flask unreachable (e.g. no Python backend in this deployment): degrade gracefully.
  if (!result.ok) {
    return Response.json({
      doc_names: [],
      base_doc_names: [],
      session_doc_names: [],
      count: 0,
      session_id: sessionId,
      disabled: true,
    });
  }

  return Response.json(result.data);
}
