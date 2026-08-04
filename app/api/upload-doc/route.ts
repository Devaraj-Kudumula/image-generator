import { proxyFormToFlaskApi } from "@/lib/server/flask-proxy";

export const maxDuration = 120;

export async function POST(req: Request) {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return Response.json({ error: "Invalid upload payload" }, { status: 400 });
  }

  const result = await proxyFormToFlaskApi<{
    success?: boolean;
    doc_name?: string;
    chunks_inserted?: number;
    session_doc_names?: string[];
    error?: string;
  }>("/api/upload-doc", form);

  if (!result.ok) {
    return Response.json(
      {
        error:
          result.error ||
          "Document upload is not available. Is the Flask backend running?",
        disabled: result.status === 503,
      },
      { status: result.status }
    );
  }

  return Response.json(result.data);
}
