export async function GET(req: Request) {
  const sessionId = new URL(req.url).searchParams.get("session_id")?.trim() || "";
  return Response.json({
    doc_names: [],
    base_doc_names: [],
    session_doc_names: [],
    count: 0,
    session_id: sessionId,
    disabled: true,
  });
}
