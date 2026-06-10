export async function POST() {
  return Response.json(
    {
      error: "Document chat is not available in this deployment.",
      disabled: true,
    },
    { status: 503 }
  );
}
