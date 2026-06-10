export async function POST() {
  return Response.json(
    {
      error: "Document upload is not available in this deployment.",
      disabled: true,
    },
    { status: 503 }
  );
}
