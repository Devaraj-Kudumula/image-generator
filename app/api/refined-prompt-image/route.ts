export async function POST() {
  return Response.json(
    {
      error:
        "Refined prompt regeneration is not available in the built-in API yet.",
    },
    { status: 501 }
  );
}
