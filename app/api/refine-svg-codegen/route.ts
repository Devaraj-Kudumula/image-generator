export async function POST() {
  return Response.json(
    {
      error:
        "SVG refinement is not available in the built-in API yet. This feature requires the Python diagram pipeline.",
    },
    { status: 501 }
  );
}
