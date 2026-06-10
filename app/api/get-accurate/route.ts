export async function POST() {
  return Response.json(
    {
      error:
        "Accuracy refinement is not available in the built-in API yet. This feature requires the Python image QA pipeline.",
    },
    { status: 501 }
  );
}
