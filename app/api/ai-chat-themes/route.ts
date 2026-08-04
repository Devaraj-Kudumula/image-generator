import themesData from "@/lib/chat-themes.json";

export async function GET() {
  return Response.json({ themes: themesData });
}
