import { isGeminiReady } from "@/lib/server/gemini";

export async function GET() {
  return Response.json({
    status: "healthy",
    openai_configured: Boolean(process.env.OPENAI_API_KEY),
    google_configured: isGeminiReady(),
    gemini_client_ready: isGeminiReady(),
    rag_available: false,
    is_serverless: false,
  });
}
