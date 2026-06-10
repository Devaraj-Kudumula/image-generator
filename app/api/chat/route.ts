import { convertToModelMessages, streamText } from "ai";
import { openai } from "@ai-sdk/openai";

export const maxDuration = 60;

const DEFAULT_SYSTEM = `You are a careful, expert assistant helping users think through medical and scientific illustration, anatomy, imaging, and related topics. Provide thorough, well-structured answers suitable for medical illustration brainstorming. You only output text — images are generated separately by the product.`;

export async function POST(req: Request) {
  const { messages, systemPromptOverride } = await req.json();

  const result = streamText({
    model: openai("gpt-4o"),
    system: systemPromptOverride || DEFAULT_SYSTEM,
    messages: convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
}
