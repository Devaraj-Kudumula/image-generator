import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import {
  AI_CHAT_SYSTEM,
  AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS,
  buildThemeSystemPrompt,
  conversationMaxContextTokens,
  conversationModel,
  conversationTemperature,
  normalizeHistory,
  trimHistoryForContextCap,
} from "@/lib/server/ai-chat";

export const maxDuration = 180;

export async function POST(req: Request) {
  try {
    const data = await req.json();
    const userMessage = String(data?.user_message || "").trim();
    const history = normalizeHistory(data?.history);
    const overrideRaw = data?.system_prompt_override;

    let systemText = AI_CHAT_SYSTEM;
    if (typeof overrideRaw === "string") {
      const stripped = overrideRaw.trim();
      if (stripped) {
        if (stripped.length > AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS) {
          return Response.json(
            {
              error: `system_prompt_override exceeds ${AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS} characters`,
            },
            { status: 400 }
          );
        }
        systemText = buildThemeSystemPrompt(stripped);
      }
    }

    if (!userMessage) {
      return Response.json({ error: "user_message is required" }, { status: 400 });
    }

    if (!process.env.OPENAI_API_KEY) {
      return Response.json(
        { error: "Conversation LLM is not initialized" },
        { status: 503 }
      );
    }

    const { history: trimmedHistory, trimmedPairs, estimatedInputTokens } =
      trimHistoryForContextCap(systemText, history, userMessage, conversationMaxContextTokens());

    const result = await generateText({
      model: openai(conversationModel()),
      system: systemText,
      temperature: conversationTemperature(),
      messages: [
        ...trimmedHistory.map((entry) => ({
          role: entry.role,
          content: entry.content,
        })),
        { role: "user" as const, content: userMessage },
      ],
    });

    return Response.json({
      answer: result.text.trim(),
      usage: result.usage,
      metrics: {
        model: conversationModel(),
        history_turns_sent: history.length,
        history_turns_trimmed_pairs: trimmedPairs,
        estimated_input_tokens: estimatedInputTokens,
        context_token_cap: conversationMaxContextTokens(),
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Error generating reply";
    return Response.json({ error: message }, { status: 500 });
  }
}
