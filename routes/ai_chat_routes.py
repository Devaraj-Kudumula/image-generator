"""
AI Chat routes: free-form chat with GPT (no document retrieval).

Used by the AI Chat → Image Generation page, where the user chats with the
LLM, hovers on assistant replies, and pushes a chosen reply to the existing
image generation flow.
"""
import time
import logging
import traceback

from flask import request, jsonify

from app_state import state

logger = logging.getLogger(__name__)


def _extract_usage(response):
    """Best-effort extraction of token usage from a langchain ChatOpenAI response."""
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", None) or {}

    candidate = (
        usage_metadata
        or response_metadata.get("token_usage")
        or response_metadata.get("usage")
        or {}
    )
    if not isinstance(candidate, dict):
        candidate = {}

    prompt_tokens = (
        candidate.get("prompt_tokens")
        or candidate.get("input_tokens")
    )
    completion_tokens = (
        candidate.get("completion_tokens")
        or candidate.get("output_tokens")
    )
    total_tokens = candidate.get("total_tokens")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def register(app):
    @app.route("/ai-chat-message", methods=["POST"])
    def ai_chat_message():
        """Generate a chat reply from the LLM (no document retrieval)."""
        request_start = time.time()
        logger.info("[/ai-chat-message] Request received")

        try:
            data = request.get_json() or {}
            user_message = str((data or {}).get("user_message") or "").strip()
            history = (data or {}).get("history") or []

            if not user_message:
                return jsonify({"error": "user_message is required"}), 400
            if state.llm is None:
                return jsonify({"error": "LLM is not initialized"}), 503

            messages = []

            if isinstance(history, list):
                for entry in history:
                    if not isinstance(entry, dict):
                        continue
                    role = entry.get("role")
                    content = entry.get("content")
                    if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                        messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": user_message})

            api_start = time.time()
            response = state.llm.invoke(messages)
            api_latency_ms = int((time.time() - api_start) * 1000)

            usage = _extract_usage(response)
            response_metadata = getattr(response, "response_metadata", {}) or {}
            model_name = response_metadata.get("model_name") or "gpt-4"
            finish_reason = response_metadata.get("finish_reason")

            request_ms = int((time.time() - request_start) * 1000)
            logger.info(
                "[/ai-chat-message] OK in %dms (api %dms, total_tokens=%s)",
                request_ms,
                api_latency_ms,
                usage.get("total_tokens"),
            )

            return jsonify({
                "answer": (response.content or "").strip(),
                "metrics": {
                    "model": model_name,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "latency_ms": api_latency_ms,
                    "request_ms": request_ms,
                    "finish_reason": finish_reason,
                },
            }), 200
        except Exception as e:
            logger.error("[/ai-chat-message] Error: %s", e)
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
