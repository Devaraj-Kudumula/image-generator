"""
AI Chat routes: free-form chat with GPT (no document retrieval).

Used by the AI Chat → Image Generation page, where the user chats with the
LLM, hovers on assistant replies, and pushes a chosen reply to the existing
image generation flow.
"""
import time
import json
import logging
import traceback

from flask import request, jsonify, Response, stream_with_context

import config
from app_state import state
from prompts import AI_CHAT_SYSTEM, AI_CHAT_THEME_PROMPTS

logger = logging.getLogger(__name__)

# Hard cap for client-supplied system override (chars) to avoid abuse / huge payloads.
_AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS = 12000


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


def _tiktoken_encoding_for_model(model_name: str):
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def _message_list_token_estimate(messages, encoding):
    """Rough token count for OpenAI-style message list (content strings only)."""
    total = 0
    per_message_overhead = 4
    for msg in messages:
        content = msg.get("content")
        text = content if isinstance(content, str) else ""
        if encoding is not None:
            total += len(encoding.encode(text))
        else:
            total += max(1, len(text) // 4)
        total += per_message_overhead
    return total


def _normalize_history_entries(history):
    """Return list of {role, content} dicts in order; only user/assistant with non-empty text."""
    out = []
    if not isinstance(history, list):
        return out
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out


def _build_messages_with_context_cap(
    system_text,
    history_entries,
    user_message,
    max_context_tokens,
    model_name,
):
    """
    Always include system + latest user message. Prior turns are included in full
    order until the estimated *total* fits under max_context_tokens. Oldest complete
    (user, assistant) pairs are removed first; if one message remains over budget,
    drop from the front until the history slice fits its sub-budget.
    """
    encoding = _tiktoken_encoding_for_model(model_name)
    user_message = (user_message or "").strip()
    if not user_message:
        return [], 0, len(history_entries)

    system_msg = {"role": "system", "content": system_text}
    user_msg = {"role": "user", "content": user_message}
    fixed_tokens = _message_list_token_estimate([system_msg, user_msg], encoding)
    # Small buffer for API message framing / tool-less chat overhead
    history_budget = max_context_tokens - fixed_tokens - 16
    if history_budget < 0:
        logger.warning(
            "[/ai-chat-message] OPENAI_CONVERSATION_MAX_CONTEXT_TOKENS=%s is smaller than "
            "system + latest user (~%s est. tokens); using history-only budget 0",
            max_context_tokens,
            fixed_tokens,
        )
        history_budget = 0

    hist = list(history_entries)
    trimmed_pairs = 0

    def history_token_count():
        return _message_list_token_estimate(hist, encoding)

    while history_token_count() > history_budget:
        if len(hist) >= 2:
            hist = hist[2:]
            trimmed_pairs += 1
            continue
        if hist:
            hist = hist[1:]
            continue
        break

    messages = [system_msg] + hist + [user_msg]
    total_est = _message_list_token_estimate(messages, encoding)
    if total_est > max_context_tokens:
        logger.warning(
            "[/ai-chat-message] Estimated total tokens %s still above cap %s (fixed messages dominate)",
            total_est,
            max_context_tokens,
        )
    return messages, total_est, trimmed_pairs


def register(app):
    @app.route("/ai-chat-themes", methods=["GET"])
    def ai_chat_themes():
        """Theme ids, labels, and full system prompt text for the AI Chat page."""
        themes = {}
        for theme_id, meta in (AI_CHAT_THEME_PROMPTS or {}).items():
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("label") or theme_id).strip() or theme_id
            prompt = str(meta.get("prompt") or "").strip()
            themes[theme_id] = {"label": label, "prompt": prompt}
        return jsonify({"themes": themes}), 200

    @app.route("/ai-chat-message", methods=["POST"])
    def ai_chat_message():
        """Generate a chat reply from the LLM (no document retrieval)."""
        request_start = time.time()
        logger.info("[/ai-chat-message] Request received")

        try:
            data = request.get_json() or {}
            user_message = str((data or {}).get("user_message") or "").strip()
            history = (data or {}).get("history") or []
            override_raw = (data or {}).get("system_prompt_override")
            system_text = AI_CHAT_SYSTEM
            if isinstance(override_raw, str):
                stripped = override_raw.strip()
                if stripped:
                    if len(stripped) > _AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS:
                        return jsonify({
                            "error": (
                                f"system_prompt_override exceeds {_AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS} characters"
                            ),
                        }), 400
                    system_text = stripped

            if not user_message:
                return jsonify({"error": "user_message is required"}), 400
            if state.conversation_llm is None:
                return jsonify({"error": "Conversation LLM is not initialized"}), 503

            history_entries = _normalize_history_entries(history)
            max_ctx = config.OPENAI_CONVERSATION_MAX_CONTEXT_TOKENS
            model_name = config.OPENAI_CONVERSATION_MODEL

            messages, est_input_tokens, trimmed_pairs = _build_messages_with_context_cap(
                system_text,
                history_entries,
                user_message,
                max_ctx,
                model_name,
            )
            if trimmed_pairs:
                logger.info(
                    "[/ai-chat-message] Trimmed %d oldest message pair(s) (~%s est. input tokens, cap=%s)",
                    trimmed_pairs,
                    est_input_tokens,
                    max_ctx,
                )

            api_start = time.time()
            response = state.conversation_llm.invoke(messages)
            api_latency_ms = int((time.time() - api_start) * 1000)

            usage = _extract_usage(response)
            response_metadata = getattr(response, "response_metadata", {}) or {}
            model_name = response_metadata.get("model_name") or model_name
            finish_reason = response_metadata.get("finish_reason")

            raw_content = response.content
            if isinstance(raw_content, list):
                # Multimodal / structured content edge case
                answer = ""
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        answer += part.get("text") or ""
                    elif isinstance(part, str):
                        answer += part
            else:
                answer = (raw_content or "") if isinstance(raw_content, str) else str(raw_content or "")

            request_ms = int((time.time() - request_start) * 1000)
            logger.info(
                "[/ai-chat-message] OK in %dms (api %dms, total_tokens=%s, history_turns=%s)",
                request_ms,
                api_latency_ms,
                usage.get("total_tokens"),
                len(history_entries),
            )

            return jsonify({
                "answer": answer.strip(),
                "metrics": {
                    "model": model_name,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "latency_ms": api_latency_ms,
                    "request_ms": request_ms,
                    "finish_reason": finish_reason,
                    "history_turns_sent": len(history_entries),
                    "history_turns_trimmed_pairs": trimmed_pairs,
                    "estimated_input_tokens": est_input_tokens,
                    "context_token_cap": max_ctx,
                },
            }), 200
        except Exception as e:
            logger.error("[/ai-chat-message] Error: %s", e)
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    @app.route("/ai-chat-message/stream", methods=["POST"])
    def ai_chat_message_stream():
        """
        Same as /ai-chat-message, but streams the reply token-by-token as
        Server-Sent Events so the UI can render text as it arrives.

        Event payloads (one JSON object per `data:` line):
          {"delta": "<text chunk>"}                  incremental text
          {"done": true, "metrics": {...}}           final event with metrics
          {"error": "<message>"}                     error during streaming
        """
        request_start = time.time()
        logger.info("[/ai-chat-message/stream] Request received")

        # --- Validation happens up front so failures return a normal JSON error
        #     (with the right status) before any streaming begins. ---
        try:
            data = request.get_json() or {}
            user_message = str((data or {}).get("user_message") or "").strip()
            history = (data or {}).get("history") or []
            override_raw = (data or {}).get("system_prompt_override")
            system_text = AI_CHAT_SYSTEM
            if isinstance(override_raw, str):
                stripped = override_raw.strip()
                if stripped:
                    if len(stripped) > _AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS:
                        return jsonify({
                            "error": (
                                f"system_prompt_override exceeds {_AI_CHAT_SYSTEM_OVERRIDE_MAX_CHARS} characters"
                            ),
                        }), 400
                    system_text = stripped

            if not user_message:
                return jsonify({"error": "user_message is required"}), 400
            if state.conversation_llm is None:
                return jsonify({"error": "Conversation LLM is not initialized"}), 503

            history_entries = _normalize_history_entries(history)
            max_ctx = config.OPENAI_CONVERSATION_MAX_CONTEXT_TOKENS
            model_name = config.OPENAI_CONVERSATION_MODEL

            messages, est_input_tokens, trimmed_pairs = _build_messages_with_context_cap(
                system_text,
                history_entries,
                user_message,
                max_ctx,
                model_name,
            )
            if trimmed_pairs:
                logger.info(
                    "[/ai-chat-message/stream] Trimmed %d oldest message pair(s) (~%s est. input tokens, cap=%s)",
                    trimmed_pairs,
                    est_input_tokens,
                    max_ctx,
                )
        except Exception as e:
            logger.error("[/ai-chat-message/stream] Setup error: %s", e)
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

        def event_stream():
            def sse(payload):
                return f"data: {json.dumps(payload)}\n\n"

            api_start = time.time()
            usage = {}
            finish_reason = None
            resolved_model = model_name
            chars = 0
            try:
                for chunk in state.conversation_llm.stream(messages):
                    # Accumulate best-effort usage / metadata from chunks.
                    chunk_usage = getattr(chunk, "usage_metadata", None)
                    if chunk_usage:
                        usage = chunk_usage
                    meta = getattr(chunk, "response_metadata", {}) or {}
                    finish_reason = meta.get("finish_reason") or finish_reason
                    resolved_model = meta.get("model_name") or resolved_model

                    raw = getattr(chunk, "content", "")
                    if isinstance(raw, list):
                        text = ""
                        for part in raw:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text += part.get("text") or ""
                            elif isinstance(part, str):
                                text += part
                    else:
                        text = raw if isinstance(raw, str) else str(raw or "")

                    if text:
                        chars += len(text)
                        yield sse({"delta": text})

                api_latency_ms = int((time.time() - api_start) * 1000)
                request_ms = int((time.time() - request_start) * 1000)
                logger.info(
                    "[/ai-chat-message/stream] OK in %dms (api %dms, chars=%d, history_turns=%s)",
                    request_ms,
                    api_latency_ms,
                    chars,
                    len(history_entries),
                )
                yield sse({
                    "done": True,
                    "metrics": {
                        "model": resolved_model,
                        "prompt_tokens": (usage or {}).get("input_tokens") or (usage or {}).get("prompt_tokens"),
                        "completion_tokens": (usage or {}).get("output_tokens") or (usage or {}).get("completion_tokens"),
                        "total_tokens": (usage or {}).get("total_tokens"),
                        "latency_ms": api_latency_ms,
                        "request_ms": request_ms,
                        "finish_reason": finish_reason,
                        "history_turns_sent": len(history_entries),
                        "history_turns_trimmed_pairs": trimmed_pairs,
                        "estimated_input_tokens": est_input_tokens,
                        "context_token_cap": max_ctx,
                    },
                })
            except Exception as e:
                logger.error("[/ai-chat-message/stream] Stream error: %s", e)
                logger.error(traceback.format_exc())
                yield sse({"error": str(e)})

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable proxy buffering (e.g. nginx)
            },
        )
