"""
Track LLM usage across GPT and Gemini calls.
"""
import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional

from flask import has_request_context, request

import config

_PROVIDER_KEYS = ("gpt", "gemini")


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


class LlmMetricsService:
    """In-memory tracker for LLM call counts, tokens, and estimated costs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_started_unix = int(time.time())
        self._state = self._new_state()

    def _new_provider_state(self) -> Dict[str, Any]:
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    def _new_state(self) -> Dict[str, Any]:
        return {
            "session_started_unix": self._session_started_unix,
            "overall": {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "last_updated_unix": None,
            },
            "providers": {
                "gpt": self._new_provider_state(),
                "gemini": self._new_provider_state(),
            },
            "chats": {},
            "recent_calls": [],
        }

    def _new_chat_state(self, chat_id: str) -> Dict[str, Any]:
        return {
            "chat_id": chat_id,
            "overall": {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "last_updated_unix": None,
            },
            "providers": {
                "gpt": self._new_provider_state(),
                "gemini": self._new_provider_state(),
            },
        }

    def _resolve_chat_id(self, explicit_chat_id: Optional[str]) -> str:
        if explicit_chat_id and str(explicit_chat_id).strip():
            return str(explicit_chat_id).strip()

        if has_request_context():
            header_chat = (request.headers.get("X-Chat-Id") or "").strip()
            if header_chat:
                return header_chat

            arg_chat = (request.args.get("chat_id") or "").strip()
            if arg_chat:
                return arg_chat

            try:
                payload = request.get_json(silent=True) or {}
            except Exception:
                payload = {}
            body_chat = (
                str(payload.get("chat_id", "")).strip()
                if isinstance(payload, dict)
                else ""
            )
            if body_chat:
                return body_chat

        return "default"

    def _pricing_for(self, provider: str, model_name: str) -> Dict[str, float]:
        model = (model_name or "").lower().strip()
        if provider == "gpt":
            table = config.OPENAI_PRICING_PER_1M_TOKENS
            if model.startswith("gpt-5"):
                return table.get("gpt-5", table.get("default", {"input": 0.0, "output": 0.0}))
            if model.startswith("gpt-4"):
                return table.get("gpt-4", table.get("default", {"input": 0.0, "output": 0.0}))
            return table.get("default", {"input": 0.0, "output": 0.0})

        table = config.GEMINI_PRICING_PER_1M_TOKENS
        if model.startswith("gemini-3"):
            return table.get("gemini-3", table.get("default", {"input": 0.0, "output": 0.0}))
        if model.startswith("gemini-2"):
            return table.get("gemini-2", table.get("default", {"input": 0.0, "output": 0.0}))
        return table.get("default", {"input": 0.0, "output": 0.0})

    def _estimate_cost(
        self,
        provider: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        rates = self._pricing_for(provider, model_name)
        input_rate = float(rates.get("input", 0.0))
        output_rate = float(rates.get("output", 0.0))
        cost = ((prompt_tokens / 1_000_000) * input_rate) + (
            (completion_tokens / 1_000_000) * output_rate
        )
        return round(cost, 8)

    def record_call(
        self,
        provider: str,
        model_name: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        provider_key = (provider or "").strip().lower()
        if provider_key not in _PROVIDER_KEYS:
            return

        prompt = _safe_int(prompt_tokens)
        completion = _safe_int(completion_tokens)
        total = _safe_int(total_tokens)
        if total <= 0:
            total = prompt + completion

        estimated = (
            round(float(estimated_cost_usd), 8)
            if estimated_cost_usd is not None
            else self._estimate_cost(provider_key, model_name, prompt, completion)
        )

        resolved_chat_id = self._resolve_chat_id(chat_id)

        call_entry = {
            "chat_id": resolved_chat_id,
            "provider": provider_key,
            "model": model_name,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "estimated_cost_usd": estimated,
            "timestamp_unix": int(time.time()),
        }

        with self._lock:
            provider_state = self._state["providers"][provider_key]
            overall = self._state["overall"]
            chats = self._state["chats"]
            if resolved_chat_id not in chats:
                chats[resolved_chat_id] = self._new_chat_state(resolved_chat_id)
            chat_state = chats[resolved_chat_id]
            chat_provider = chat_state["providers"][provider_key]
            chat_overall = chat_state["overall"]

            provider_state["calls"] += 1
            provider_state["prompt_tokens"] += prompt
            provider_state["completion_tokens"] += completion
            provider_state["total_tokens"] += total
            provider_state["estimated_cost_usd"] = round(
                provider_state["estimated_cost_usd"] + estimated,
                8,
            )

            overall["calls"] += 1
            overall["prompt_tokens"] += prompt
            overall["completion_tokens"] += completion
            overall["total_tokens"] += total
            overall["estimated_cost_usd"] = round(
                overall["estimated_cost_usd"] + estimated,
                8,
            )
            overall["last_updated_unix"] = call_entry["timestamp_unix"]

            chat_provider["calls"] += 1
            chat_provider["prompt_tokens"] += prompt
            chat_provider["completion_tokens"] += completion
            chat_provider["total_tokens"] += total
            chat_provider["estimated_cost_usd"] = round(
                chat_provider["estimated_cost_usd"] + estimated,
                8,
            )

            chat_overall["calls"] += 1
            chat_overall["prompt_tokens"] += prompt
            chat_overall["completion_tokens"] += completion
            chat_overall["total_tokens"] += total
            chat_overall["estimated_cost_usd"] = round(
                chat_overall["estimated_cost_usd"] + estimated,
                8,
            )
            chat_overall["last_updated_unix"] = call_entry["timestamp_unix"]

            self._state["recent_calls"].append(call_entry)
            if len(self._state["recent_calls"]) > config.LLM_METRICS_RECENT_CALL_LIMIT:
                self._state["recent_calls"] = self._state["recent_calls"][-config.LLM_METRICS_RECENT_CALL_LIMIT:]

    def get_snapshot(self, chat_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            snapshot = deepcopy(self._state)

        if chat_id and str(chat_id).strip():
            chat_key = str(chat_id).strip()
            chat_snapshot = snapshot.get("chats", {}).get(chat_key)
            if chat_snapshot is None:
                chat_snapshot = self._new_chat_state(chat_key)
            chat_snapshot["session_started_unix"] = snapshot.get(
                "session_started_unix"
            )
            chat_snapshot["scope"] = "chat"
            snapshot = chat_snapshot
        else:
            snapshot["scope"] = "global"

        snapshot["cost_currency"] = "USD"
        snapshot["cost_note"] = "Estimated cost from configured per-1M token rates."
        return snapshot


llm_metrics = LlmMetricsService()


def _extract_langchain_openai_usage(response: Any) -> Dict[str, int]:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = {}
    if isinstance(metadata, dict):
        usage = metadata.get("token_usage") or metadata.get("usage") or {}

    prompt_tokens = _safe_int(
        usage.get("prompt_tokens") if isinstance(usage, dict) else None
    )
    completion_tokens = _safe_int(
        usage.get("completion_tokens") if isinstance(usage, dict) else None
    )
    total_tokens = _safe_int(
        usage.get("total_tokens") if isinstance(usage, dict) else None
    )

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            prompt_tokens = _safe_int(usage_metadata.get("input_tokens"))
            completion_tokens = _safe_int(usage_metadata.get("output_tokens"))
            total_tokens = _safe_int(usage_metadata.get("total_tokens"))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def record_langchain_openai_call(
    response: Any,
    model_name: str,
    chat_id: Optional[str] = None,
) -> None:
    usage = _extract_langchain_openai_usage(response)
    llm_metrics.record_call(
        provider="gpt",
        model_name=model_name,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        chat_id=chat_id,
    )


def record_openai_sdk_call(
    response: Any,
    model_name: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> None:
    usage = getattr(response, "usage", None)
    prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", 0))
    completion_tokens = _safe_int(getattr(usage, "completion_tokens", 0))
    total_tokens = _safe_int(getattr(usage, "total_tokens", 0))

    resolved_model = model_name or getattr(response, "model", "gpt-unknown")
    llm_metrics.record_call(
        provider="gpt",
        model_name=resolved_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        chat_id=chat_id,
    )


def _extract_gemini_usage(response: Any) -> Dict[str, int]:
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is None and isinstance(response, dict):
        usage_meta = response.get("usage_metadata")

    if usage_meta is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    if isinstance(usage_meta, dict):
        prompt_tokens = _safe_int(
            usage_meta.get("prompt_token_count")
            or usage_meta.get("input_token_count")
            or usage_meta.get("prompt_tokens")
        )
        completion_tokens = _safe_int(
            usage_meta.get("candidates_token_count")
            or usage_meta.get("output_token_count")
            or usage_meta.get("completion_tokens")
        )
        total_tokens = _safe_int(
            usage_meta.get("total_token_count")
            or usage_meta.get("total_tokens")
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    return {
        "prompt_tokens": _safe_int(
            getattr(usage_meta, "prompt_token_count", None)
            or getattr(usage_meta, "input_token_count", None)
            or getattr(usage_meta, "prompt_tokens", None)
        ),
        "completion_tokens": _safe_int(
            getattr(usage_meta, "candidates_token_count", None)
            or getattr(usage_meta, "output_token_count", None)
            or getattr(usage_meta, "completion_tokens", None)
        ),
        "total_tokens": _safe_int(
            getattr(usage_meta, "total_token_count", None)
            or getattr(usage_meta, "total_tokens", None)
        ),
    }


def record_gemini_call(
    response: Any,
    model_name: str,
    chat_id: Optional[str] = None,
) -> None:
    usage = _extract_gemini_usage(response)
    llm_metrics.record_call(
        provider="gemini",
        model_name=model_name,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        chat_id=chat_id,
    )
