"""CLI entry point for Get Accurate pipeline (used by Next.js API routes)."""
import base64
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from google import genai  # noqa: E402
from app_state import state  # noqa: E402
import config  # noqa: E402
from backend.image_utils import image_bytes_to_data_url  # noqa: E402
from services import image_service  # noqa: E402


def _ensure_clients() -> None:
    state.openai_api_key = config.OPENAI_API_KEY
    if not state.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured")
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY not configured")
    state.gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        image_b64 = payload.get("image_base64", "")
        if not image_b64:
            print(json.dumps({"error": "image_base64 is required"}))
            return 1

        image_bytes = base64.b64decode(image_b64, validate=True)
        if not image_bytes:
            print(json.dumps({"error": "image_base64 decoded to empty bytes"}))
            return 1

        _ensure_clients()

        original_prompt = payload.get("original_prompt") or ""
        include_trace = bool(payload.get("include_trace"))
        aspect_ratio = image_service.normalize_aspect_ratio(
            payload.get("aspect_ratio")
        )
        image_data_url = image_bytes_to_data_url(image_bytes)

        (
            final_filename,
            final_bytes,
            final_data_url,
            flaws_count,
            iterations,
            accuracy_trace,
            accurate_usage,
        ) = image_service.get_accurate_image(
            "",
            image_data_url,
            original_prompt or None,
            collect_trace=include_trace,
            aspect_ratio=aspect_ratio,
        )

        response = {
            "success": True,
            "filename": final_filename,
            "image_data_url": final_data_url,
            "aspect_ratio": aspect_ratio,
            "flaws_detected": flaws_count,
            "iterations": iterations,
            "usage": accurate_usage or {},
            "png_byte_length": len(final_bytes),
        }
        if include_trace and accuracy_trace is not None:
            response["accuracy_trace"] = accuracy_trace

        print(json.dumps(response))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
