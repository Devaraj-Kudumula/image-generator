"""CLI entry point for PNG→SVG vectorization (used by Next.js API routes)."""
import base64
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import vectorize_service  # noqa: E402


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

        include_meta = bool(payload.get("include_meta"))
        debug_dump = bool(payload.get("debug_dump"))

        if include_meta or debug_dump:
            svg_string, trace_meta = vectorize_service.vectorize_png_to_svg_with_meta(
                image_bytes
            )
        else:
            svg_string = vectorize_service.vectorize_png_to_svg(image_bytes)
            trace_meta = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        svg_filename = f"vector_{timestamp}.svg"

        response = {
            "success": True,
            "svg": svg_string,
            "svg_filename": svg_filename,
        }
        if trace_meta is not None:
            response["trace_meta"] = trace_meta
        if debug_dump:
            debug_filename = vectorize_service.dump_svg_for_debug(
                svg_string, prefix="vector_debug"
            )
            if debug_filename:
                response["debug_svg_filename"] = debug_filename

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
