import base64
import binascii
from io import BytesIO
from typing import Optional

from PIL import Image


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    """Encode PNG bytes as a data URL for stateless client usage."""
    return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"


def decode_image_data_url(image_data_url: str) -> bytes:
    """Decode a data URL into raw image bytes."""
    if not isinstance(image_data_url, str) or not image_data_url.strip():
        raise ValueError("image_data_url is empty")

    normalized = image_data_url.strip()
    if normalized.startswith("data:"):
        _, _, encoded = normalized.partition(",")
        if not encoded:
            raise ValueError("image_data_url is malformed")
        normalized = encoded

    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as decode_error:
        raise ValueError("image_data_url is not valid base64") from decode_error


def extract_png_bytes_from_gemini_response(response) -> Optional[bytes]:
    """
    Extract PNG image bytes from a Gemini API response.

    Returns None if no inline image data is found.
    """
    if not response or not getattr(response, "candidates", None):
        return None

    first_candidate = response.candidates[0]
    content = getattr(first_candidate, "content", None)
    if not content or not getattr(content, "parts", None):
        return None

    for part in content.parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            image = Image.open(BytesIO(inline_data.data))
            buf = BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()

    return None


def diagnose_missing_image(response) -> str:
    """
    Explain why a 200-OK Gemini response carried no image (safety block, refusal,
    text-only reply, etc.). Used to turn an opaque 500 into an actionable message.
    """
    if not response:
        return "empty response object"
    candidates = getattr(response, "candidates", None)
    if not candidates:
        # Prompt-level block lives on prompt_feedback, not candidates.
        feedback = getattr(response, "prompt_feedback", None)
        block = getattr(feedback, "block_reason", None) if feedback else None
        if block:
            return f"prompt blocked by safety filter (block_reason={block})"
        return "no candidates returned"

    cand = candidates[0]
    bits = []
    finish = getattr(cand, "finish_reason", None)
    if finish:
        bits.append(f"finish_reason={finish}")
    safety = getattr(cand, "safety_ratings", None)
    if safety:
        flagged = [
            f"{getattr(r, 'category', '?')}={getattr(r, 'probability', '?')}"
            for r in safety
            if getattr(r, "blocked", False) or str(getattr(r, "probability", "")).upper() in ("HIGH", "MEDIUM")
        ]
        if flagged:
            bits.append("safety=" + ", ".join(flagged))

    # Any text the model returned instead of an image.
    content = getattr(cand, "content", None)
    parts = getattr(content, "parts", None) if content else None
    texts = [getattr(p, "text", None) for p in (parts or []) if getattr(p, "text", None)]
    if texts:
        snippet = " ".join(texts).strip().replace("\n", " ")
        bits.append(f'model returned text instead of image: "{snippet[:200]}"')

    return "; ".join(bits) if bits else "response had no image part and no diagnostic info"

