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

