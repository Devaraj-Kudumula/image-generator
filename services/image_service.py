"""
Image generation and editing via Gemini; image storage helpers.
"""
import logging
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

from PIL import Image

from backend.image_utils import (
    image_bytes_to_data_url,
    decode_image_data_url,
    extract_png_bytes_from_gemini_response,
)
from app_state import state
import config

logger = logging.getLogger(__name__)


def generate_image(prompt: str) -> Tuple[str, bytes, str]:
    """
    Generate an image using Gemini. Returns (filename, image_bytes, image_data_url).
    Raises on missing client or API errors.
    """
    if not state.gemini_client:
        raise ValueError("Gemini client not initialized")

    response = state.gemini_client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[prompt],
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'image_{timestamp}.png'

    try:
        image_bytes = extract_png_bytes_from_gemini_response(response)
    except Exception as part_error:
        raise ValueError(
            f"Error processing Gemini response: {str(part_error)}"
        ) from part_error
    if not image_bytes:
        raise ValueError("No image generated in response")

    config.IMAGE_STORE[filename] = image_bytes
    image_data_url = image_bytes_to_data_url(image_bytes)

    if not config.IS_SERVERLESS:
        try:
            config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            (config.IMAGES_DIR / filename).write_bytes(image_bytes)
            logger.info("Image saved to disk: %s", config.IMAGES_DIR / filename)
        except OSError:
            pass

    logger.info("Image stored (in-memory); filename=%s", filename)
    return (filename, image_bytes, image_data_url)


def load_image_for_edit(
    filename: str, image_data_url: Optional[str] = None
) -> Image.Image:
    """
    Load image from request data URL, in-memory store, or disk.
    Returns PIL Image. Raises ValueError if not found.
    """
    if image_data_url:
        try:
            return Image.open(
                BytesIO(decode_image_data_url(image_data_url))
            )
        except Exception as decode_error:
            logger.warning(
                "Invalid image_data_url provided: %s",
                decode_error,
            )

    if filename in config.IMAGE_STORE:
        return Image.open(BytesIO(config.IMAGE_STORE[filename]))

    if (
        not config.IS_SERVERLESS
        and filename
        and (config.IMAGES_DIR / filename).exists()
    ):
        return Image.open(config.IMAGES_DIR / filename)

    raise ValueError(
        f"File not found: {filename}. "
        "On Vercel, pass image_data_url for stateless edits."
    )


def edit_image(
    filename: str,
    changes: str,
    image_data_url: Optional[str] = None,
) -> Tuple[str, bytes, str]:
    """
    Edit an existing image with Gemini. Returns (new_filename, edited_bytes, edited_data_url).
    """
    if not state.gemini_client:
        raise ValueError("Gemini client not initialized")

    image = load_image_for_edit(filename, image_data_url)

    prompt = (
        "Edit the following image based on the requested changes:\n\n"
        f"Changes: {changes}"
    )
    try:
        response = state.gemini_client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, image],
        )
    except Exception as api_error:
        raise ValueError(f"Error calling Gemini API: {str(api_error)}") from api_error

    try:
        edited_bytes = extract_png_bytes_from_gemini_response(response)
    except Exception as part_error:
        raise ValueError(
            f"Error processing Gemini response: {str(part_error)}"
        ) from part_error
    if not edited_bytes:
        raise ValueError("No edited image generated in response")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_filename = f'edited_{timestamp}.png'

    config.IMAGE_STORE[new_filename] = edited_bytes
    edited_image_data_url = image_bytes_to_data_url(edited_bytes)

    if not config.IS_SERVERLESS:
        try:
            config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            (config.IMAGES_DIR / new_filename).write_bytes(edited_bytes)
            logger.info(
                "Edited image saved to disk: %s",
                config.IMAGES_DIR / new_filename,
            )
        except OSError:
            pass

    logger.info("Edited image stored; filename=%s", new_filename)
    return (new_filename, edited_bytes, edited_image_data_url)


def get_image_bytes(filename: str) -> Optional[bytes]:
    """Return image bytes from store or disk, or None if not found."""
    if filename in config.IMAGE_STORE:
        return config.IMAGE_STORE[filename]
    if not config.IS_SERVERLESS and (config.IMAGES_DIR / filename).exists():
        return (config.IMAGES_DIR / filename).read_bytes()
    return None
