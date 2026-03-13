"""
Image generation and editing via Gemini; image storage helpers.
"""
import logging
import math
import re
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

import openai as openai_lib
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


def _store_image_bytes(prefix: str, image_bytes: bytes) -> Tuple[str, str]:
    """Store image bytes in the IMAGE_STORE (and optionally disk). Returns (filename, data_url)."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{prefix}_{timestamp}.png'
    config.IMAGE_STORE[filename] = image_bytes
    data_url = image_bytes_to_data_url(image_bytes)
    if not config.IS_SERVERLESS:
        try:
            config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            (config.IMAGES_DIR / filename).write_bytes(image_bytes)
        except OSError:
            pass
    return filename, data_url


def get_accurate_image(
    filename: str,
    image_data_url: Optional[str] = None,
) -> Tuple[str, bytes, str, int, int]:
    """
    Use GPT-4o vision to detect label/arrow flaws in the image, then apply
    sequential Gemini correction passes (max 3 flaws per pass, max 5 passes).

    Returns (final_filename, final_bytes, final_data_url, flaws_count, iterations).
    """
    MAX_FLAWS_PER_PROMPT = 3
    MAX_ITERATIONS = 5

    if not state.gemini_client:
        raise ValueError("Gemini client not initialized")
    if not state.openai_api_key:
        raise ValueError("OpenAI API key not configured")

    # Ensure we have a data URL to pass to GPT-4 vision. OpenAI cannot fetch
    # localhost or server-relative URLs, so we must send base64 data only.
    if not image_data_url or not (image_data_url.strip().lower().startswith("data:")):
        image_pil = load_image_for_edit(filename, image_data_url or None)
        buf = BytesIO()
        image_pil.save(buf, format='PNG')
        image_data_url = image_bytes_to_data_url(buf.getvalue())

    # --- Step 1: detect flaws with best available vision model (gpt-5.4) ---
    oa_client = openai_lib.OpenAI(api_key=state.openai_api_key)

    flaw_detection_prompt = (
        "Examine this diagram image with extreme care.\n\n"
        "STEP 1 — Inventory: List every visible label (text annotation) and every arrow "
        "present in the image.\n\n"
        "STEP 2 — Verify each item against your medical/scientific knowledge:\n"
        "  • Is each label text anatomically/scientifically correct for the structure it "
        "annotates?\n"
        "  • Is each arrow pointing in the correct direction?\n"
        "  • Is each arrow connected to the correct elements?\n"
        "  • Are any labels placed on the wrong structure?\n\n"
        "STEP 3 — Report flaws in a numbered list. Order by severity:\n"
        "  - List the biggest/most critical flaws FIRST (wrong labels, wrong arrow directions, "
        "misplaced annotations, major anatomical errors).\n"
        "  - Then list minor/less significant flaws (typos, slight misplacements, cosmetic issues).\n"
        "  - Each item must concisely describe ONE flaw: what is wrong and what it should be.\n"
        "  - If flaws exist: output a single numbered list (1. ... 2. ...) with the most critical "
        "flaw as #1 and the least critical last. Enumerate all inaccuracies in this order.\n"
        "  - If you find absolutely no errors after this thorough review: output the single "
        "token NO_FLAWS_DETECTED and nothing else.\n\n"
        "Do NOT skip steps. Do NOT be lenient — even subtle anatomical or directional errors "
        "must be reported. Be as thorough as a human expert reviewing the same image."
    )

    vision_response = oa_client.chat.completions.create(
        model="gpt-5.4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous medical illustration quality-control expert. "
                    "Your sole job is to detect inaccuracies in labels and arrows in "
                    "scientific diagrams. You are thorough, critical, and never skip "
                    "verification. When asked to review a diagram you always find "
                    "every error, no matter how subtle."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": flaw_detection_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}},
                ],
            },
        ],
        max_completion_tokens=2000,
    )

    flaw_text = vision_response.choices[0].message.content.strip()
    logger.info(
        "--- Vision model flaw detection INPUT ---\n"
        "[system]: You are a rigorous medical illustration quality-control expert. "
        "Your sole job is to detect inaccuracies in labels and arrows in scientific "
        "diagrams. You are thorough, critical, and never skip verification. When asked "
        "to review a diagram you always find every error, no matter how subtle.\n\n"
        "[user prompt]:\n%s",
        flaw_detection_prompt,
    )
    logger.info("--- Vision model flaw detection OUTPUT ---\n%s", flaw_text)

    # --- Step 2: handle no-flaw case ---
    def _return_original_as_accurate() -> Tuple[str, bytes, str, int, int]:
        image_pil = load_image_for_edit(filename, image_data_url)
        buf = BytesIO()
        image_pil.save(buf, format='PNG')
        orig_bytes = buf.getvalue()
        acc_filename, acc_data_url = _store_image_bytes('accurate', orig_bytes)
        return (acc_filename, orig_bytes, acc_data_url, 0, 0)

    # Match the exact keyword OR common natural-language "no flaws" phrases
    _no_flaw_pattern = re.compile(
        r'no[_ ]flaws?[_ ]detected|no errors? found|no inaccuracies|looks? correct|'
        r'no issues? found|everything (is |looks? )?correct',
        re.IGNORECASE,
    )
    if flaw_text.upper().strip() == 'NO_FLAWS_DETECTED' or (
        len(flaw_text) < 80 and _no_flaw_pattern.search(flaw_text)
    ):
        logger.info("No flaws detected in the image")
        return _return_original_as_accurate()

    # --- Step 3: parse the numbered flaw list ---
    _no_flaw_line = re.compile(
        r'^(no[_ ]flaws?|none found|no errors?|no inaccuracies|looks? (good|correct)|'
        r'everything (is |looks? )?correct)',
        re.IGNORECASE,
    )
    flaws = []
    for line in flaw_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        clean = re.sub(r'^[\d]+[.):\-]\s*', '', line)
        clean = re.sub(r'^[-•*]\s*', '', clean).strip()
        # Skip lines that are just "no flaws" variants or step headers
        if not clean or _no_flaw_line.match(clean):
            continue
        if re.match(r'^step\s+\d', clean, re.IGNORECASE):
            continue
        flaws.append(clean)

    if not flaws:
        logger.warning("Could not parse flaws from response; returning original")
        return _return_original_as_accurate()

    # Cap flaws to what fits in 5 passes of 3 flaws each; ignore the rest
    max_flaws_used = MAX_ITERATIONS * MAX_FLAWS_PER_PROMPT
    if len(flaws) > max_flaws_used:
        logger.info(
            "Detected %d flaws; using first %d (max %d passes × %d flaws), ignoring the rest",
            len(flaws), max_flaws_used, MAX_ITERATIONS, MAX_FLAWS_PER_PROMPT,
        )
        flaws = flaws[:max_flaws_used]
    else:
        logger.info("Detected %d flaws", len(flaws))

    # --- Step 4: build grouped correction prompts (max 3 flaws per pass, max 5 passes) ---
    num_iterations = min(math.ceil(len(flaws) / MAX_FLAWS_PER_PROMPT), MAX_ITERATIONS)
    correction_prompts = []
    for i in range(num_iterations):
        batch = flaws[i * MAX_FLAWS_PER_PROMPT: (i + 1) * MAX_FLAWS_PER_PROMPT]
        prompt_text = (
            "Fix the following specific issues in this diagram image:\n"
            + "\n".join(f"- {f}" for f in batch)
        )
        correction_prompts.append(prompt_text)

    logger.info("Applying %d correction pass(es)", len(correction_prompts))

    # --- Step 5: sequential Gemini refinement ---
    current_filename = filename
    current_data_url = image_data_url
    current_bytes: bytes = b''

    for i, correction_prompt in enumerate(correction_prompts):
        logger.info(
            "Correction pass %d/%d: %s...",
            i + 1, len(correction_prompts), correction_prompt[:100],
        )
        current_filename, current_bytes, current_data_url = edit_image(
            current_filename, correction_prompt, current_data_url
        )

    return (current_filename, current_bytes, current_data_url, len(flaws), len(correction_prompts))
