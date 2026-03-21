"""
Image generation and editing via Gemini; image storage helpers.
"""
import logging
import math
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import openai as openai_lib
from PIL import Image

from backend.image_utils import (
    image_bytes_to_data_url,
    decode_image_data_url,
    extract_png_bytes_from_gemini_response,
)
from app_state import state
import config
from services.llm_metrics_service import record_gemini_call, record_openai_sdk_call

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
    record_gemini_call(response, "gemini-3-pro-image-preview")

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


def _summarize_image_for_trace(
    image_data_url: Optional[str], filename_hint: Optional[str] = None
) -> Dict[str, Any]:
    """Compact description of an image payload for UI trace (no raw base64)."""
    out: Dict[str, Any] = {}
    if filename_hint:
        out["filename"] = filename_hint
    if not image_data_url:
        out["note"] = "No data URL (image loaded by filename from server store)."
        return out
    n = len(image_data_url)
    lower = image_data_url.strip().lower()
    if lower.startswith("data:"):
        semi = image_data_url.find(";")
        mime = image_data_url[5:semi] if semi > 5 else "unknown"
        out["mime_type"] = mime
        out["char_length"] = n
        out["note"] = "Image sent to the model; payload omitted from this log."
    else:
        out["char_length"] = n
        out["note"] = "Image reference; full payload omitted from this log."
    return out


def edit_image(
    filename: str,
    changes: str,
    image_data_url: Optional[str] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
    trace_step_id: Optional[str] = None,
    trace_title: Optional[str] = None,
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
        record_gemini_call(response, "gemini-3-pro-image-preview")
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
    if trace is not None and trace_step_id:
        trace.append(
            {
                "id": trace_step_id,
                "title": trace_title or trace_step_id,
                "provider": "google",
                "model": "gemini-3-pro-image-preview",
                "input": {
                    "filename": filename,
                    "edit_instruction": changes,
                    "full_prompt": prompt,
                    "source_image": _summarize_image_for_trace(
                        image_data_url, filename
                    ),
                },
                "output": {
                    "filename": new_filename,
                    "png_byte_length": len(edited_bytes),
                    "note": "Edited PNG from Gemini; image shown in the chat.",
                },
            }
        )
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NO_FLAW_PATTERN = re.compile(
    r'no[_ ]flaws?[_ ]detected|no errors? found|no inaccuracies|looks? correct|'
    r'no issues? found|everything (is |looks? )?correct',
    re.IGNORECASE,
)
_NO_FLAW_LINE = re.compile(
    r'^(no[_ ]flaws?|none found|no errors?|no inaccuracies|looks? (good|correct)|'
    r'everything (is |looks? )?correct)',
    re.IGNORECASE,
)


def _parse_flaw_lines(text: str) -> List[str]:
    """Parse a numbered/bulleted flaw list from model output into a clean list of strings."""
    flaws = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        clean = re.sub(r'^[\d]+[.):\-]\s*', '', line)
        clean = re.sub(r'^[-•*]\s*', '', clean).strip()
        if not clean or _NO_FLAW_LINE.match(clean):
            continue
        if re.match(r'^step\s+\d', clean, re.IGNORECASE):
            continue
        flaws.append(clean)
    return flaws


def _is_no_flaw_response(text: str) -> bool:
    """Return True if the model response indicates no flaws were found."""
    stripped = text.strip()
    if stripped.upper() == 'NO_FLAWS_DETECTED':
        return True
    if len(stripped) < 80 and _NO_FLAW_PATTERN.search(stripped):
        return True
    return False


def _detect_flaws_via_openai(
    oa_client: openai_lib.OpenAI,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    model: str = "gpt-5.4",
    tag: str = "flaw-detection",
    trace: Optional[List[Dict[str, Any]]] = None,
    trace_step_id: str = "openai-vision",
    trace_title: str = "OpenAI vision",
) -> str:
    """Call OpenAI vision model and return raw text response."""
    response = oa_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}},
                ],
            },
        ],
        max_completion_tokens=2000,
    )
    record_openai_sdk_call(response, model)
    text = response.choices[0].message.content.strip()
    logger.info("--- [%s] INPUT ---\n[system]: %s\n\n[user]: %s", tag, system_prompt, user_prompt)
    logger.info("--- [%s] OUTPUT ---\n%s", tag, text)
    if trace is not None:
        trace.append(
            {
                "id": trace_step_id,
                "title": trace_title,
                "provider": "openai",
                "model": model,
                "input": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "image": _summarize_image_for_trace(image_data_url),
                },
                "output": {"text": text},
            }
        )
    return text


# ---------------------------------------------------------------------------
# get_accurate_image  (structural accuracy first, label polish as final pass)
# ---------------------------------------------------------------------------

def get_accurate_image(
    filename: str,
    image_data_url: Optional[str] = None,
    original_prompt: Optional[str] = None,
    collect_trace: bool = False,
) -> Tuple[str, bytes, str, int, int, Optional[List[Dict[str, Any]]]]:
    """
    Two-stage accuracy pipeline:

    DETECTION  — Two OpenAI vision calls:
      Stage A: Structural flaws (anatomy, proportions, topology)
      Stage B: Label/text flaws (correct names, arrow targets, legibility)

    CORRECTION — Sequential Gemini passes in this order:
      Pass 1..N : Structural fixes  (OpenAI-generated correction prompts, batched 3 flaws each)
      Pass N+1  : Label polish      (OpenAI-generated, always the final pass — fixes label
                                     errors AND cleans up any text distortion from earlier passes)

    OpenAI generates every correction prompt from the detected flaws + original intent,
    so Gemini always receives a high-quality, targeted instruction rather than a raw flaw dump.

    Returns (final_filename, final_bytes, final_data_url, total_flaws_count, iterations,
             accuracy_trace or None). When collect_trace is True, the last element is a list of
             per-call input/output records for the UI; otherwise None.
    """
    MAX_FLAWS_PER_PROMPT = 3
    MAX_STRUCTURAL_ITERATIONS = 4  # structural batches (up to 12 structural flaws addressed)

    if not state.gemini_client:
        raise ValueError("Gemini client not initialized")
    if not state.openai_api_key:
        raise ValueError("OpenAI API key not configured")

    # Ensure we have a base64 data URL — OpenAI vision cannot fetch localhost URLs.
    if not image_data_url or not image_data_url.strip().lower().startswith("data:"):
        image_pil = load_image_for_edit(filename, image_data_url or None)
        buf = BytesIO()
        image_pil.save(buf, format='PNG')
        image_data_url = image_bytes_to_data_url(buf.getvalue())

    oa_client = openai_lib.OpenAI(api_key=state.openai_api_key)
    trace: Optional[List[Dict[str, Any]]] = [] if collect_trace else None

    intent_snippet = (
        f"\n\nORIGINAL PROMPT (preserve this intent):\n{original_prompt.strip()}"
        if original_prompt and original_prompt.strip()
        else ""
    )

    # -----------------------------------------------------------------------
    # Stage A — OpenAI detects structural flaws
    # -----------------------------------------------------------------------
    structural_detection_system = (
        "You are a rigorous medical illustration quality-control expert. "
        "Your sole job is to detect inaccuracies in the structural design of "
        "scientific diagrams (anatomy, proportions, spatial relationships, topology). "
        "You are thorough, critical, and never lenient — report every structural error, "
        "no matter how subtle."
    )

    structural_detection_prompt = (
        "Examine this diagram image with extreme care.\n\n"
        "STEP 1 — Inventory: Describe the overall structural design — shapes, "
        "relative positions, proportions, and connections between parts.\n\n"
        "STEP 2 — Verify the structural design against your medical/scientific knowledge:\n"
        "  • Are structures anatomically/scientifically correct in shape and form?\n"
        "  • Are sizes and proportions realistic relative to each other?\n"
        "  • Are spatial relationships and topology (what connects to what, and where) accurate?\n"
        "  • Are any components missing, duplicated, distorted, or in the wrong location?\n\n"
        "STEP 3 — Report structural flaws as a numbered list, most critical first. "
        "Each item: ONE flaw and what it should be instead.\n"
        "If there are absolutely no structural errors: output only NO_FLAWS_DETECTED."
        + (
            f"\n\nORIGINAL PROMPT — use this to prioritise which structural properties matter most:\n"
            f"{original_prompt.strip()}"
            if original_prompt and original_prompt.strip() else ""
        )
    )

    structural_flaw_text = _detect_flaws_via_openai(
        oa_client,
        structural_detection_system,
        structural_detection_prompt,
        image_data_url,
        tag="structural-detection",
        trace=trace,
        trace_step_id="structural-detection",
        trace_title="Structural flaw detection (OpenAI vision)",
    )

    # -----------------------------------------------------------------------
    # Stage B — OpenAI detects label & text flaws
    # -----------------------------------------------------------------------
    label_detection_system = (
        "You are a rigorous medical illustration quality-control expert specialising in "
        "label and annotation accuracy. You check spelling, anatomical correctness of each "
        "label name, arrow targets, and visual legibility of all text. "
        "You are thorough and never lenient — report every label error, however small."
    )

    label_detection_prompt = (
        "Examine this diagram image with extreme care, focusing exclusively on labels, "
        "annotations, callout lines, and arrows.\n\n"
        "STEP 1 — Inventory: List every label, annotation, and arrow visible.\n\n"
        "STEP 2 — Verify each label and arrow:\n"
        "  • Is the label text spelled correctly?\n"
        "  • Does the label correctly name the structure it refers to?\n"
        "  • Is the arrow/callout pointing to the correct structure?\n"
        "  • Is the text clean, undistorted, and fully legible "
        "(no blurring, warping, overlapping, or garbling)?\n"
        "  • Are any labels missing, duplicated, or on the wrong structure?\n\n"
        "STEP 3 — Report label/annotation flaws as a numbered list, most critical first. "
        "Each item: ONE flaw, what is wrong, and what it should say or point to instead.\n"
        "If there are absolutely no label errors: output only NO_FLAWS_DETECTED."
        + (
            f"\n\nORIGINAL PROMPT for context:\n{original_prompt.strip()}"
            if original_prompt and original_prompt.strip() else ""
        )
    )

    label_flaw_text = _detect_flaws_via_openai(
        oa_client,
        label_detection_system,
        label_detection_prompt,
        image_data_url,
        tag="label-detection",
        trace=trace,
        trace_step_id="label-detection",
        trace_title="Label & annotation flaw detection (OpenAI vision)",
    )

    # -----------------------------------------------------------------------
    # Parse flaw lists
    # -----------------------------------------------------------------------
    structural_flaws: List[str] = (
        [] if _is_no_flaw_response(structural_flaw_text)
        else _parse_flaw_lines(structural_flaw_text)
    )
    label_flaws: List[str] = (
        [] if _is_no_flaw_response(label_flaw_text)
        else _parse_flaw_lines(label_flaw_text)
    )

    logger.info(
        "Detected %d structural flaw(s) and %d label flaw(s)",
        len(structural_flaws), len(label_flaws),
    )

    total_flaws = len(structural_flaws) + len(label_flaws)

    def _return_original_as_accurate() -> Tuple[str, bytes, str, int, int]:
        image_pil = load_image_for_edit(filename, image_data_url)
        buf = BytesIO()
        image_pil.save(buf, format='PNG')
        orig_bytes = buf.getvalue()
        acc_filename, acc_data_url = _store_image_bytes('accurate', orig_bytes)
        if trace is not None:
            trace.append(
                {
                    "id": "result",
                    "title": "Final result",
                    "provider": "app",
                    "model": "",
                    "input": {},
                    "output": {
                        "message": "No flaws detected; original image stored as accurate result.",
                        "filename": acc_filename,
                        "png_byte_length": len(orig_bytes),
                    },
                }
            )
        return (acc_filename, orig_bytes, acc_data_url, 0, 0)

    if not structural_flaws and not label_flaws:
        logger.info("No flaws detected — returning original as accurate")
        fn, bs, du, fc, it = _return_original_as_accurate()
        return (fn, bs, du, fc, it, trace)

    # -----------------------------------------------------------------------
    # OpenAI generates structural correction prompts (batched, 3 flaws each)
    # -----------------------------------------------------------------------
    max_structural_flaws = MAX_STRUCTURAL_ITERATIONS * MAX_FLAWS_PER_PROMPT
    if len(structural_flaws) > max_structural_flaws:
        logger.info(
            "Capping structural flaws from %d to %d",
            len(structural_flaws), max_structural_flaws,
        )
        structural_flaws = structural_flaws[:max_structural_flaws]

    num_structural_passes = min(
        math.ceil(len(structural_flaws) / MAX_FLAWS_PER_PROMPT),
        MAX_STRUCTURAL_ITERATIONS,
    )

    correction_prompts: List[str] = []

    structural_prompt_system = (
        "You are an expert at writing precise image-editing instructions for "
        "AI image models. Given a list of structural flaws in a scientific diagram "
        "and the original generation intent, write a single, clear, actionable "
        "editing instruction that tells the image model exactly what to fix. "
        "Be specific about what is wrong and what the correct version should look like. "
        "Do NOT fix labels or text — structural changes only. "
        "Output the instruction as plain text (no preamble, no bullet points)."
    )

    for i in range(num_structural_passes):
        batch = structural_flaws[i * MAX_FLAWS_PER_PROMPT: (i + 1) * MAX_FLAWS_PER_PROMPT]
        flaw_list = "\n".join(f"- {f}" for f in batch)
        structural_user_msg = f"Structural flaws to fix:\n{flaw_list}" + intent_snippet

        # Ask OpenAI to turn the raw flaw list into a precise Gemini edit instruction
        openai_prompt_gen_response = oa_client.chat.completions.create(
            model="gpt-5.4",
            temperature=0,
            messages=[
                {"role": "system", "content": structural_prompt_system},
                {"role": "user", "content": structural_user_msg},
            ],
            max_completion_tokens=500,
        )
        record_openai_sdk_call(openai_prompt_gen_response, "gpt-5.4")
        generated_prompt = openai_prompt_gen_response.choices[0].message.content.strip()
        logger.info("Generated structural correction prompt %d/%d:\n%s", i + 1, num_structural_passes, generated_prompt)
        if trace is not None:
            trace.append(
                {
                    "id": f"structural-prompt-gen-{i + 1}",
                    "title": (
                        f"OpenAI: structural edit instruction "
                        f"({i + 1}/{num_structural_passes})"
                    ),
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "input": {
                        "system_prompt": structural_prompt_system,
                        "user_prompt": structural_user_msg,
                    },
                    "output": {"generated_edit_instruction": generated_prompt},
                }
            )
        correction_prompts.append(generated_prompt)

    # -----------------------------------------------------------------------
    # OpenAI generates the final label polish prompt (always the last pass)
    # This combines all detected label flaws + a standing instruction to clean
    # up any text distortion introduced by earlier structural edit passes.
    # -----------------------------------------------------------------------
    label_flaw_summary = (
        "\n".join(f"- {f}" for f in label_flaws)
        if label_flaws
        else "No specific label errors detected, but re-render all text cleanly."
    )

    label_polish_system = (
        "You are an expert at writing precise image-editing instructions for "
        "AI image models. Given a list of label/annotation flaws in a scientific diagram "
        "and the original generation intent, write a single, clear, actionable "
        "editing instruction that tells the image model exactly what to fix. "
        "The instruction must: fix every listed label error, ensure all arrows point "
        "to the correct structures, and re-render ALL text in a clean sans-serif font "
        "with no blurring, warping, distortion, or overlapping — even if no specific "
        "label errors were found, because prior edit passes may have degraded text quality. "
        "Do NOT change any underlying structures or anatomy. "
        "Output the instruction as plain text (no preamble, no bullet points)."
    )
    label_polish_user = f"Label/annotation flaws to fix:\n{label_flaw_summary}" + intent_snippet

    label_polish_gen_response = oa_client.chat.completions.create(
        model="gpt-5.4",
        temperature=0,
        messages=[
            {"role": "system", "content": label_polish_system},
            {"role": "user", "content": label_polish_user},
        ],
        max_completion_tokens=500,
    )
    record_openai_sdk_call(label_polish_gen_response, "gpt-5.4")
    label_polish_prompt = label_polish_gen_response.choices[0].message.content.strip()
    logger.info("Generated label polish prompt:\n%s", label_polish_prompt)
    if trace is not None:
        trace.append(
            {
                "id": "label-polish-prompt-gen",
                "title": "OpenAI: label polish edit instruction",
                "provider": "openai",
                "model": "gpt-5.4",
                "input": {
                    "system_prompt": label_polish_system,
                    "user_prompt": label_polish_user,
                },
                "output": {"generated_edit_instruction": label_polish_prompt},
            }
        )
    correction_prompts.append(label_polish_prompt)

    logger.info(
        "Applying %d correction pass(es): %d structural + 1 label polish",
        len(correction_prompts), num_structural_passes,
    )

    # -----------------------------------------------------------------------
    # Sequential Gemini refinement
    # -----------------------------------------------------------------------
    current_filename = filename
    current_data_url = image_data_url
    current_bytes: bytes = b''

    for i, correction_prompt in enumerate(correction_prompts):
        is_last = (i == len(correction_prompts) - 1)
        logger.info(
            "Correction pass %d/%d [%s]: %s...",
            i + 1, len(correction_prompts),
            "label-polish" if is_last else "structural",
            correction_prompt[:120],
        )
        pass_kind = "label-polish" if is_last else "structural"
        current_filename, current_bytes, current_data_url = edit_image(
            current_filename,
            correction_prompt,
            current_data_url,
            trace=trace,
            trace_step_id=f"gemini-correction-{i + 1}",
            trace_title=(
                f"Gemini edit — {pass_kind} ({i + 1}/{len(correction_prompts)})"
            ),
        )

    if trace is not None:
        trace.append(
            {
                "id": "result",
                "title": "Final result",
                "provider": "app",
                "model": "",
                "input": {},
                "output": {
                    "message": "Accuracy pipeline complete.",
                    "filename": current_filename,
                    "png_byte_length": len(current_bytes),
                    "flaws_addressed": total_flaws,
                    "gemini_passes": len(correction_prompts),
                },
            }
        )

    return (
        current_filename,
        current_bytes,
        current_data_url,
        total_flaws,
        len(correction_prompts),
        trace,
    )
