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

from google.genai import types

from backend.image_utils import (
    image_bytes_to_data_url,
    decode_image_data_url,
    extract_png_bytes_from_gemini_response,
)
from app_state import state
import config
from prompts import (
    EDIT_IMAGE_USER_PREFIX,
    EDIT_VISUAL_CONTINUITY,
    STRUCTURAL_DETECTION_SYSTEM,
    STRUCTURAL_DETECTION_USER,
    STRUCTURAL_DETECTION_ORIGINAL_PROMPT_SUFFIX,
    LABEL_DETECTION_SYSTEM,
    LABEL_DETECTION_USER,
    LABEL_DETECTION_ORIGINAL_PROMPT_SUFFIX,
    STRUCTURAL_CORRECTION_SYSTEM,
    LABEL_POLISH_SYSTEM,
    INTENT_SUFFIX_TEMPLATE,
    REFINED_REGEN_VISION_SYSTEM,
    REFINED_REGEN_VISION_USER,
    REFINED_REGEN_VISION_ORIGINAL_PROMPT_SUFFIX,
    REFINED_REGEN_PROMPT_SYSTEM,
)
logger = logging.getLogger(__name__)


ALLOWED_ASPECT_RATIOS = ("1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9")
DEFAULT_ASPECT_RATIO = "16:9"


def normalize_aspect_ratio(value: Optional[str]) -> str:
    """Return a valid aspect ratio string, falling back to DEFAULT_ASPECT_RATIO."""
    if not value:
        return DEFAULT_ASPECT_RATIO
    candidate = str(value).strip()
    if candidate in ALLOWED_ASPECT_RATIOS:
        return candidate
    return DEFAULT_ASPECT_RATIO


def _extract_gemini_usage(response: Any) -> Dict[str, Any]:
    """Best-effort extraction of token usage from a Gemini GenerateContentResponse."""
    usage_md = getattr(response, "usage_metadata", None)
    if usage_md is None:
        return {}
    prompt_tokens = (
        getattr(usage_md, "prompt_token_count", None)
        or getattr(usage_md, "promptTokenCount", None)
    )
    completion_tokens = (
        getattr(usage_md, "candidates_token_count", None)
        or getattr(usage_md, "candidatesTokenCount", None)
    )
    total_tokens = (
        getattr(usage_md, "total_token_count", None)
        or getattr(usage_md, "totalTokenCount", None)
    )
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "provider": "gemini",
    }


def _extract_openai_usage(response: Any) -> Dict[str, Any]:
    """Best-effort extraction of token usage from an OpenAI ChatCompletion response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "provider": "openai",
    }


def generate_image(
    prompt: str,
    aspect_ratio: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, bytes, str, Dict[str, Any]]:
    """
    Generate an image using Gemini. Returns (filename, image_bytes, image_data_url, usage).
    `usage` is a dict with prompt_tokens / completion_tokens / total_tokens keys
    (from Gemini's usage_metadata). Empty dict if not available.
    Raises on missing client or API errors.
    """
    if not state.gemini_client:
        raise ValueError("Gemini client not initialized")

    ratio = normalize_aspect_ratio(aspect_ratio)
    gemini_model = model or "gemini-3-pro-image-preview"
    response = state.gemini_client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=ratio,
            ),
        ),
    )
    usage = _extract_gemini_usage(response)

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
    return (filename, image_bytes, image_data_url, usage)


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
    preserve_visual_identity: bool = False,
    aspect_ratio: Optional[str] = None,
) -> Tuple[str, bytes, str, Dict[str, Any]]:
    """
    Edit an existing image with Gemini.
    Returns (new_filename, edited_bytes, edited_data_url, usage).
    """
    if not state.gemini_client:
        raise ValueError("Gemini client not initialized")

    image = load_image_for_edit(filename, image_data_url)

    prompt = (
        EDIT_IMAGE_USER_PREFIX
        + f"Changes: {changes}"
        + (EDIT_VISUAL_CONTINUITY if preserve_visual_identity else "")
    )
    ratio = normalize_aspect_ratio(aspect_ratio)
    try:
        response = state.gemini_client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=ratio,
                ),
            ),
        )
    except Exception as api_error:
        raise ValueError(f"Error calling Gemini API: {str(api_error)}") from api_error
    usage = _extract_gemini_usage(response)

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
    return (new_filename, edited_bytes, edited_image_data_url, usage)


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

def _openai_chat_temperature_kwargs(model: str) -> Dict[str, Any]:
    """
    gpt-5.x rejects temperature=0 (and other non-default values); only the API
    default is allowed. Omit the parameter so OpenAI uses model default.
    """
    m = (model or "").strip().lower()
    if m.startswith("gpt-5"):
        return {}
    return {"temperature": 0}


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
) -> Tuple[str, Dict[str, Any]]:
    """Call OpenAI vision model and return (raw text response, usage dict)."""
    response = oa_client.chat.completions.create(
        model=model,
        **_openai_chat_temperature_kwargs(model),
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
    raw = response.choices[0].message.content
    text = (raw or "").strip()
    usage = _extract_openai_usage(response)
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
    return text, usage


# ---------------------------------------------------------------------------
# get_accurate_image  (illustration correctness first, annotation polish as final pass)
# ---------------------------------------------------------------------------

def get_accurate_image(
    filename: str,
    image_data_url: Optional[str] = None,
    original_prompt: Optional[str] = None,
    collect_trace: bool = False,
    aspect_ratio: Optional[str] = None,
) -> Tuple[str, bytes, str, int, int, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Two-stage accuracy pipeline:

    DETECTION  — Two OpenAI vision calls:
      Stage A: Medical illustration correctness without text fixes — anatomy, view vs. brief,
               topology, misleading placement for learners
      Stage B: Labels and annotations — names, arrow targets, consistency with prompt, legibility

    CORRECTION — Sequential Gemini passes in this order:
      Pass 1..N : Structure/view fixes  (OpenAI-generated correction prompts, batched 3 flaws each)
      Pass N+1  : Label polish          (OpenAI-generated, always the final pass — fixes annotation
                                         issues AND cleans up any text distortion from earlier passes)

    OpenAI generates every correction prompt from the detected flaws + original intent,
    so Gemini always receives a high-quality, targeted instruction rather than a raw flaw dump.

    Returns (final_filename, final_bytes, final_data_url, total_flaws_count, iterations,
             accuracy_trace or None). When collect_trace is True, the last element is a list of
             per-call input/output records for the UI; otherwise None.
    """
    MAX_FLAWS_PER_PROMPT = 3
    MAX_STRUCTURAL_ITERATIONS = 4  # structural batches (up to 12 structural flaws addressed)

    ratio = normalize_aspect_ratio(aspect_ratio)

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

    aggregated_usage: Dict[str, Dict[str, int]] = {
        "openai": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "gemini": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    def _accumulate(provider: str, usage: Dict[str, Any]) -> None:
        if not usage:
            return
        bucket = aggregated_usage[provider]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                bucket[key] += int(value)

    intent_snippet = (
        INTENT_SUFFIX_TEMPLATE.format(original_prompt=original_prompt.strip())
        if original_prompt and original_prompt.strip()
        else ""
    )

    # -----------------------------------------------------------------------
    # Stage A — OpenAI detects illustration correctness (structure, view, pedagogy)
    # -----------------------------------------------------------------------
    structural_detection_prompt = (
        STRUCTURAL_DETECTION_USER
        + (
            STRUCTURAL_DETECTION_ORIGINAL_PROMPT_SUFFIX.format(
                original_prompt=original_prompt.strip()
            )
            if original_prompt and original_prompt.strip() else ""
        )
    )

    structural_flaw_text, _usage_a = _detect_flaws_via_openai(
        oa_client,
        STRUCTURAL_DETECTION_SYSTEM,
        structural_detection_prompt,
        image_data_url,
        tag="structural-detection",
        trace=trace,
        trace_step_id="structural-detection",
        trace_title="Illustration correctness — structure & view (OpenAI vision)",
    )
    _accumulate("openai", _usage_a)

    # -----------------------------------------------------------------------
    # Stage B — OpenAI detects label & annotation flaws
    # -----------------------------------------------------------------------
    label_detection_prompt = (
        LABEL_DETECTION_USER
        + (
            LABEL_DETECTION_ORIGINAL_PROMPT_SUFFIX.format(
                original_prompt=original_prompt.strip()
            )
            if original_prompt and original_prompt.strip() else ""
        )
    )

    label_flaw_text, _usage_b = _detect_flaws_via_openai(
        oa_client,
        LABEL_DETECTION_SYSTEM,
        label_detection_prompt,
        image_data_url,
        tag="label-detection",
        trace=trace,
        trace_step_id="label-detection",
        trace_title="Labels & annotations vs. structures & brief (OpenAI vision)",
    )
    _accumulate("openai", _usage_b)

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
        "Detected %d illustration flaw(s) and %d annotation flaw(s)",
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
        return (fn, bs, du, fc, it, trace, aggregated_usage)

    # -----------------------------------------------------------------------
    # OpenAI generates illustration-correction prompts (batched, 3 flaws each)
    # -----------------------------------------------------------------------
    max_structural_flaws = MAX_STRUCTURAL_ITERATIONS * MAX_FLAWS_PER_PROMPT
    if len(structural_flaws) > max_structural_flaws:
        logger.info(
            "Capping illustration flaws from %d to %d",
            len(structural_flaws), max_structural_flaws,
        )
        structural_flaws = structural_flaws[:max_structural_flaws]

    num_structural_passes = min(
        math.ceil(len(structural_flaws) / MAX_FLAWS_PER_PROMPT),
        MAX_STRUCTURAL_ITERATIONS,
    )

    correction_prompts: List[str] = []

    for i in range(num_structural_passes):
        batch = structural_flaws[i * MAX_FLAWS_PER_PROMPT: (i + 1) * MAX_FLAWS_PER_PROMPT]
        flaw_list = "\n".join(f"- {f}" for f in batch)
        structural_user_msg = f"Illustration correctness issues to fix:\n{flaw_list}" + intent_snippet

        # Ask OpenAI to turn the raw flaw list into a precise Gemini edit instruction
        _corr_model = "gpt-5.4"
        openai_prompt_gen_response = oa_client.chat.completions.create(
            model=_corr_model,
            **_openai_chat_temperature_kwargs(_corr_model),
            messages=[
                {"role": "system", "content": STRUCTURAL_CORRECTION_SYSTEM},
                {"role": "user", "content": structural_user_msg},
            ],
            max_completion_tokens=500,
        )
        _gp = openai_prompt_gen_response.choices[0].message.content
        generated_prompt = (_gp or "").strip()
        _accumulate("openai", _extract_openai_usage(openai_prompt_gen_response))
        logger.info("Generated illustration correction prompt %d/%d:\n%s", i + 1, num_structural_passes, generated_prompt)
        if trace is not None:
            trace.append(
                {
                    "id": f"structural-prompt-gen-{i + 1}",
                    "title": (
                        f"OpenAI: illustration correction instruction "
                        f"({i + 1}/{num_structural_passes})"
                    ),
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "input": {
                        "system_prompt": STRUCTURAL_CORRECTION_SYSTEM,
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
        else "No specific annotation issues detected, but re-render all text cleanly."
    )

    label_polish_user = f"Label and annotation issues to fix:\n{label_flaw_summary}" + intent_snippet

    _polish_model = "gpt-5.4"
    label_polish_gen_response = oa_client.chat.completions.create(
        model=_polish_model,
        **_openai_chat_temperature_kwargs(_polish_model),
        messages=[
            {"role": "system", "content": LABEL_POLISH_SYSTEM},
            {"role": "user", "content": label_polish_user},
        ],
        max_completion_tokens=500,
    )
    _lp = label_polish_gen_response.choices[0].message.content
    label_polish_prompt = (_lp or "").strip()
    _accumulate("openai", _extract_openai_usage(label_polish_gen_response))
    logger.info("Generated label polish prompt:\n%s", label_polish_prompt)
    if trace is not None:
        trace.append(
            {
                "id": "label-polish-prompt-gen",
                "title": "OpenAI: label polish edit instruction",
                "provider": "openai",
                "model": "gpt-5.4",
                "input": {
                    "system_prompt": LABEL_POLISH_SYSTEM,
                    "user_prompt": label_polish_user,
                },
                "output": {"generated_edit_instruction": label_polish_prompt},
            }
        )
    correction_prompts.append(label_polish_prompt)

    logger.info(
        "Applying %d correction pass(es): %d illustration + 1 annotation polish",
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
        current_filename, current_bytes, current_data_url, _gem_usage = edit_image(
            current_filename,
            correction_prompt,
            current_data_url,
            trace=trace,
            trace_step_id=f"gemini-correction-{i + 1}",
            trace_title=(
                f"Gemini edit — {pass_kind} ({i + 1}/{len(correction_prompts)})"
            ),
            preserve_visual_identity=True,
            aspect_ratio=ratio,
        )
        _accumulate("gemini", _gem_usage)

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
        aggregated_usage,
    )


def refined_prompt_regenerate_image(
    filename: str,
    image_data_url: Optional[str] = None,
    original_prompt: Optional[str] = None,
    collect_trace: bool = False,
    aspect_ratio: Optional[str] = None,
) -> Tuple[str, bytes, str, str, str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Vision QA (OpenAI) → refined full prompt (OpenAI text) → new image (Gemini).

    Returns (final_filename, final_bytes, final_data_url, refined_prompt,
             vision_analysis_text, trace | None, aggregated_usage).
    """
    if not state.gemini_client:
        raise ValueError("Gemini client not initialized")
    if not state.openai_api_key:
        raise ValueError("OpenAI API key not configured")

    if not image_data_url or not image_data_url.strip().lower().startswith("data:"):
        image_pil = load_image_for_edit(filename, image_data_url or None)
        buf = BytesIO()
        image_pil.save(buf, format='PNG')
        image_data_url = image_bytes_to_data_url(buf.getvalue())

    oa_client = openai_lib.OpenAI(api_key=state.openai_api_key)
    trace: Optional[List[Dict[str, Any]]] = [] if collect_trace else None

    aggregated_usage: Dict[str, Dict[str, int]] = {
        "openai": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "gemini": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    def _accumulate(provider: str, usage: Dict[str, Any]) -> None:
        if not usage:
            return
        bucket = aggregated_usage[provider]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                bucket[key] += int(value)

    vision_user = REFINED_REGEN_VISION_USER
    if original_prompt and original_prompt.strip():
        vision_user += REFINED_REGEN_VISION_ORIGINAL_PROMPT_SUFFIX.format(
            original_prompt=original_prompt.strip()
        )

    vision_analysis, vision_usage = _detect_flaws_via_openai(
        oa_client,
        REFINED_REGEN_VISION_SYSTEM,
        vision_user,
        image_data_url,
        model=config.OPENAI_REFINED_REGEN_VISION_MODEL,
        tag="refined-regen-vision",
        trace=trace,
        trace_step_id="refined-regen-vision",
        trace_title="Vision QA — mistakes vs. prompt (refined regen)",
    )
    _accumulate("openai", vision_usage)

    orig_for_refine = (original_prompt or "").strip()
    if not orig_for_refine:
        orig_for_refine = (
            "(Not provided — infer subject, region, and teaching goal from the QA notes "
            "and the figure; the replacement prompt must stand alone.)"
        )

    ref_user = (
        "Original image generation prompt:\n"
        + orig_for_refine
        + "\n\nVision QA analysis of the current image:\n"
        + vision_analysis
    )
    _refine_model = config.OPENAI_REFINED_REGEN_TEXT_MODEL
    refine_response = oa_client.chat.completions.create(
        model=_refine_model,
        **_openai_chat_temperature_kwargs(_refine_model),
        messages=[
            {"role": "system", "content": REFINED_REGEN_PROMPT_SYSTEM},
            {"role": "user", "content": ref_user},
        ],
        max_completion_tokens=8192,
    )
    raw_refined = refine_response.choices[0].message.content
    refined_prompt = (raw_refined or "").strip()
    _accumulate("openai", _extract_openai_usage(refine_response))
    if trace is not None:
        trace.append(
            {
                "id": "refined-prompt-gen",
                "title": "OpenAI: refined image generation prompt",
                "provider": "openai",
                "model": config.OPENAI_REFINED_REGEN_TEXT_MODEL,
                "input": {
                    "system_prompt": REFINED_REGEN_PROMPT_SYSTEM,
                    "user_prompt": ref_user,
                },
                "output": {"refined_prompt": refined_prompt},
            }
        )

    if not refined_prompt:
        raise ValueError("Refined prompt generation returned empty text")

    gen_filename, gen_bytes, gen_data_url, gen_usage = generate_image(
        refined_prompt, aspect_ratio=aspect_ratio
    )
    _accumulate("gemini", gen_usage)

    if trace is not None:
        trace.append(
            {
                "id": "result",
                "title": "Final result",
                "provider": "app",
                "model": "",
                "input": {},
                "output": {
                    "message": "Refined prompt regeneration complete.",
                    "filename": gen_filename,
                    "refined_prompt": refined_prompt,
                    "png_byte_length": len(gen_bytes),
                },
            }
        )

    return (
        gen_filename,
        gen_bytes,
        gen_data_url,
        refined_prompt,
        vision_analysis,
        trace,
        aggregated_usage,
    )
