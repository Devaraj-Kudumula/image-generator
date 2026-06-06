"""
Diagram refinement via LLM-generated matplotlib code with visual feedback loop.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import openai as openai_lib

import config
from backend.image_utils import image_bytes_to_data_url
from prompts import (
    DIAGRAM_REFINE_SYSTEM,
    DIAGRAM_REFINE_INIT_USER,
    DIAGRAM_REFINE_ITER_USER,
    DIAGRAM_REFINE_EXEC_ERROR_USER,
    DIAGRAM_REFINE_INSTRUCTIONS_SUFFIX,
)
from services.code_exec import CodeExecError, run_matplotlib_code

logger = logging.getLogger(__name__)

_PYTHON_BLOCK_RE = re.compile(
    r'```(?:python)?\s*\n(.*?)```',
    re.DOTALL | re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r'STATUS\s*:\s*(DONE|CONTINUE)\b',
    re.IGNORECASE,
)


def _openai_chat_temperature_kwargs(model: str) -> Dict[str, Any]:
    m = (model or "").strip().lower()
    if m.startswith("gpt-5"):
        return {}
    return {"temperature": 0}


def _extract_openai_usage(response: Any) -> Dict[str, Any]:
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


def _summarize_image_for_trace(image_data_url: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not image_data_url:
        out["note"] = "No data URL."
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
    return out


def _parse_code_and_status(raw: str) -> Tuple[str, str]:
    """Extract python code block and STATUS from LLM response."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Model returned empty response")

    match = _PYTHON_BLOCK_RE.search(text)
    if match:
        code = match.group(1).strip()
    else:
        # Fallback: treat entire response as code if it looks like Python
        if "fig" in text and ("import matplotlib" in text or "plt." in text):
            code = text
        else:
            raise ValueError(
                "Model response did not contain a ```python code block"
            )

    status_match = _STATUS_RE.search(text)
    status = status_match.group(1).upper() if status_match else "CONTINUE"
    return code, status


def _call_vision_llm(
    oa_client: openai_lib.OpenAI,
    user_text: str,
    image_data_urls: List[str],
    model: str,
    trace: Optional[List[Dict[str, Any]]] = None,
    trace_step_id: str = "diagram-refine-vision",
    trace_title: str = "Diagram refine vision",
) -> Tuple[str, Dict[str, Any]]:
    """Call OpenAI vision with one or more images."""
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for url in image_data_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "high"},
        })

    response = oa_client.chat.completions.create(
        model=model,
        **_openai_chat_temperature_kwargs(model),
        messages=[
            {"role": "system", "content": DIAGRAM_REFINE_SYSTEM},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=8000,
    )
    raw = response.choices[0].message.content
    text = (raw or "").strip()
    usage = _extract_openai_usage(response)

    logger.info("--- [diagram-refine] INPUT ---\n%s", user_text[:500])
    logger.info("--- [diagram-refine] OUTPUT ---\n%s", text[:2000])

    if trace is not None:
        trace.append({
            "id": trace_step_id,
            "title": trace_title,
            "provider": "openai",
            "model": model,
            "input": {
                "user_prompt": user_text,
                "images": [
                    _summarize_image_for_trace(u) for u in image_data_urls
                ],
            },
            "output": {"text": text},
        })

    return text, usage


def refine_image_to_vector(
    image_bytes: bytes,
    max_iterations: Optional[int] = None,
    instructions: Optional[str] = None,
    collect_trace: bool = False,
) -> Dict[str, Any]:
    """
    Run the see -> write code -> execute -> render -> feedback loop.

    Returns dict with svg, png_data_url, iterations, code, refine_trace, usage.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured")

    max_iter = max_iterations or config.DIAGRAM_REFINE_MAX_ITERATIONS
    timeout_s = config.DIAGRAM_REFINE_EXEC_TIMEOUT
    model = config.DIAGRAM_REFINE_MODEL

    oa_client = openai_lib.OpenAI(api_key=config.OPENAI_API_KEY)
    source_data_url = image_bytes_to_data_url(image_bytes)

    refine_trace: Optional[List[Dict[str, Any]]] = [] if collect_trace else None
    total_usage: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "provider": "openai",
    }

    def _accumulate_usage(usage: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = usage.get(key)
            if val is not None:
                total_usage[key] = (total_usage.get(key) or 0) + val

    # Build initial user prompt
    init_user = DIAGRAM_REFINE_INIT_USER
    if instructions and instructions.strip():
        init_user += DIAGRAM_REFINE_INSTRUCTIONS_SUFFIX.format(
            instructions=instructions.strip()
        )

    current_code: Optional[str] = None
    current_svg: Optional[str] = None
    current_png_bytes: Optional[bytes] = None
    current_render_url: Optional[str] = None
    iterations = 0
    last_error: Optional[str] = None

    for turn in range(max_iter):
        # --- LLM turn: generate or refine code ---
        if turn == 0:
            user_prompt = init_user
            images = [source_data_url]
            step_title = f"Turn {turn}: initial codegen"
        elif last_error:
            user_prompt = DIAGRAM_REFINE_EXEC_ERROR_USER.format(error=last_error)
            images = [source_data_url]
            if current_render_url:
                images.append(current_render_url)
            step_title = f"Turn {turn}: fix execution error"
            last_error = None
        else:
            user_prompt = DIAGRAM_REFINE_ITER_USER
            images = [source_data_url, current_render_url or source_data_url]
            step_title = f"Turn {turn}: visual refinement"

        raw_response, usage = _call_vision_llm(
            oa_client,
            user_prompt,
            images,
            model,
            trace=refine_trace,
            trace_step_id=f"diagram-refine-turn-{turn}",
            trace_title=step_title,
        )
        _accumulate_usage(usage)
        iterations = turn + 1

        try:
            code, status = _parse_code_and_status(raw_response)
        except ValueError as parse_err:
            last_error = str(parse_err)
            logger.warning("Parse error on turn %d: %s", turn, parse_err)
            if turn >= max_iter - 1:
                raise ValueError(
                    f"Failed to parse model output after {iterations} iteration(s): "
                    f"{parse_err}"
                ) from parse_err
            continue

        current_code = code

        # --- Execute code ---
        try:
            svg_string, png_bytes = run_matplotlib_code(code, timeout_s=timeout_s)
            current_svg = svg_string
            current_png_bytes = png_bytes
            current_render_url = image_bytes_to_data_url(png_bytes)
        except CodeExecError as exec_err:
            last_error = str(exec_err)
            logger.warning("Execution error on turn %d: %s", turn, exec_err)
            if refine_trace is not None:
                refine_trace.append({
                    "id": f"diagram-refine-exec-{turn}",
                    "title": f"Turn {turn}: execution failed",
                    "provider": "local",
                    "model": "matplotlib",
                    "input": {"code_preview": code[:500]},
                    "output": {"error": last_error},
                })
            if turn >= max_iter - 1:
                raise ValueError(
                    f"Code execution failed after {iterations} iteration(s): "
                    f"{exec_err}"
                ) from exec_err
            continue

        if refine_trace is not None:
            refine_trace.append({
                "id": f"diagram-refine-render-{turn}",
                "title": f"Turn {turn}: render",
                "provider": "local",
                "model": "matplotlib",
                "input": {"code_preview": code[:500]},
                "output": {
                    "svg_bytes": len(current_svg),
                    "png_bytes": len(current_png_bytes),
                    "status": status,
                },
            })

        if status == "DONE":
            logger.info(
                "Diagram refine finished: turn=%d status=%s",
                turn,
                status,
            )
            break

    if not current_svg or not current_png_bytes or not current_code:
        raise ValueError("Diagram refinement produced no output")

    return {
        "svg": current_svg,
        "png_data_url": image_bytes_to_data_url(current_png_bytes),
        "iterations": iterations,
        "code": current_code,
        "refine_trace": refine_trace,
        "usage": {"openai": total_usage},
    }
