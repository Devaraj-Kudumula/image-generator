"""
PNG to SVG vectorization for canvas editing.

Pipeline (vtracer backend): super-resolution -> edge-preserving smooth ->
color quantization -> vtracer -> SVG post-process (seam fill, scour).

Paid backends: set TRACE_BACKEND=vectorizer_ai|recraft and API keys.
"""
import logging
import re
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageFilter

import config
from services import ocr_text_service
from services.vectorize_backends.paid_backends import (
    SUPPORTED as PAID_BACKENDS,
    vectorize_via_paid_backend,
)

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

_SUPERRES_INSTANCE = None
_SUPERRES_LOADED_MODEL = None

_SUPERRES_MODEL_FILES = {
    'FSRCNN_x2': 'FSRCNN_x2.pb',
    'FSRCNN_x3': 'FSRCNN_x3.pb',
    'EDSR_x4': 'EDSR_x4.pb',
}

_SUPERRES_DOWNLOAD_URLS = {
    'FSRCNN_x2': (
        'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb'
    ),
    'FSRCNN_x3': (
        'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb'
    ),
    'EDSR_x4': (
        'https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb'
    ),
}


def _pil_to_bgr_array(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert('RGB'))
    return rgb[:, :, ::-1].copy()


def _bgr_array_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8), mode='RGB')


def _try_import_cv2():
    try:
        import cv2  # noqa: PLC0415
        return cv2
    except ImportError:
        return None


def _superres_model_path(model_name: str) -> Optional[Path]:
    filename = _SUPERRES_MODEL_FILES.get(model_name)
    if not filename:
        logger.warning("Unknown super-resolution model: %s", model_name)
        return None
    path = config.TRACE_SUPERRES_MODELS_DIR / filename
    if path.is_file():
        return path
    if config.TRACE_SUPERRES_AUTO_DOWNLOAD:
        url = _SUPERRES_DOWNLOAD_URLS.get(model_name)
        if url:
            try:
                config.TRACE_SUPERRES_MODELS_DIR.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading super-resolution model %s", filename)
                urllib.request.urlretrieve(url, path)  # noqa: S310
                if path.is_file():
                    return path
            except OSError as exc:
                logger.warning("Could not download super-resolution model: %s", exc)
    return None


def _get_superres_upscaler():
    global _SUPERRES_INSTANCE, _SUPERRES_LOADED_MODEL  # noqa: PLW0603

    model_name = config.TRACE_SUPERRES_MODEL
    if _SUPERRES_INSTANCE is not None and _SUPERRES_LOADED_MODEL == model_name:
        return _SUPERRES_INSTANCE

    cv2 = _try_import_cv2()
    if cv2 is None:
        return None

    try:
        from cv2 import dnn_superres  # noqa: PLC0415
    except ImportError:
        logger.debug("cv2.dnn_superres not available (opencv-contrib required)")
        return None

    model_path = _superres_model_path(model_name)
    if model_path is None:
        logger.info(
            "Super-resolution model not found at %s; using LANCZOS upscale",
            config.TRACE_SUPERRES_MODELS_DIR,
        )
        return None

    scale_map = {'FSRCNN_x2': 2, 'FSRCNN_x3': 3, 'EDSR_x4': 4}
    scale = scale_map.get(model_name, 2)
    algo = model_name.rsplit('_', 1)[0]

    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel(algo, scale)
        _SUPERRES_INSTANCE = sr
        _SUPERRES_LOADED_MODEL = model_name
        logger.info("Loaded super-resolution model %s (scale %d)", model_name, scale)
        return sr
    except Exception as exc:
        logger.warning("Failed to load super-resolution model: %s", exc)
        return None


def _superres_upscale(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Upscale with dnn_superres when possible; otherwise LANCZOS."""
    sr = _get_superres_upscaler()
    if sr is None:
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    cv2 = _try_import_cv2()
    if cv2 is None:
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    scale_map = {'FSRCNN_x2': 2, 'FSRCNN_x3': 3, 'EDSR_x4': 4}
    scale = scale_map.get(config.TRACE_SUPERRES_MODEL, 2)

    bgr = _pil_to_bgr_array(image)
    try:
        upscaled = sr.upsample(bgr)
        result = _bgr_array_to_pil(upscaled)
        if result.size != (target_w, target_h):
            result = result.resize((target_w, target_h), Image.Resampling.LANCZOS)
        logger.info(
            "Super-res upscaled %dx%d -> %dx%d (model %s)",
            image.size[0],
            image.size[1],
            result.size[0],
            result.size[1],
            config.TRACE_SUPERRES_MODEL,
        )
        return result
    except Exception as exc:
        logger.warning("Super-resolution failed (%s); using LANCZOS", exc)
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def _resize_for_tracing(
    image: Image.Image,
    max_dim: int,
    target_dim: int,
    upscale_enabled: bool,
) -> Tuple[Image.Image, int, int]:
    """
    Fit image into trace bounds: downscale if above max_dim, optionally upscale
    small images to target_dim (super-resolution when enabled).
    """
    width, height = image.size
    longest = max(width, height)

    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.info("Downscaled image for tracing to %dx%d", new_w, new_h)
        return image, new_w, new_h

    if upscale_enabled and target_dim > 0 and longest < target_dim:
        scale = target_dim / float(longest)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        use_superres = (
            config.TRACE_SUPERRES_ENABLED
            and _try_import_cv2() is not None
        )
        if use_superres:
            image = _superres_upscale(image, new_w, new_h)
        else:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.info("Upscaled image for tracing to %dx%d", new_w, new_h)
        return image, new_w, new_h

    return image, width, height


def _flatten_alpha(image: Image.Image, threshold: int) -> Image.Image:
    """Composite RGBA onto white with hard alpha cutout to avoid fringe paths."""
    if image.mode != "RGBA":
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    alpha = image.split()[3]
    alpha = alpha.point(lambda p: 255 if p >= threshold else 0)
    image = image.copy()
    image.putalpha(alpha)
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=alpha)
    return background


def _apply_edge_preserving_smooth(image: Image.Image) -> Image.Image:
    method = (config.TRACE_SMOOTH_METHOD or 'none').lower()
    if method in ('none', 'off', 'false'):
        return image

    cv2 = _try_import_cv2()
    if cv2 is None:
        return image

    bgr = _pil_to_bgr_array(image)
    try:
        if method == 'edge_preserving':
            smoothed = cv2.edgePreservingFilter(
                bgr,
                flags=1,
                sigma_s=config.TRACE_EDGE_PRESERVE_SIGMA_S,
                sigma_r=config.TRACE_EDGE_PRESERVE_SIGMA_R,
            )
        else:
            d = config.TRACE_BILATERAL_D
            if d % 2 == 0:
                d += 1
            smoothed = cv2.bilateralFilter(
                bgr,
                d,
                config.TRACE_BILATERAL_SIGMA_COLOR,
                config.TRACE_BILATERAL_SIGMA_SPACE,
            )
        return _bgr_array_to_pil(smoothed)
    except Exception as exc:
        logger.warning("Edge-preserving smooth failed (%s); skipping", exc)
        return image


def _quantize_colors(image: Image.Image, n_colors: int) -> Image.Image:
    """Reduce colors with k-means to flatten gradients before tracing."""
    if n_colors < 2:
        return image

    cv2 = _try_import_cv2()
    if cv2 is None:
        return image

    rgb = np.array(image.convert('RGB'))
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    max_sample = max(1000, config.TRACE_QUANTIZE_SAMPLE_MAX)
    if len(pixels) > max_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), size=max_sample, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    try:
        _compactness, labels_sample, centers = cv2.kmeans(
            sample,
            n_colors,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
    except Exception as exc:
        logger.warning("Color quantization failed (%s); skipping", exc)
        return image

    centers = np.uint8(np.round(centers))
    full_labels = np.argmin(
        np.linalg.norm(pixels[:, None, :] - centers[None, :, :].astype(np.float32), axis=2),
        axis=1,
    )
    quantized = centers[full_labels].reshape(h, w, 3)
    return Image.fromarray(quantized, mode='RGB')


def _flatten_only(image: Image.Image) -> Image.Image:
    """Alpha flatten and RGB conversion only (before resize / OCR)."""
    if config.TRACE_FLATTEN_ALPHA:
        image = _flatten_alpha(image, config.TRACE_ALPHA_THRESHOLD)
    elif image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    elif image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


def _smooth_quantize(image: Image.Image) -> Image.Image:
    """
    Edge-preserving smooth, optional denoise/sharpen, color quantization.
    Run after OCR text masking so labels are not traced as blobs.
    """
    if config.TRACE_DENOISE:
        image = image.filter(ImageFilter.MedianFilter(size=3))

    image = _apply_edge_preserving_smooth(image)

    if config.TRACE_SHARPEN:
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=config.TRACE_SHARPEN_RADIUS,
                percent=config.TRACE_SHARPEN_PERCENT,
                threshold=config.TRACE_SHARPEN_THRESHOLD,
            )
        )

    if config.TRACE_QUANTIZE_ENABLED and config.TRACE_QUANTIZE_COLORS >= 2:
        image = _quantize_colors(image, config.TRACE_QUANTIZE_COLORS)

    return image


def _preprocess_for_tracing(image: Image.Image) -> Image.Image:
    """Legacy full preprocess (flatten + smooth + quantize)."""
    return _smooth_quantize(_flatten_only(image))


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _prepare_trace_image(
    image_bytes: bytes,
    vocab_terms: Optional[List[str]] = None,
) -> Tuple[bytes, int, int, List[Dict[str, Any]], Optional[Image.Image]]:
    """
    Load, resize to trace dimensions, OCR + mask text, smooth/quantize, trace.

    Returns PNG bytes, width, height, OCR words, and clean resized image for colors.
    """
    image = Image.open(BytesIO(image_bytes))
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    image = _flatten_only(image)

    target_dim = config.TRACE_TARGET_DIMENSION if config.TRACE_UPSCALE_ENABLED else 0
    image, width, height = _resize_for_tracing(
        image,
        max_dim=config.TRACE_MAX_DIMENSION,
        target_dim=target_dim,
        upscale_enabled=config.TRACE_UPSCALE_ENABLED,
    )

    base_resized = image.copy()
    ocr_words: List[Dict[str, Any]] = []

    if config.TRACE_OCR_ENABLED and ocr_text_service.is_ocr_available():
        ocr_words = ocr_text_service.extract_words(base_resized)
        if ocr_words:
            ocr_text_service.correct_words_to_vocabulary(ocr_words, vocab_terms)
            image = ocr_text_service.mask_text_regions(image, ocr_words)
            logger.info("Masked %d OCR word region(s) before tracing", len(ocr_words))

    image = _smooth_quantize(image)

    return _image_to_png_bytes(image), width, height, ocr_words, base_resized


def _inject_text_layer(
    svg_str: str,
    words: List[Dict[str, Any]],
    color_image: Optional[Image.Image],
) -> str:
    if not words or color_image is None:
        return svg_str
    if not config.TRACE_OCR_ENABLED:
        return svg_str
    return ocr_text_service.inject_text_into_svg(svg_str, words, color_image)


def _vtracer_kwargs() -> Dict[str, Any]:
    return {
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": config.VTRACER_FILTER_SPECKLE,
        "color_precision": config.VTRACER_COLOR_PRECISION,
        "layer_difference": config.VTRACER_LAYER_DIFFERENCE,
        "corner_threshold": config.VTRACER_CORNER_THRESHOLD,
        "length_threshold": config.VTRACER_LENGTH_THRESHOLD,
        "max_iterations": config.VTRACER_MAX_ITERATIONS,
        "splice_threshold": config.VTRACER_SPLICE_THRESHOLD,
        "path_precision": config.VTRACER_PATH_PRECISION,
    }


def _parse_svg_length(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    match = re.match(r"^([\d.]+)", value)
    if match:
        return float(match.group(1))
    return None


def _parse_path_points(d_attr: str) -> List[Tuple[float, float]]:
    """Rough bbox from path d (M/L/C commands) for tiny-path filtering."""
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d_attr or "")
    if len(nums) < 4:
        return []
    floats = [float(n) for n in nums]
    xs = floats[0::2]
    ys = floats[1::2]
    if not xs or not ys:
        return []
    return list(zip(xs, ys))


def _path_bbox_area(d_attr: str) -> float:
    pts = _parse_path_points(d_attr)
    if len(pts) < 2:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def _local_name(tag: str) -> str:
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def _seam_fill_svg(root: ET.Element) -> None:
    """Hairline stroke matching fill to close gaps between stacked shapes."""
    stroke_w = str(config.TRACE_SVG_SEAM_STROKE_WIDTH)
    for elem in root.iter():
        if _local_name(elem.tag) != 'path':
            continue
        fill = elem.get('fill')
        if not fill or fill.lower() in ('none', 'transparent'):
            continue
        elem.set('stroke', fill)
        elem.set('stroke-width', stroke_w)
        elem.set('stroke-linejoin', 'round')
        if not elem.get('stroke-linecap'):
            elem.set('stroke-linecap', 'round')


def _remove_tiny_paths(root: ET.Element, min_area: float) -> int:
    removed = 0
    for parent in list(root.iter()):
        children = list(parent)
        for child in children:
            if _local_name(child.tag) != 'path':
                continue
            d_attr = child.get('d') or ''
            if _path_bbox_area(d_attr) < min_area:
                parent.remove(child)
                removed += 1
    return removed


def _normalize_svg_root(svg_str: str, width: int, height: int) -> str:
    """
    Ensure root <svg> has consistent width, height, and viewBox for Fabric.js.
    """
    try:
        root = ET.fromstring(svg_str)
    except ET.ParseError:
        logger.warning("Could not parse SVG for normalization; returning raw output")
        return svg_str

    tag = root.tag
    if not (tag.endswith("svg") or tag == f"{{{SVG_NS}}}svg"):
        return svg_str

    vb = root.get("viewBox")
    vb_w = vb_h = None
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            try:
                vb_w = float(parts[2])
                vb_h = float(parts[3])
            except ValueError:
                pass

    root_w = _parse_svg_length(root.get("width"))
    root_h = _parse_svg_length(root.get("height"))

    out_w = int(round(vb_w or root_w or width))
    out_h = int(round(vb_h or root_h or height))

    root.set("width", str(out_w))
    root.set("height", str(out_h))
    if not vb:
        root.set("viewBox", f"0 0 {out_w} {out_h}")

    return ET.tostring(root, encoding="unicode")


def _enhance_svg_structure(svg_str: str) -> str:
    try:
        root = ET.fromstring(svg_str)
    except ET.ParseError:
        return svg_str

    if config.TRACE_SVG_SEAM_FILL:
        _seam_fill_svg(root)

    if config.TRACE_SVG_MIN_PATH_AREA > 0:
        n = _remove_tiny_paths(root, config.TRACE_SVG_MIN_PATH_AREA)
        if n:
            logger.debug("Removed %d tiny SVG paths", n)

    return ET.tostring(root, encoding="unicode")


def _scour_svg(svg_str: str) -> str:
    """Lossless SVG cleanup: strip metadata, preserve coordinate precision."""
    if not config.TRACE_SVG_SCOUR:
        return svg_str
    try:
        import scour.scour as scour_lib
    except ImportError:
        logger.debug("scour not installed; skipping SVG cleanup")
        return svg_str

    precision = config.VTRACER_PATH_PRECISION
    options = scour_lib.sanitizeOptions(
        scour_lib.parse_args(
            [
                "scour",
                f"--set-precision={precision}",
                "--strip-xml-space",
                "--disable-group-collapsing",
                "--disable-simplify-colors",
                "-q",
            ]
        )
    )
    try:
        cleaned = scour_lib.scourString(svg_str, options)
        return cleaned if cleaned and cleaned.strip() else svg_str
    except Exception as exc:
        logger.warning("scour failed (%s); using unoptimized SVG", exc)
        return svg_str


def _postprocess_svg(svg_str: str, width: int, height: int) -> str:
    normalized = _normalize_svg_root(svg_str, width, height)
    enhanced = _enhance_svg_structure(normalized)
    return _scour_svg(enhanced)


def count_svg_paths(svg_str: str) -> int:
    """Count <path> elements (for A/B harness metrics)."""
    try:
        root = ET.fromstring(svg_str)
    except ET.ParseError:
        return 0
    return sum(1 for elem in root.iter() if _local_name(elem.tag) == 'path')


def get_vectorize_settings() -> Dict[str, Any]:
    """Expose active trace settings (for debug responses and scripts)."""
    return {
        "trace_backend": config.TRACE_BACKEND,
        "trace_max_dimension": config.TRACE_MAX_DIMENSION,
        "trace_target_dimension": config.TRACE_TARGET_DIMENSION,
        "trace_upscale_enabled": config.TRACE_UPSCALE_ENABLED,
        "trace_superres_enabled": config.TRACE_SUPERRES_ENABLED,
        "trace_superres_model": config.TRACE_SUPERRES_MODEL,
        "trace_flatten_alpha": config.TRACE_FLATTEN_ALPHA,
        "trace_denoise": config.TRACE_DENOISE,
        "trace_smooth_method": config.TRACE_SMOOTH_METHOD,
        "trace_sharpen": config.TRACE_SHARPEN,
        "trace_quantize_enabled": config.TRACE_QUANTIZE_ENABLED,
        "trace_quantize_colors": config.TRACE_QUANTIZE_COLORS,
        "trace_svg_scour": config.TRACE_SVG_SCOUR,
        "trace_svg_seam_fill": config.TRACE_SVG_SEAM_FILL,
        "trace_svg_min_path_area": config.TRACE_SVG_MIN_PATH_AREA,
        "trace_ocr_enabled": config.TRACE_OCR_ENABLED,
        "trace_ocr_available": ocr_text_service.is_ocr_available(),
        "vtracer": _vtracer_kwargs(),
        "is_serverless": config.IS_SERVERLESS,
    }


def _vectorize_vtracer(
    image_bytes: bytes, vocab_terms: Optional[List[str]] = None
) -> Tuple[str, int, int]:
    """vtracer pipeline: prepare image, trace, post-process."""
    try:
        import vtracer
    except ImportError as exc:
        raise ValueError(
            "vtracer is not installed. Add vtracer to requirements.txt and reinstall."
        ) from exc

    png_bytes, width, height, ocr_words, base_resized = _prepare_trace_image(
        image_bytes, vocab_terms
    )

    try:
        svg = vtracer.convert_raw_image_to_svg(
            png_bytes,
            img_format="png",
            **_vtracer_kwargs(),
        )
    except Exception as exc:
        logger.exception("vtracer failed")
        raise ValueError(f"Vectorization failed: {exc}") from exc

    if not svg or not str(svg).strip():
        raise ValueError("Vectorization produced empty SVG")

    svg_str = str(svg).strip()
    if not svg_str.lstrip().startswith("<"):
        raise ValueError("Vectorization produced invalid SVG")

    svg_str = _postprocess_svg(svg_str, width, height)
    svg_str = _inject_text_layer(svg_str, ocr_words, base_resized)
    if ocr_words:
        logger.info("Injected %d OCR text element(s) into SVG", len(ocr_words))
    return svg_str, width, height


def _vectorize_core(
    image_bytes: bytes, vocab_terms: Optional[List[str]] = None
) -> Tuple[str, int, int]:
    """
    Run vectorization via configured backend.
    Returns (svg_str, trace_width, trace_height).
    """
    if not image_bytes:
        raise ValueError("Empty image bytes")

    backend = (config.TRACE_BACKEND or 'vtracer').strip().lower()

    if backend in PAID_BACKENDS:
        svg_str, paid_meta = vectorize_via_paid_backend(image_bytes, backend)
        _png_bytes, width, height, ocr_words, base_resized = _prepare_trace_image(
            image_bytes, vocab_terms
        )
        svg_str = _postprocess_svg(svg_str, width, height)
        svg_str = _inject_text_layer(svg_str, ocr_words, base_resized)
        logger.info(
            "Vectorized via %s (%dx%d) -> SVG length %d meta=%s",
            backend,
            width,
            height,
            len(svg_str),
            paid_meta,
        )
        return svg_str, width, height

    if backend != 'vtracer':
        raise ValueError(
            f"Unknown TRACE_BACKEND '{config.TRACE_BACKEND}'. "
            f"Use vtracer, {', '.join(sorted(PAID_BACKENDS))}."
        )

    svg_str, width, height = _vectorize_vtracer(image_bytes, vocab_terms)
    logger.info(
        "Vectorized image (%dx%d trace) -> SVG length %d paths=%d",
        width,
        height,
        len(svg_str),
        count_svg_paths(svg_str),
    )
    return svg_str, width, height


def vectorize_png_to_svg(
    image_bytes: bytes, vocab_terms: Optional[List[str]] = None
) -> str:
    """
    Convert PNG bytes to an SVG string.

    `vocab_terms` (optional): known labels / source-prompt text used to correct
    OCR misreads in the text layer.

    Raises ValueError on empty input or tracing failure.
    """
    svg_str, _, _ = _vectorize_core(image_bytes, vocab_terms)
    return svg_str


def vectorize_png_to_svg_with_meta(
    image_bytes: bytes, vocab_terms: Optional[List[str]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Like vectorize_png_to_svg but returns trace dimensions and settings metadata.
    """
    svg_str, width, height = _vectorize_core(image_bytes, vocab_terms)
    meta = {
        "trace_width": width,
        "trace_height": height,
        "svg_length": len(svg_str),
        "path_count": count_svg_paths(svg_str),
        "settings": get_vectorize_settings(),
    }
    return svg_str, meta


def dump_svg_for_debug(svg_str: str, prefix: str = "vector_debug") -> Optional[str]:
    """
    Write SVG to VECTORIZE_DEBUG_DIR when not serverless. Returns filename or None.
    """
    if config.IS_SERVERLESS:
        return None
    try:
        config.VECTORIZE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.svg"
        path = config.VECTORIZE_DEBUG_DIR / filename
        path.write_text(svg_str, encoding="utf-8")
        logger.info("Debug SVG written to %s", path.resolve())
        return filename
    except OSError as exc:
        logger.warning("Could not write debug SVG: %s", exc)
        return None
