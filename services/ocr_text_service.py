"""
OCR text extraction and SVG text layer for canvas vectorization.

Uses Tesseract (via pytesseract) when available; all entry points no-op safely
if the binary or package is missing.
"""
import logging
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_pytesseract = None
_tesseract_configured = False

# Built-in anatomy/medical token vocabulary used to correct OCR misreads
# (e.g. "interosseı" -> "interossei", "radıus" -> "radius"). Caller-supplied
# labels / the source prompt are added on top of this at request time.
ANATOMY_VOCABULARY = frozenset({
    # bones
    "phalanx", "phalanges", "distal", "middle", "proximal", "metacarpal",
    "metacarpals", "carpal", "carpals", "radius", "ulna", "humerus", "scaphoid",
    "lunate", "triquetrum", "pisiform", "trapezium", "trapezoid", "capitate",
    "hamate", "sesamoid", "epiphysis", "diaphysis",
    # joints / connective
    "joint", "ligament", "ligaments", "collateral", "capsule", "cartilage",
    "tendon", "tendons", "flexor", "extensor", "retinaculum", "aponeurosis",
    "sheath", "pulley", "pulleys", "fascia",
    # muscles
    "muscle", "muscles", "thenar", "hypothenar", "lumbrical", "lumbricals",
    "interosseous", "interossei", "adductor", "abductor", "opponens", "flexor",
    "extensor", "brevis", "longus", "pollicis", "digiti", "minimi",
    # nerves
    "nerve", "nerves", "median", "ulnar", "radial", "digital", "palmar",
    "dorsal", "branch",
    # vessels
    "artery", "arteries", "arterial", "vein", "veins", "venous", "network",
    "arch", "superficial", "deep", "anastomosis", "capillary",
    # orientation / general
    "anterior", "posterior", "medial", "lateral", "superior", "inferior",
    "volar", "ventral", "head", "shaft", "base", "neck", "body", "process",
    "tubercle", "tuberosity", "fossa", "groove", "notch",
})

# Common Tesseract character confusions to normalize before matching.
_OCR_CHAR_FIXES = str.maketrans({
    "ı": "i",  # dotless i (very common with this font)
    "İ": "I",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "–": "-",
})

_WORD_RE = re.compile(r"[A-Za-z]+")


def _try_import_pytesseract():
    global _pytesseract, _tesseract_configured  # noqa: PLW0603
    if _pytesseract is not None:
        return _pytesseract
    try:
        import pytesseract as pt  # noqa: PLC0415
    except ImportError:
        logger.debug("pytesseract not installed; OCR text layer disabled")
        return None

    if not _tesseract_configured:
        cmd = (config.TESSERACT_CMD or "").strip()
        if cmd:
            pt.pytesseract.tesseract_cmd = cmd
        _tesseract_configured = True

    try:
        pt.get_tesseract_version()
    except Exception as exc:
        logger.info("Tesseract not available (%s); OCR text layer disabled", exc)
        return None

    _pytesseract = pt
    return pt


def _try_import_cv2():
    try:
        import cv2  # noqa: PLC0415
        return cv2
    except ImportError:
        return None


def is_ocr_available() -> bool:
    """True if pytesseract and the Tesseract binary are usable."""
    return _try_import_pytesseract() is not None


def extract_words(
    image: Image.Image,
    conf_threshold: Optional[int] = None,
    min_height: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run Tesseract on a trace-sized RGB image.

    Returns list of dicts: text, left, top, width, height (pixel coords).
    """
    pt = _try_import_pytesseract()
    if pt is None:
        return []

    conf_threshold = (
        conf_threshold
        if conf_threshold is not None
        else config.TRACE_OCR_MIN_CONFIDENCE
    )
    min_height = (
        min_height if min_height is not None else config.TRACE_OCR_MIN_HEIGHT
    )

    rgb = image.convert("RGB")
    tess_config = (config.TRACE_OCR_TESSERACT_CONFIG or "").strip()
    try:
        data = pt.image_to_data(
            rgb, output_type=pt.Output.DICT, config=tess_config
        )
    except Exception as exc:
        logger.warning("Tesseract OCR failed: %s", exc)
        return []

    words: List[Dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < 0 or conf < conf_threshold:
            continue

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        if width < 2 or height < min_height:
            continue

        words.append({
            "text": text,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        })

    logger.info("OCR extracted %d word(s)", len(words))
    return words


def build_vocabulary(extra_terms: Optional[Sequence[str]] = None) -> set:
    """
    Build the lowercase token vocabulary for OCR correction: the built-in
    anatomy terms plus any caller-supplied labels/prompt text (split into words).
    """
    vocab = set(ANATOMY_VOCABULARY)
    for term in (extra_terms or []):
        if not term:
            continue
        for token in _WORD_RE.findall(str(term).lower()):
            if len(token) >= 3:
                vocab.add(token)
    return vocab


def _normalize_token(text: str) -> str:
    return text.translate(_OCR_CHAR_FIXES)


def _match_case(original: str, replacement: str) -> str:
    """Apply the casing pattern of `original` to `replacement`."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _best_vocab_match(token: str, vocabulary: set, min_ratio: float) -> Optional[str]:
    """Return the closest vocabulary term to `token`, or None if none is close enough."""
    lower = token.lower()
    if lower in vocabulary:
        return lower
    best = None
    best_ratio = min_ratio
    for cand in vocabulary:
        # Cheap length prefilter before the (costlier) ratio computation.
        if abs(len(cand) - len(lower)) > 2:
            continue
        ratio = SequenceMatcher(None, lower, cand).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = cand
    return best


def correct_words_to_vocabulary(
    words: List[Dict[str, Any]],
    extra_terms: Optional[Sequence[str]] = None,
    min_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Snap each OCR word to the nearest known vocabulary term (in place).

    Fixes character-level misreads such as dotless-i ("radıus" -> "radius",
    "interosseı" -> "interossei") without touching words that have no close
    match. Returns the same list for convenience.
    """
    if not words or not config.TRACE_OCR_VOCAB_CORRECTION:
        return words

    min_ratio = (
        min_ratio if min_ratio is not None else config.TRACE_OCR_VOCAB_MIN_RATIO
    )
    vocabulary = build_vocabulary(extra_terms)
    corrected = 0

    for word in words:
        # Normalize common char confusions (e.g. dotless-ı -> i) FIRST, so the
        # ASCII word regex below can see whole words, then snap to vocabulary.
        normalized_text = _normalize_token(word.get("text") or "")

        def _sub(match):
            nonlocal corrected
            tok = match.group(0)
            cand = _best_vocab_match(tok, vocabulary, min_ratio)
            if cand and cand.lower() != tok.lower():
                corrected += 1
                return _match_case(tok, cand)
            return tok

        word["text"] = _WORD_RE.sub(_sub, normalized_text)

    if corrected:
        logger.info("OCR vocabulary correction fixed %d token(s)", corrected)
    return words


def sample_text_color(image: Image.Image, word: Dict[str, Any]) -> str:
    """
    Pick a fill color for SVG text from pixels inside the word bounding box.
    Uses a dark percentile of ink pixels so anti-aliased strokes stay legible.
    """
    rgb = np.array(image.convert("RGB"))
    h_img, w_img = rgb.shape[:2]
    left = max(0, word["left"])
    top = max(0, word["top"])
    right = min(w_img, left + word["width"])
    bottom = min(h_img, top + word["height"])
    if right <= left or bottom <= top:
        return "#1a1d24"

    crop = rgb[top:bottom, left:right].reshape(-1, 3)
    if crop.size == 0:
        return "#1a1d24"

    luminance = crop.astype(np.float32).mean(axis=1)
    bg_lum = float(np.median(luminance))
    dark_mask = luminance < bg_lum - 12
    if np.any(dark_mask):
        pixels = crop[dark_mask]
        ink_lum = luminance[dark_mask]
    else:
        pixels = crop
        ink_lum = luminance

    # 25th percentile of ink luminance -> darker, crisper label color
    lum_threshold = float(np.percentile(ink_lum, 25))
    darker_mask = ink_lum <= lum_threshold
    if np.any(darker_mask):
        pixels = pixels[darker_mask]

    color = np.median(pixels, axis=0).astype(np.uint8)
    # Clamp toward near-black so faint gray anti-aliasing does not wash out text
    max_channel = int(np.max(color))
    if max_channel > 80:
        darken = 80.0 / max(max_channel, 1)
        color = np.clip(color.astype(np.float32) * darken, 0, 80).astype(np.uint8)
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def mask_text_regions(
    image: Image.Image,
    words: List[Dict[str, Any]],
    dilate_px: Optional[int] = None,
) -> Image.Image:
    """
    Remove text regions before tracing so vtracer does not produce blob paths.
    Uses cv2.inpaint when available; otherwise fills with local median color.
    """
    if not words:
        return image

    dilate_px = dilate_px if dilate_px is not None else config.TRACE_OCR_MASK_DILATE
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    h_img, w_img = arr.shape[:2]

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for word in words:
        left = max(0, word["left"] - dilate_px)
        top = max(0, word["top"] - dilate_px)
        right = min(w_img, word["left"] + word["width"] + dilate_px)
        bottom = min(h_img, word["top"] + word["height"] + dilate_px)
        mask[top:bottom, left:right] = 255

    cv2 = _try_import_cv2()
    if cv2 is not None:
        try:
            bgr = arr[:, :, ::-1].copy()
            inpainted = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return Image.fromarray(inpainted[:, :, ::-1], mode="RGB")
        except Exception as exc:
            logger.warning("cv2.inpaint failed (%s); using median fill", exc)

    result = arr.copy()
    pad = 6
    for word in words:
        left = max(0, word["left"] - pad)
        top = max(0, word["top"] - pad)
        right = min(w_img, word["left"] + word["width"] + pad)
        bottom = min(h_img, word["top"] + word["height"] + pad)
        ring_left = max(0, left - pad)
        ring_top = max(0, top - pad)
        ring_right = min(w_img, right + pad)
        ring_bottom = min(h_img, bottom + pad)
        ring = result[ring_top:ring_bottom, ring_left:ring_right].copy()
        inner_mask = np.zeros(ring.shape[:2], dtype=bool)
        inner_mask[
            (top - ring_top):(bottom - ring_top),
            (left - ring_left):(right - ring_left),
        ] = True
        border = ring[~inner_mask] if np.any(~inner_mask) else ring.reshape(-1, 3)
        if border.size:
            fill = np.median(border.reshape(-1, 3), axis=0).astype(np.uint8)
            result[top:bottom, left:right] = fill

    return Image.fromarray(result, mode="RGB")


def build_text_svg_elements(
    words: List[Dict[str, Any]],
    colors: Optional[List[str]] = None,
    font_scale: Optional[float] = None,
) -> List[ET.Element]:
    """Build SVG <text> elements in trace image pixel coordinates."""
    if not words:
        return []

    font_scale = font_scale if font_scale is not None else config.TRACE_OCR_FONT_SCALE
    elements: List[ET.Element] = []

    for i, word in enumerate(words):
        fill = colors[i] if colors and i < len(colors) else "#1a1d24"
        height = max(word["height"], 8)
        width = max(word["width"], 8)
        text = word["text"]
        char_count = max(len(text), 1)

        # Fit font to both OCR box height and width (Arial ~0.55em avg char width)
        size_from_height = height * font_scale
        size_from_width = (2.0 * width) / char_count
        font_size = max(8, round(min(size_from_height, size_from_width)))
        baseline_y = word["top"] + height * 0.80

        elem = ET.Element(f"{{{SVG_NS}}}text")
        elem.set("x", str(word["left"]))
        elem.set("y", f"{baseline_y:.2f}")
        elem.set("font-size", str(font_size))
        elem.set("font-family", "Arial, Helvetica, sans-serif")
        elem.set("fill", fill)
        elem.set("textLength", str(width))
        elem.set("lengthAdjust", "spacingAndGlyphs")
        elem.text = text
        elements.append(elem)

    return elements


def inject_text_into_svg(
    svg_str: str,
    words: List[Dict[str, Any]],
    color_image: Image.Image,
    font_scale: Optional[float] = None,
) -> str:
    """Append OCR text elements to an existing SVG string (on top of paths)."""
    if not words:
        return svg_str

    colors = [sample_text_color(color_image, w) for w in words]
    text_elems = build_text_svg_elements(words, colors, font_scale=font_scale)
    if not text_elems:
        return svg_str

    try:
        root = ET.fromstring(svg_str)
    except ET.ParseError:
        logger.warning("Could not parse SVG for text injection")
        return svg_str

    for elem in text_elems:
        root.append(elem)

    return ET.tostring(root, encoding="unicode")
