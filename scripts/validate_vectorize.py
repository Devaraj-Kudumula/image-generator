"""Quick validation for vectorize pipeline changes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import re

from services import vectorize_service
import config

img_path = ROOT / "static/images/image_20260604_182614.png"
image_bytes = img_path.read_bytes()

settings = vectorize_service.get_vectorize_settings()
print("quantize_colors:", settings["trace_quantize_colors"])
print("min_path_area:", settings["trace_svg_min_path_area"])
print("layer_difference:", settings["vtracer"]["layer_difference"])
print("filter_speckle:", settings["vtracer"]["filter_speckle"])
print("ocr_mask_dilate:", config.TRACE_OCR_MASK_DILATE)

svg, meta = vectorize_service.vectorize_png_to_svg_with_meta(image_bytes)
print("trace:", meta["trace_width"], "x", meta["trace_height"])
print("path_count:", meta["path_count"])
print("svg_length:", meta["svg_length"])

text_elems = re.findall(r"<text[^>]+>[^<]+</text>", svg)
print("text_elements:", len(text_elems))
print("textLength_attrs:", len(re.findall(r"textLength=", svg)))
if text_elems:
    print("sample_text:", text_elems[0][:220])

fills = re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)
dark = [f for f in fills if int(f[1:3], 16) <= 100]
print("dark_text_fills:", len(dark), "sample:", dark[:5])

debug = vectorize_service.dump_svg_for_debug(svg, prefix="vector_validate")
print("debug_svg:", debug)
