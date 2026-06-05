#!/usr/bin/env python3
"""
Local A/B helper: vectorize PNG(s) with multiple presets and compare metrics.

Usage:
  python scripts/compare_vectorize.py path/to/image.png
  python scripts/compare_vectorize.py path/to/folder --presets all
  python scripts/compare_vectorize.py image.png -o out_dir --rasterize
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from services.vectorize_service import (  # noqa: E402
    count_svg_paths,
    get_vectorize_settings,
    vectorize_png_to_svg_with_meta,
)

PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "high_fidelity": {
        "TRACE_QUANTIZE_COLORS": "24",
        "TRACE_TARGET_DIMENSION": "2048",
        "VTRACER_LAYER_DIFFERENCE": "16",
        "VTRACER_FILTER_SPECKLE": "4",
    },
    "flat_diagram": {
        "TRACE_QUANTIZE_COLORS": "16",
        "TRACE_SMOOTH_METHOD": "bilateral",
        "VTRACER_LAYER_DIFFERENCE": "24",
        "VTRACER_FILTER_SPECKLE": "8",
        "VTRACER_COLOR_PRECISION": "6",
    },
    "shaded_render": {
        "TRACE_QUANTIZE_COLORS": "28",
        "TRACE_SMOOTH_METHOD": "edge_preserving",
        "VTRACER_LAYER_DIFFERENCE": "18",
        "VTRACER_FILTER_SPECKLE": "5",
    },
    "legacy": {
        "TRACE_DENOISE": "true",
        "TRACE_SHARPEN": "true",
        "TRACE_QUANTIZE_ENABLED": "false",
        "TRACE_SUPERRES_ENABLED": "false",
        "VTRACER_LAYER_DIFFERENCE": "8",
        "VTRACER_FILTER_SPECKLE": "2",
    },
}


def _apply_preset(name: str) -> Dict[str, str]:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Choose from: {', '.join(PRESETS)}")
    return dict(PRESETS[name])


def _apply_env_overrides(overrides: Dict[str, str]) -> None:
    for key, value in overrides.items():
        os.environ[key] = str(value)
    _reload_config()


def _reload_config() -> None:
    """Re-read vectorization-related config from os.environ."""
    import importlib

    importlib.reload(config)


def _rasterize_svg(svg_path: Path, png_path: Path) -> bool:
    try:
        import cairosvg
    except ImportError:
        return False
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
        return True
    except Exception as exc:
        print(f"  rasterize failed: {exc}", file=sys.stderr)
        return False


def _collect_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    exts = {'.png', '.jpg', '.jpeg', '.webp'}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def _run_one(
    image_path: Path,
    preset_name: str,
    out_dir: Path,
    rasterize: bool,
) -> Dict[str, Any]:
    overrides = _apply_preset(preset_name)
    _apply_env_overrides(overrides)

    image_bytes = image_path.read_bytes()
    stem = f"{image_path.stem}_{preset_name}"
    svg_path = out_dir / f"{stem}.svg"

    svg_str, meta = vectorize_png_to_svg_with_meta(image_bytes)
    svg_path.write_text(svg_str, encoding="utf-8")

    row = {
        "image": str(image_path.name),
        "preset": preset_name,
        "svg_file": str(svg_path.name),
        "trace_width": meta["trace_width"],
        "trace_height": meta["trace_height"],
        "svg_length": meta["svg_length"],
        "path_count": meta.get("path_count", count_svg_paths(svg_str)),
        "overrides": overrides,
    }

    if rasterize:
        png_path = out_dir / f"{stem}_preview.png"
        row["preview_png"] = png_path.name if _rasterize_svg(svg_path, png_path) else None

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Vectorize PNG(s) for quality comparison")
    parser.add_argument("image", type=Path, help="Input PNG/JPEG or folder of images")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <image_stem>_vectorize_ab)",
    )
    parser.add_argument(
        "--presets",
        default="default,high_fidelity,flat_diagram",
        help="Comma-separated preset names or 'all'",
    )
    parser.add_argument(
        "--rasterize",
        action="store_true",
        help="Also write PNG previews via cairosvg (optional)",
    )
    parser.add_argument(
        "--settings-only",
        action="store_true",
        help="Print active settings and exit",
    )
    args = parser.parse_args()

    if args.settings_only:
        print(json.dumps(get_vectorize_settings(), indent=2))
        return 0

    if not args.image.exists():
        print(f"Error: not found: {args.image}", file=sys.stderr)
        return 1

    images = _collect_images(args.image)
    if not images:
        print("Error: no images found", file=sys.stderr)
        return 1

    if args.presets.strip().lower() == "all":
        preset_names = list(PPRESETS.keys())
    else:
        preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]

    for name in preset_names:
        if name not in PRESETS:
            print(f"Error: unknown preset '{name}'", file=sys.stderr)
            return 1

    if args.output:
        out_dir = args.output
    elif args.image.is_file():
        out_dir = args.image.parent / f"{args.image.stem}_vectorize_ab"
    else:
        out_dir = args.image / "vectorize_ab"

    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for image_path in images:
        for preset_name in preset_names:
            print(f"Tracing {image_path.name} preset={preset_name} ...")
            try:
                row = _run_one(image_path, preset_name, out_dir, args.rasterize)
                results.append(row)
                print(
                    f"  -> {row['svg_file']} paths={row['path_count']} "
                    f"size={row['svg_length']} trace={row['trace_width']}x{row['trace_height']}"
                )
            except Exception as exc:
                print(f"  FAILED: {exc}", file=sys.stderr)
                results.append({
                    "image": image_path.name,
                    "preset": preset_name,
                    "error": str(exc),
                })

    report_path = out_dir / "comparison_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path.resolve()}")
    print(f"Output dir: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
