"""
Vercel build step: copy frontend static assets into public/ for CDN serving.

Vercel serves files from public/ at the site root. Generated images stay in-memory
on serverless (see config.IS_SERVERLESS) and are not copied here.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATIC_SRC = ROOT / "static"
PUBLIC_STATIC = ROOT / "public" / "static"


def main() -> None:
    if not STATIC_SRC.is_dir():
        raise SystemExit(f"Missing static source directory: {STATIC_SRC}")

    PUBLIC_STATIC.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in sorted(STATIC_SRC.iterdir()):
        if item.name == "images":
            continue
        dest = PUBLIC_STATIC / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        copied += 1

    print(f"Copied {copied} static asset(s) to {PUBLIC_STATIC}")


if __name__ == "__main__":
    main()
