"""
Vercel serverless entry point.

Vercel only treats files inside the `api/` directory as Serverless Functions.
This thin wrapper puts the repo root on sys.path (so `server` and its sibling
modules import cleanly) and pins the working directory to the root (so Flask's
`send_from_directory('.', ...)` calls resolve against the project files), then
re-exports the Flask `app` as the WSGI handler.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from server import app  # noqa: E402

# Vercel's Python runtime detects the module-level WSGI callable named `app`.
