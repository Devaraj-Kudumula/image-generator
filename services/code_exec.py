"""
Sandboxed execution of LLM-generated matplotlib code for diagram reconstruction.
Local-only: runs in an isolated subprocess with AST-based safety scan.
"""
import ast
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Tuple

logger = logging.getLogger(__name__)

_ALLOWED_MODULE_ROOTS = frozenset({'matplotlib', 'numpy', 'math'})

_FORBIDDEN_MODULE_ROOTS = frozenset({
    'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests', 'shutil',
    'pathlib', 'pickle', 'ctypes', 'multiprocessing', 'threading', 'builtins',
    'importlib', 'code', 'pty', 'signal', 'http', 'ftplib', 'smtplib',
    'webbrowser', 'sqlite3', 'csv', 'tempfile', 'glob', 'io', 'PIL', 'pillow',
})

_FORBIDDEN_CALL_NAMES = frozenset({
    'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals',
    'getattr', 'delattr', 'input', 'help', 'breakpoint',
})

_FORBIDDEN_ATTR_PREFIXES = (
    'os.', 'sys.', 'subprocess.', 'socket.', 'shutil.', 'pathlib.',
)

_HARNESS_TEMPLATE = '''\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

{user_code}

if "fig" not in dir() or fig is None:
    raise RuntimeError("Code must define a matplotlib Figure named fig")

fig.savefig("out.svg", format="svg", bbox_inches="tight", pad_inches=0.05)
fig.savefig("out.png", format="png", dpi=150, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
'''


class CodeExecError(Exception):
    """Raised when generated code fails safety scan or execution."""


def _module_root(name: str) -> str:
    return (name or '').split('.')[0]


def _check_module_allowed(module_name: str) -> None:
    root = _module_root(module_name)
    if root in _FORBIDDEN_MODULE_ROOTS:
        raise CodeExecError(f"Disallowed import: {module_name}")
    if root not in _ALLOWED_MODULE_ROOTS:
        raise CodeExecError(
            f"Disallowed import: {module_name} "
            f"(only matplotlib, numpy, math are allowed)"
        )


def _attr_chain(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _attr_chain(node.value)
        if value:
            return f"{value}.{node.attr}"
        return node.attr
    return ''


def validate_matplotlib_code(code: str) -> None:
    """Raise CodeExecError if code contains disallowed constructs."""
    if not code or not code.strip():
        raise CodeExecError("Generated code is empty")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeExecError(f"Syntax error in generated code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module_allowed(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                _check_module_allowed(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CALL_NAMES:
                    raise CodeExecError(f"Disallowed call: {node.func.id}(...)")
            elif isinstance(node.func, ast.Attribute):
                chain = _attr_chain(node.func)
                for prefix in _FORBIDDEN_ATTR_PREFIXES:
                    if chain == prefix.rstrip('.') or chain.startswith(prefix):
                        raise CodeExecError(f"Disallowed call: {chain}(...)")
                if chain.split('.')[-1] in _FORBIDDEN_CALL_NAMES:
                    raise CodeExecError(f"Disallowed call: {chain}(...)")


def run_matplotlib_code(code: str, timeout_s: int = 25) -> Tuple[str, bytes]:
    """
    Execute matplotlib code in a sandboxed subprocess.

    The code must define a Figure named ``fig``. Returns (svg_string, png_bytes).
    Raises CodeExecError on safety or runtime failure.
    """
    validate_matplotlib_code(code)

    tmp_dir = tempfile.mkdtemp(prefix="diagram_refine_")
    harness_path = os.path.join(tmp_dir, "harness.py")
    svg_path = os.path.join(tmp_dir, "out.svg")
    png_path = os.path.join(tmp_dir, "out.png")

    try:
        harness_source = _HARNESS_TEMPLATE.format(user_code=code)
        with open(harness_path, "w", encoding="utf-8") as fh:
            fh.write(harness_source)

        env = {
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": tmp_dir,
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "TEMP": tmp_dir,
            "TMP": tmp_dir,
        }
        # Matplotlib needs a writable config dir and home on Windows.
        for key in (
            "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
            "APPDATA", "LOCALAPPDATA", "COMSPEC", "WINDIR",
        ):
            if key in os.environ:
                env[key] = os.environ[key]
        if "USERPROFILE" not in env and "HOME" not in env:
            env["USERPROFILE"] = tmp_dir
            env["HOME"] = tmp_dir

        result = subprocess.run(
            [sys.executable, "-I", harness_path],
            cwd=tmp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            if len(stderr) > 2000:
                stderr = stderr[:2000] + "\n...(truncated)"
            raise CodeExecError(
                f"Matplotlib execution failed (exit {result.returncode}):\n{stderr}"
            )

        if not os.path.isfile(svg_path):
            raise CodeExecError("Execution succeeded but out.svg was not created")
        if not os.path.isfile(png_path):
            raise CodeExecError("Execution succeeded but out.png was not created")

        with open(svg_path, "r", encoding="utf-8") as fh:
            svg_string = fh.read()
        with open(png_path, "rb") as fh:
            png_bytes = fh.read()

        if not svg_string.strip():
            raise CodeExecError("Generated SVG is empty")

        logger.info(
            "Matplotlib code executed successfully (svg=%d bytes, png=%d bytes)",
            len(svg_string),
            len(png_bytes),
        )
        return svg_string, png_bytes

    except subprocess.TimeoutExpired as exc:
        raise CodeExecError(
            f"Matplotlib execution timed out after {timeout_s}s"
        ) from exc
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass
