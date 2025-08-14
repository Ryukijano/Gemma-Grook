#!/usr/bin/env python3
"""
Execute a Jupyter notebook as if it were a script.

- Loads the notebook, runs all cells in order, and writes an executed copy.
- Designed to be side-effect free beyond notebook execution; makes no changes in-place unless --inplace is used.
- Sets sensible JAX GPU env defaults if not already set.

Usage:
  python scripts/run_notebook.py --in /scratch/cbjp404/Isaac-GR00T/test_notebook5.ipynb
  python scripts/run_notebook.py --in /abs/path.ipynb --out /abs/out.ipynb --timeout 7200 --allow-errors
  python scripts/run_notebook.py --in /abs/path.ipynb --save-html
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def _set_default_env() -> None:
    # Favor GPU without preallocating full memory; allow fallback to CPU.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Try CUDA first if present, then CPU. Do not force a single platform here.
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    # Avoid TF grabbing GPUs if present in the same env
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    # Headless plotting by default
    os.environ.setdefault("MPLBACKEND", "Agg")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a Jupyter notebook and write the executed copy")
    parser.add_argument("--in", dest="nb_in", required=True, help="Absolute path to input .ipynb")
    parser.add_argument("--out", dest="nb_out", default=None, help="Absolute path to output .ipynb")
    parser.add_argument("--timeout", type=int, default=3600, help="Cell execution timeout in seconds")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    parser.add_argument("--allow-errors", action="store_true", help="Continue writing even if execution errors occur")
    parser.add_argument("--inplace", action="store_true", help="Write the executed notebook back to the input path")
    parser.add_argument("--save-html", action="store_true", help="Also export an HTML report next to the output ipynb")
    return parser.parse_args(argv)


def derive_default_out_path(nb_in: Path) -> Path:
    stem = nb_in.stem
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return nb_in.with_name(f"{stem}-executed-{ts}.ipynb")


def save_html(nb_node, out_ipynb: Path) -> Path:
    try:
        from nbconvert import HTMLExporter
        from nbconvert.writers import FilesWriter
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] nbconvert not available for HTML export: {exc}")
        return out_ipynb

    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(nb_node)
    writer = FilesWriter()
    out_html = out_ipynb.with_suffix(".html")
    writer.write(body, resources, notebook_name=out_html.stem, resources_dir=out_html.stem + "_files")
    return out_html


def main(argv: list[str]) -> int:
    _set_default_env()
    args = parse_args(argv)

    in_path = Path(args.nb_in).resolve()
    if not in_path.exists():
        print(f"[ERROR] Notebook not found: {in_path}")
        return 2

    if args.inplace:
        out_path = in_path
    else:
        out_path = Path(args.nb_out).resolve() if args.nb_out else derive_default_out_path(in_path)

    print(f"[INFO] Executing: {in_path}")
    print(f"[INFO] Writing to: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=args.timeout, kernel_name=args.kernel, allow_errors=args.allow_errors)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(in_path.parent)}})
    except Exception as exc:  # pragma: no cover - surfaced to user
        print(f"[ERROR] Execution failed: {exc}")
        if not args.allow_errors:
            return 1

    with out_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    if args.save_html:
        out_html = save_html(nb, out_path)
        print(f"[INFO] HTML report: {out_html}")

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


