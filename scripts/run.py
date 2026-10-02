#!/usr/bin/env python3
"""Load a Python experiment config: ``python scripts/run.py configs/live_sor.py``."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("ltr_run_config", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load config {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=str, help="Python config (e.g. configs/live_sor.py)")
    args = p.parse_args()
    path = Path(args.config)
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    mod = load_config(path.resolve())
    if not hasattr(mod, "main"):
        raise SystemExit(f"{path} must define main()")
    mod.main()


if __name__ == "__main__":
    main()
