#!/usr/bin/env python3
"""Replot regret vs number of configs from a sweep results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ltr.plot import plot_regret_vs_k


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=str)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    path = Path(args.csv)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "K" not in rows[0]:
        raise SystemExit(f"{path}: expected a K-sweep results.csv with a K column")
    ks = np.array([float(r["K"]) for r in rows])
    series = {
        "Tsallis-INF": np.array([float(r["tinf_total"]) for r in rows]),
        "Exp3-Spectral": np.array([float(r["exp3_total"]) for r in rows]),
        "Tsallis-Spectral": np.array([float(r["tspec_total"]) for r in rows]),
    }
    if "best_fixed_total" in rows[0]:
        series["best fixed ω"] = np.array([float(r["best_fixed_total"]) for r in rows])
    out = Path(args.out) if args.out else path.with_name("regret_vs_k.png")
    plot_regret_vs_k(ks, series, out, title=path.stem)
    print(out)


if __name__ == "__main__":
    main()
