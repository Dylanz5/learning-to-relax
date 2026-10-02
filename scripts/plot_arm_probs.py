#!/usr/bin/env python3
"""Replot arm-probability heatmaps from a saved run .npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ltr.plot import plot_arm_probabilities


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=str)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()
    path = Path(args.npz)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    z = np.load(path)
    keys: list[tuple[str, str]] = []
    if "algo_names" in z:
        for name in z["algo_names"]:
            name = str(name)
            keys.append((f"{name}_action_probs", name))
    else:
        keys = [
            ("tinf_action_probs", "tinf"),
            ("exp3_action_probs", "exp3"),
            ("tspec_action_probs", "tspec"),
        ]
    for key, name in keys:
        if key not in z:
            continue
        arr = np.asarray(z[key], dtype=float)
        mean = np.mean(arr, axis=1) if arr.ndim == 3 else arr
        dest = out_dir / f"{name}_action_probs.png"
        plot_arm_probabilities(mean, dest, title=f"{name}: arm probability vs time")
        print(dest)


if __name__ == "__main__":
    main()
