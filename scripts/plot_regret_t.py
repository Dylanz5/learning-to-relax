#!/usr/bin/env python3
"""Replot regret vs timestep from a saved run .npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ltr.plot import plot_regret_over_time

_LABELS = {
    "tinf": "Tsallis-INF",
    "exp3": "Exp3-Spectral",
    "tspec": "Tsallis-Spectral",
    "best_arm": "best arm",
    "best_fixed": "best fixed ω",
    "best_sampled": "best sampled ω",
}


def _curves_from_npz(z) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    names = [str(n) for n in z["algo_names"]] if "algo_names" in z else []
    keys = []
    if names:
        keys = [(f"{n}_costs", n) for n in names]
    else:
        keys = [
            ("tinf_costs", "tinf"),
            ("exp3_costs", "exp3"),
            ("tspec_costs", "tspec"),
        ]
    seen = {k for k, _ in keys}
    for key in z.files:
        if key.endswith("_costs") and key not in seen and key != "omega_costs":
            keys.append((key, key[: -len("_costs")]))
            seen.add(key)
    for key, name in keys:
        if key not in z:
            continue
        c = np.asarray(z[key], dtype=float)
        if c.ndim == 1:
            curves[_LABELS.get(name, name)] = np.cumsum(c)
        else:
            curves[_LABELS.get(name, name)] = np.mean(np.cumsum(c, axis=0), axis=1)
    if "omega_costs" in z:
        om = np.asarray(z["omega_costs"], dtype=float)
        label = "best sampled ω" if "best_arm_costs" in z.files else "best fixed ω"
        curves[label] = np.mean(np.cumsum(om.min(axis=2), axis=0), axis=1)
    return curves


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=str)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    path = Path(args.npz)
    z = np.load(path)
    curves = _curves_from_npz(z)
    out = Path(args.out) if args.out else path.with_name("regret_t.png")
    plot_regret_over_time(curves, out, title=path.parent.name)
    print(out)


if __name__ == "__main__":
    main()
