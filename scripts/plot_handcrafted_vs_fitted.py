#!/usr/bin/env python
"""Plot Tsallis-INF vs Spectral-Tsallis (handcrafted vs fitted graph).

Reads saved run ``.npz`` files from ``scripts/run_learning_precomputed.py``
(``--save-loss-npz``) and draws cumulative wall-clock vs instance index.
No fixed-arm baselines are shown.

Default sources (pitzDailyLarge wallclock-even, Jul 27 sweeps):
- handcrafted: best hand-crafted Laplacian Spectral-Tsallis run
- fitted: best ``laplacian_per_t_full`` Spectral-Tsallis run
- Tsallis-INF is taken from the handcrafted ``.npz`` (graph-independent)

Example::

    PYTHONPATH=src MPLCONFIGDIR=.mplcache \\
    python scripts/plot_handcrafted_vs_fitted.py \\
      --out plots/pitzDailyLarge_handcrafted_vs_fitted.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

DEFAULT_HANDCRAFTED = (
    REPO
    / "plots/sweeps/20260727_112608_tspec_pitzDailyLarge_wallclock_even_pitzDailyLarge_laplacian_hand_crafted"
    / "runs/20260727_1308_sweep_et0p3_g0_mu0p001_sm0_a0p75_s0_pitzDailyLarge_laplacian_hand_crafted_trials1_precomputed_pitzDailyLarge_wallclock_even"
    / "20260727_1308_sweep_et0p3_g0_mu0p001_sm0_a0p75_s0_pitzDailyLarge_laplacian_hand_crafted_trials1_precomputed_pitzDailyLarge_wallclock_even.npz"
)

DEFAULT_FITTED = (
    REPO
    / "plots/sweeps/20260727_132711_tspec_pitzDailyLarge_wallclock_even_pitzDailyLarge_laplacian_per_t_full"
    / "runs/20260727_1357_sweep_et0p1_g0_mu0p001_sm0_a0p9_s0_pitzDailyLarge_laplacian_per_t_full_trials1_precomputed_pitzDailyLarge_wallclock_even"
    / "20260727_1357_sweep_et0p1_g0_mu0p001_sm0_a0p9_s0_pitzDailyLarge_laplacian_per_t_full_trials1_precomputed_pitzDailyLarge_wallclock_even.npz"
)


def _as_npz_path(path: Path) -> Path:
    if path.suffix.lower() == ".npz":
        return path
    if path.suffix.lower() == ".png":
        cand = path.with_suffix(".npz")
        if cand.is_file():
            return cand
        raise SystemExit(f"{path}: expected matching .npz at {cand}")
    raise SystemExit(f"{path}: expected a .npz (or .png with matching .npz)")


def _require_keys(z: np.lib.npyio.NpzFile, keys: list[str], *, path: Path) -> None:
    missing = [k for k in keys if k not in z]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}; have {z.files}")


def _mean_cum(costs: np.ndarray) -> np.ndarray:
    """Mean over trials of cumulative sum over time. ``costs``: (T, trials)."""
    return np.mean(np.cumsum(costs, axis=0), axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--handcrafted",
        type=str,
        default=str(DEFAULT_HANDCRAFTED),
        help="Run .npz with Spectral-Tsallis on the handcrafted graph (also supplies Tsallis-INF).",
    )
    p.add_argument(
        "--fitted",
        type=str,
        default=str(DEFAULT_FITTED),
        help="Run .npz with Spectral-Tsallis on the fitted (per-t full) graph.",
    )
    p.add_argument(
        "--tinf-source",
        type=str,
        default=None,
        help="Optional .npz for Tsallis-INF (defaults to --handcrafted).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(REPO / "plots/pitzDailyLarge_handcrafted_vs_fitted.png"),
        help="Output PNG path.",
    )
    p.add_argument("--title", type=str, default=None, help="Optional plot title.")
    args = p.parse_args()

    hand_path = _as_npz_path(Path(args.handcrafted))
    fit_path = _as_npz_path(Path(args.fitted))
    tinf_path = _as_npz_path(Path(args.tinf_source)) if args.tinf_source else hand_path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with np.load(hand_path, allow_pickle=False) as z:
        _require_keys(z, ["tspec_costs"], path=hand_path)
        hand_tspec = np.asarray(z["tspec_costs"], dtype=float)

    with np.load(fit_path, allow_pickle=False) as z:
        _require_keys(z, ["tspec_costs"], path=fit_path)
        fit_tspec = np.asarray(z["tspec_costs"], dtype=float)

    with np.load(tinf_path, allow_pickle=False) as z:
        _require_keys(z, ["tinf_costs"], path=tinf_path)
        tinf = np.asarray(z["tinf_costs"], dtype=float)

    for name, arr in (
        ("handcrafted tspec_costs", hand_tspec),
        ("fitted tspec_costs", fit_tspec),
        ("tinf_costs", tinf),
    ):
        if arr.ndim != 2:
            raise SystemExit(f"{name}: expected shape (T, trials), got {arr.shape}")

    T = min(hand_tspec.shape[0], fit_tspec.shape[0], tinf.shape[0])
    if not (hand_tspec.shape[0] == fit_tspec.shape[0] == tinf.shape[0]):
        print(
            f"[warn] truncating to shared T={T} "
            f"(hand={hand_tspec.shape[0]}, fitted={fit_tspec.shape[0]}, tinf={tinf.shape[0]})",
            flush=True,
        )
        hand_tspec = hand_tspec[:T]
        fit_tspec = fit_tspec[:T]
        tinf = tinf[:T]

    instances = np.arange(1, T + 1)
    y_tinf = _mean_cum(tinf)
    y_hand = _mean_cum(hand_tspec)
    y_fit = _mean_cum(fit_tspec)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(instances, y_tinf, lw=2, label="Tsallis-INF")
    ax.plot(instances, y_hand, lw=2, label="Spectral-Tsallis with handcrafted graph")
    ax.plot(instances, y_fit, lw=2, label="Spectral-Tsallis with fitted graph")
    ax.set_xlabel("instances", fontsize=14)
    ax.set_ylabel("cumulative wallclock", fontsize=14)
    if args.title:
        ax.set_title(args.title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=256)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    print(
        f"final totals — Tsallis-INF={y_tinf[-1]:.4g}, "
        f"handcrafted={y_hand[-1]:.4g}, fitted={y_fit[-1]:.4g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
