"""Overlay fixed-omega baselines with learner curves from another run.

This script is meant to mix-and-match:
- omegas + omega_costs from an "omega source" run (typically produced by scripts/learning.py)
- tinf_costs + exp3_costs from a "learner source" run of your choosing

It plots mean cumulative iterations (x-axis) vs instances remaining (y-axis),
matching the style used in scripts/learning.py.

Example:

    PYTHONPATH=src MPLCONFIGDIR=.mplcache \\
    python scripts/plot_omegas_with_learners.py \\
      --omega-source plots/double-chain-64lv_20260507_0018_trials8_learning_low_variance.npz \\
      --learner-source plots/double-chain-64lv_20260506_2245_trials8_learning_low_variance.npz \\
      --out plots/mixed_omega_vs_learners.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _require_keys(z: np.lib.npyio.NpzFile, keys: list[str], *, path: Path) -> None:
    missing = [k for k in keys if k not in z]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}; have {z.files}")


def _mean_cum(x: np.ndarray) -> np.ndarray:
    """Mean over trials of the cumulative sum over time."""
    # Expect x shape (T, trials)
    return np.mean(np.cumsum(x, axis=0), axis=1)


def _mean_std_cum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over trials of the cumulative sum over time."""
    # x: (T, trials)
    cs = np.cumsum(x, axis=0)
    return np.mean(cs, axis=1), np.std(cs, axis=1)

def _as_npz_path(path: Path) -> Path:
    """Accept either .npz or a matching .png stem (swap to .npz)."""
    if path.suffix.lower() == ".npz":
        return path
    if path.suffix.lower() == ".png":
        cand = path.with_suffix(".npz")
        if cand.is_file():
            return cand
        raise SystemExit(f"{path}: got a .png; expected .npz. Also tried {cand} but it does not exist.")
    raise SystemExit(f"{path}: expected a .npz (or .png with matching .npz), got suffix {path.suffix!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omega-source", type=str, required=True, help=".npz containing omegas and omega_costs")
    p.add_argument("--learner-source", type=str, required=True, help=".npz containing tinf_costs and exp3_costs")
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    p.add_argument("--title", type=str, default=None, help="Optional plot title")
    p.add_argument(
        "--max-legend-omegas",
        type=int,
        default=12,
        help="If there are more omegas than this, omit per-omega legend entries.",
    )
    p.add_argument(
        "--std-band",
        action="store_true",
        help="Shade ±1 standard deviation bands over trials for Tsallis-INF and Exp3 curves.",
    )
    args = p.parse_args()

    omega_path = _as_npz_path(Path(args.omega_source))
    learner_path = _as_npz_path(Path(args.learner_source))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with np.load(omega_path, allow_pickle=True) as z:
        _require_keys(z, ["omegas", "omega_costs"], path=omega_path)
        omegas = np.asarray(z["omegas"], dtype=float).reshape(-1)
        omega_costs = np.asarray(z["omega_costs"])

    with np.load(learner_path, allow_pickle=True) as z:
        _require_keys(z, ["tinf_costs", "exp3_costs"], path=learner_path)
        tinf_costs = np.asarray(z["tinf_costs"])
        exp3_costs = np.asarray(z["exp3_costs"])

    if omega_costs.ndim != 3:
        raise SystemExit(f"{omega_path}: expected omega_costs shape (T, trials, n_omega), got {omega_costs.shape}")
    if tinf_costs.ndim != 2 or exp3_costs.ndim != 2:
        raise SystemExit(
            f"{learner_path}: expected tinf_costs/exp3_costs shape (T, trials), got {tinf_costs.shape}/{exp3_costs.shape}"
        )

    T_omega, trials_omega, n_omega = omega_costs.shape
    T_learn, trials_learn = tinf_costs.shape
    T = int(min(T_omega, T_learn, exp3_costs.shape[0]))
    trials = int(min(trials_omega, trials_learn, exp3_costs.shape[1]))
    if T <= 0 or trials <= 0:
        raise SystemExit("no overlapping T/trials to plot")

    # Align by truncating to common prefix.
    omega_costs = omega_costs[:T, :trials, :]
    tinf_costs = tinf_costs[:T, :trials]
    exp3_costs = exp3_costs[:T, :trials]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Baselines: fixed omegas.
    plot_all = n_omega > args.max_legend_omegas
    for j in range(n_omega):
        x = _mean_cum(omega_costs[:, :, j])
        y = T - np.arange(1, T + 1)
        ax.plot(
            x,
            y,
            lw=1.5 if plot_all else 2.0,
            ls="--",
            alpha=0.25 if plot_all else 0.9,
        )

    # Learners.
    y = T - np.arange(1, T + 1)
    if args.std_band:
        # Main means.
        m_t, s_t = _mean_std_cum(tinf_costs)
        m_e, s_e = _mean_std_cum(exp3_costs)
        ax.plot(m_t, y, lw=2.5, color="black")
        ax.plot(m_e, y, lw=2.5, color="C0")
        # ±1 std as thin companion curves + filled band between them.
        ax.plot(m_t + s_t, y, lw=1.0, color="black", alpha=0.5)
        ax.plot(m_t - s_t, y, lw=1.0, color="black", alpha=0.5)
        ax.plot(m_e + s_e, y, lw=1.0, color="C0", alpha=0.5)
        ax.plot(m_e - s_e, y, lw=1.0, color="C0", alpha=0.5)
        # Shade the area between mean and each error line (two half-bands).
        band_alpha = 0.14
        ax.fill_betweenx(y, m_t, m_t + s_t, color="black", alpha=band_alpha, linewidth=0)
        ax.fill_betweenx(y, m_t - s_t, m_t, color="black", alpha=band_alpha, linewidth=0)
        ax.fill_betweenx(y, m_e, m_e + s_e, color="C0", alpha=band_alpha, linewidth=0)
        ax.fill_betweenx(y, m_e - s_e, m_e, color="C0", alpha=band_alpha, linewidth=0)
    else:
        ax.plot(_mean_cum(tinf_costs), y, lw=2.5, color="black")
        ax.plot(_mean_cum(exp3_costs), y, lw=2.5, color="C0")

    ax.set_xlabel("total solver iterations", fontsize=14)
    ax.set_ylabel("instances remaining", fontsize=14)

    if plot_all:
        ax.legend(
            ["fixed omegas (from omega-source)", "Tsallis-INF (from learner-source)", "Exp3-Spectral (from learner-source)"],
            fontsize=10,
        )
    else:
        leg = [f"$\\omega={w:.4g}$" for w in omegas] + ["Tsallis-INF", "Exp3-Spectral"]
        ax.legend(leg, fontsize=10)


    ax.set_title(f"N=256", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=256)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

