"""Paper figures: regret vs time, regret vs K, arm probabilities vs time."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def extract_action_probs(learner: Any, n_arms: int) -> np.ndarray | None:
    fn = getattr(learner, "action_probabilities", None)
    if callable(fn):
        try:
            arr = np.asarray(fn(), dtype=float).reshape(-1)
        except Exception:
            arr = None
        else:
            if arr.shape == (n_arms,) and np.all(np.isfinite(arr)) and float(np.sum(arr)) > 0:
                arr = np.maximum(arr, 0.0)
                s = float(np.sum(arr))
                if s > 0:
                    return arr / s
    raw = getattr(learner, "_last_p", None)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.shape != (n_arms,) or not np.all(np.isfinite(arr)) or float(np.sum(arr)) <= 0:
        return None
    arr = np.maximum(arr, 0.0)
    s = float(np.sum(arr))
    return arr / s if s > 0 else None


def plot_regret_over_time(
    curves: Mapping[str, np.ndarray],
    out_path: Path,
    *,
    title: str = "",
    ylabel: str = "cumulative loss",
) -> None:
    """``curves`` maps legend label → mean cumulative loss, shape ``(T,)``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y in curves.items():
        y = np.asarray(y, dtype=float).reshape(-1)
        ax.plot(np.arange(1, y.size + 1), y, lw=2, label=name)
    ax.set_xlabel("timestep", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def plot_regret_vs_k(
    ks: np.ndarray,
    series: Mapping[str, np.ndarray],
    out_path: Path,
    *,
    title: str = "",
    ylabel: str = "final cumulative loss",
) -> None:
    """``series`` maps legend label → values aligned with ``ks``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ks = np.asarray(ks, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y in series.items():
        ax.plot(ks, np.asarray(y, dtype=float).reshape(-1), marker="o", lw=2, label=name)
    ax.set_xlabel("number of configs (arms K)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def plot_arm_probabilities(prob_matrix: np.ndarray, out_path: Path, *, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = np.asarray(prob_matrix, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"expected (T, K), got {p.shape}")
    T, n_arms = p.shape
    fig, ax = plt.subplots(figsize=(10, 5))
    positive = p[p > 0.0]
    vmin = float(positive.min()) if positive.size else 1e-12
    vmax = max(vmin, float(np.max(p)))
    im = ax.imshow(
        p.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xlabel("timestep", fontsize=12)
    ax.set_ylabel("arm index", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, max(0, T - 1))
    ax.set_ylim(0, max(0, n_arms - 1))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("selection probability (log scale)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)
