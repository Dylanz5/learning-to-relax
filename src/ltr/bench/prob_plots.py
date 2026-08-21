"""Shared probability/loss-estimate capture and heatmap plotting.

These helpers were originally embedded in ``scripts/learning.py`` (the live
bandit driver). They are factored out here so the offline replay driver
(``scripts/run_learning_precomputed.py``) can produce the same
action-probability and loss-estimate heatmaps without duplicating the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def extract_action_probs(learner: Any, n_arms: int) -> np.ndarray | None:
    """Best-effort extraction of the learner's latest arm distribution.

    Tries the common accessor methods first, then a few well-known attributes.
    Returns a normalized ``(n_arms,)`` distribution, or ``None`` if nothing
    usable is exposed.
    """
    candidates = (
        "action_probabilities",
        "get_action_probabilities",
        "last_action_probabilities",
    )
    for attr in candidates:
        fn = getattr(learner, attr, None)
        if callable(fn):
            try:
                arr = np.asarray(fn(), dtype=float).reshape(-1)
            except Exception:
                continue
            if arr.shape == (n_arms,) and np.all(np.isfinite(arr)) and float(np.sum(arr)) > 0:
                arr = np.maximum(arr, 0.0)
                s = float(np.sum(arr))
                if s > 0:
                    return arr / s
    for attr in ("_last_p", "last_p", "p"):
        raw = getattr(learner, attr, None)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.shape == (n_arms,) and np.all(np.isfinite(arr)) and float(np.sum(arr)) > 0:
            arr = np.maximum(arr, 0.0)
            s = float(np.sum(arr))
            if s > 0:
                return arr / s
    return None


def plot_action_probability_heatmap(prob_matrix: np.ndarray, out_path: Path, *, title: str) -> None:
    """Plot arm-choice probability heatmap with y=arm and x=time."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = np.asarray(prob_matrix, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"expected 2D matrix (T, arms), got shape {p.shape}")

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
    ax.set_xlabel("time step", fontsize=12)
    ax.set_ylabel("arm index", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, max(0, T - 1))
    ax.set_ylim(0, max(0, n_arms - 1))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("selection probability (log scale)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def plot_loss_estimate_heatmap(loss_matrix: np.ndarray, out_path: Path, *, title: str) -> None:
    """Plot per-arm normalized cumulative loss ``L_i = k_i/scale`` (y=arm, x=time).

    These are what the Tsallis-INF distribution is built on (``p_i ∝ (L_i-x)^-2``),
    so the arm tracing the *lowest* band wins the most probability. The gap between
    the lowest band and the rest is what drives the collapse; sudden vertical shifts
    correspond to jumps in ``scale`` (the running loss-mean normalizer).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(loss_matrix, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected 2D matrix (T, arms), got shape {a.shape}")

    T, n_arms = a.shape
    finite = a[np.isfinite(a)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if not (vmax > vmin):
        vmax = vmin + 1.0
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        a.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("time step", fontsize=12)
    ax.set_ylabel("arm index", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, max(0, T - 1))
    ax.set_ylim(0, max(0, n_arms - 1))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"normalized cumulative loss $L_i = k_i/\mathrm{scale}$", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)
