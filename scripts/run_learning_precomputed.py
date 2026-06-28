"""Replay Tsallis-INF and Exp3-Spectral on archived full-information losses.

Example ``.npz`` (all arrays float64 unless noted)::

    losses_low:   (T, K, trials)   # optional
    losses_high:  (T, K, trials)   # optional
    path_similarity: (K, K)  # nonnegative edge weights; L = D - W
    grid: (K,) optional arm labels for plots

Or use ``losses_a`` / ``losses_b`` / ``losses`` instead of the ``losses_*`` names.
See ``ltr.bench.precomputed`` for the full key list.

Usage (from repo root)::

    python scripts/run_learning_precomputed.py --data data/my_experiment.npz --config configs/bench/learning_chain_default.json

Single ``.npy`` with shape ``(T, K, trials)`` (graph from built-in chain on ``K`` arms)::

    python scripts/run_learning_precomputed.py --data data/fullinfo/boxTurb32.npy --similarity-kind chain
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ltr.bench.config import LearningExperimentConfig
from ltr.bench.precomputed import PrecomputedLossBundle, load_precomputed_npz
from ltr.bench.similarity import SIMILARITY_KINDS
from ltr.learners.exp3_spectral import Exp3Spectral
from ltr.learners.tsallis_inf import TsallisINF


def _default_baseline_arm_indices(K: int, count: int) -> np.ndarray:
    if K <= 0:
        return np.array([], dtype=int)
    if count < 0:
        # Convenience: negative means "all arms".
        return np.arange(K, dtype=int)
    if count <= 0:
        return np.array([], dtype=int)
    if count >= K:
        return np.arange(K, dtype=int)
    return np.unique(np.linspace(0, K - 1, count, dtype=int))


def _run_one_bundle(
    bundle: PrecomputedLossBundle,
    *,
    cfg: LearningExperimentConfig,
    seed: int,
    baseline_arm_indices: np.ndarray,
    plots_dir: Path,
    run_label: str,
    save_loss_npz: bool,
) -> None:
    T, K, trials = bundle.T, bundle.K, bundle.trials
    losses = bundle.losses
    grid = bundle.grid

    tinf_costs = np.zeros((T, trials), dtype=float)
    exp3_costs = np.zeros((T, trials), dtype=float)

    baseline_arm_indices = np.asarray(baseline_arm_indices, dtype=int).reshape(-1)
    omega_costs = np.stack([losses[:, j, :] for j in baseline_arm_indices], axis=2)

    for k in range(trials):
        rng = np.random.default_rng(int(seed) + k)
        tinf = TsallisINF(grid, T=T)
        exp3 = Exp3Spectral(
            grid=grid,
            eigenvectors=bundle.U,
            eigenvalues=bundle.lam,
            eta=cfg.exp3_eta,
            gamma=cfg.exp3_gamma,
            mu=cfg.exp3_mu,
            smoothness=cfg.exp3_smoothness,
        )
        for t in range(T):
            tinf.predict(rng=rng)
            assert tinf.index is not None
            ti = int(tinf.index)
            tinf_costs[t, k] = losses[t, ti, k]
            tinf.update(tinf_costs[t, k])

            exp3.predict(rng=rng)
            assert exp3._last_i is not None
            ei = int(exp3._last_i)
            exp3_costs[t, k] = losses[t, ei, k]
            exp3.update(exp3_costs[t, k])

    now = datetime.datetime.now()
    prefix = (cfg.run_name or run_label).strip()
    prefix = f"{prefix}_" if prefix else ""
    png_name = f"{prefix}{now.strftime('%Y%m%d_%H%M')}_trials{trials}_precomputed_{run_label}.png"

    if save_loss_npz:
        npz_name = Path(png_name).with_suffix(".npz").name
        out_npz = plots_dir / npz_name
        np.savez_compressed(
            out_npz,
            tinf_costs=tinf_costs.astype(np.float64),
            exp3_costs=exp3_costs.astype(np.float64),
            omega_costs=omega_costs.astype(np.float64),
            baseline_arm_indices=baseline_arm_indices.astype(np.int64),
            T=np.int32(T),
            trials=np.int32(trials),
            seed=np.int32(seed),
            bundle=np.array(run_label),
        )
        print(f"[precomputed] saved arrays to {out_npz}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_all_arms = baseline_arm_indices.size == K
    for j in range(baseline_arm_indices.size):
        # With many arms the plot gets dense; keep the baselines light so the
        # learner traces remain visually dominant.
        alpha = 0.18 if plot_all_arms else 1.0
        lw = 1.0 if plot_all_arms else 2.0
        ax.plot(
            np.mean(np.cumsum(omega_costs[:, :, j], axis=0), axis=1),
            T - np.arange(1, T + 1),
            lw=lw,
            ls="--",
            alpha=alpha,
        )
    ax.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2, color="black")
    ax.plot(np.mean(np.cumsum(exp3_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2)
    ax.set_xlabel("total loss (cumulative)", fontsize=14)
    ax.set_ylabel("instances remaining", fontsize=14)
    if plot_all_arms:
        # A legend with K entries is unreadable; keep a compact legend and annotate the rest.
        ax.legend(["baselines (all arms)", "Tsallis-INF", "Exp3-Spectral"], fontsize=10)
        ax.set_title(f"{run_label}: all {K} arms", fontsize=12)
    else:
        leg = [
            f"arm {int(baseline_arm_indices[j])} (grid={grid[int(baseline_arm_indices[j])]:.4g})"
            for j in range(baseline_arm_indices.size)
        ]
        leg += ["Tsallis-INF", "Exp3-Spectral"]
        ax.legend(leg, fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / png_name, dpi=256)
    plt.close(fig)
    print(f"[precomputed] saved figure {plots_dir / png_name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        "--npz",
        dest="data",
        type=str,
        required=True,
        help="Path to .npz (loss arrays + graph) or .npy (one (T,K,trials) tensor).",
    )
    p.add_argument(
        "--path-similarity",
        type=str,
        default=None,
        help="Optional K×K .npy: edge weights W (L=D−W) unless --similarity-is-laplacian.",
    )
    p.add_argument(
        "--similarity-kind",
        type=str,
        default=None,
        choices=SIMILARITY_KINDS,
        help="Build Laplacian from a standard graph when the archive has no graph (required for plain .npy unless --path-similarity).",
    )
    p.add_argument(
        "--similarity-is-laplacian",
        action="store_true",
        help="Treat --path-similarity as a Laplacian matrix instead of edge weights.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config for Exp3 / plot prefix (optional).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None, help="Plot filename prefix.")
    p.add_argument(
        "--baselines",
        type=int,
        default=5,
        help="Number of fixed arms to plot (evenly spaced indices); 0 skips baselines.",
    )
    p.add_argument(
        "--all-arms",
        action="store_true",
        help="Plot every arm as a baseline (equivalent to --baselines -1).",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="If set, only run this bundle key (e.g. losses_low).",
    )
    p.add_argument(
        "--save-loss-npz",
        action="store_true",
        help="Also write tinf_costs / exp3_costs next to the plot.",
    )
    args = p.parse_args()

    cfg = (
        LearningExperimentConfig.from_json_file(args.config)
        if args.config
        else LearningExperimentConfig()
    )
    if args.run_name is not None:
        cfg.run_name = args.run_name

    bundles = load_precomputed_npz(
        args.data,
        path_similarity=args.path_similarity,
        similarity_kind=args.similarity_kind,
        similarity_is_laplacian=args.similarity_is_laplacian,
    )
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    keys = [k for k in bundles if args.only is None or k == args.only]
    if args.only is not None and args.only not in bundles:
        raise SystemExit(f"--only {args.only!r} not in npz keys: {sorted(bundles)}")

    for name, bundle in bundles.items():
        if name not in keys:
            continue
        baselines = -1 if args.all_arms else args.baselines
        arms = _default_baseline_arm_indices(bundle.K, baselines)
        _run_one_bundle(
            bundle,
            cfg=cfg,
            seed=args.seed,
            baseline_arm_indices=arms,
            plots_dir=plots_dir,
            run_label=name,
            save_loss_npz=args.save_loss_npz,
        )


if __name__ == "__main__":
    main()
