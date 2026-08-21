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
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ltr.bench.config import LearningExperimentConfig
from ltr.bench.precomputed import PrecomputedLossBundle, load_precomputed_npz
from ltr.bench.prob_plots import (
    extract_action_probs,
    plot_action_probability_heatmap,
    plot_loss_estimate_heatmap,
)
from ltr.bench.similarity import SIMILARITY_KINDS
from ltr.learners.exp3_spectral import Exp3Spectral
from ltr.learners.tsallis_inf import TsallisINF
from ltr.learners.tsallis_spectral import TsallisSpectral


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
    graph_tag: str,
    save_loss_npz: bool,
) -> None:
    T, K, trials = bundle.T, bundle.K, bundle.trials
    losses = bundle.losses
    grid = bundle.grid

    tinf_costs = np.zeros((T, trials), dtype=float)
    exp3_costs = np.zeros((T, trials), dtype=float)
    tspec_costs = np.zeros((T, trials), dtype=float)

    # Per-round diagnostics captured for the same heatmaps ``scripts/learning.py``
    # produces: arm-selection probabilities per learner, plus Tsallis-INF's
    # per-arm normalized cumulative loss ``L_i = k_i/scale``.
    tinf_action_probs = np.zeros((T, trials, K), dtype=np.float64)
    exp3_action_probs = np.zeros((T, trials, K), dtype=np.float64)
    tspec_action_probs = np.zeros((T, trials, K), dtype=np.float64)
    tinf_loss_estimates = np.zeros((T, trials, K), dtype=np.float64)

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
            L=bundle.laplacian,
        )
        tspec = TsallisSpectral(
            grid=grid,
            eigenvectors=bundle.U,
            eigenvalues=bundle.lam,
            eta=cfg.tsallis_spectral_eta,
            gamma=cfg.tsallis_spectral_gamma,
            mu=cfg.tsallis_spectral_mu,
            smoothness=cfg.tsallis_spectral_smoothness,
            alpha=cfg.tsallis_spectral_alpha,
            L=bundle.laplacian,
        )
        for t in range(T):
            tinf.predict(rng=rng)
            assert tinf.index is not None
            ti = int(tinf.index)
            tp = extract_action_probs(tinf, K)
            if tp is not None:
                tinf_action_probs[t, k, :] = tp
            le = np.asarray(tinf.loss_estimates(), dtype=float).reshape(-1)
            if le.shape == (K,) and np.all(np.isfinite(le)):
                tinf_loss_estimates[t, k, :] = le
            tinf_costs[t, k] = losses[t, ti, k]
            tinf.update(tinf_costs[t, k])

            exp3.predict(rng=rng)
            assert exp3._last_i is not None
            ei = int(exp3._last_i)
            ep = extract_action_probs(exp3, K)
            if ep is not None:
                exp3_action_probs[t, k, :] = ep
            exp3_costs[t, k] = losses[t, ei, k]
            exp3.update(exp3_costs[t, k])

            tspec.predict(rng=rng)
            assert tspec._last_i is not None
            si = int(tspec._last_i)
            sp_ = extract_action_probs(tspec, K)
            if sp_ is not None:
                tspec_action_probs[t, k, :] = sp_
            tspec_costs[t, k] = losses[t, si, k]
            tspec.update(tspec_costs[t, k])

    # Graph artifacts for Exp3-Spectral. The graph is a *fixed* input (it is not
    # updated during a run), so these are identical across trials; capture them
    # from the last learner instance. We save the Laplacian actually used, the
    # recovered similarity matrix W (L = D - W, so W = diag(diag(L)) - L), the
    # graph eigenvalues, and the D-optimal exploration design q derived from it.
    laplacian = np.asarray(bundle.laplacian, dtype=np.float64)
    similarity = np.diag(np.diag(laplacian)) - laplacian
    eigenvalues = np.asarray(bundle.lam, dtype=np.float64)
    exp3_exploration = np.asarray(exp3.q, dtype=np.float64)

    now = datetime.datetime.now()
    prefix = (cfg.run_name or run_label).strip()
    prefix = f"{prefix}_" if prefix else ""
    gtag = f"{graph_tag}_" if graph_tag else ""
    # Timestamp first so runs sort chronologically by name (folder + all artifacts).
    # The graph tag is included so full-info vs two-chains (etc.) runs are distinguishable.
    png_name = f"{now.strftime('%Y%m%d_%H%M')}_{prefix}{gtag}trials{trials}_precomputed_{run_label}.png"

    # Collect all artifacts for this run (main figure, optional .npz, and the
    # three heatmaps) in a dedicated subfolder under plots/.
    stem = Path(png_name).stem
    run_dir = plots_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    if save_loss_npz:
        npz_name = Path(png_name).with_suffix(".npz").name
        out_npz = run_dir / npz_name
        np.savez_compressed(
            out_npz,
            tinf_costs=tinf_costs.astype(np.float64),
            exp3_costs=exp3_costs.astype(np.float64),
            tspec_costs=tspec_costs.astype(np.float64),
            omega_costs=omega_costs.astype(np.float64),
            tinf_action_probs=tinf_action_probs.astype(np.float32),
            exp3_action_probs=exp3_action_probs.astype(np.float32),
            tspec_action_probs=tspec_action_probs.astype(np.float32),
            tinf_loss_estimates=tinf_loss_estimates.astype(np.float32),
            baseline_arm_indices=baseline_arm_indices.astype(np.int64),
            laplacian=laplacian,
            similarity=similarity,
            eigenvalues=eigenvalues,
            exp3_exploration=exp3_exploration,
            T=np.int32(T),
            trials=np.int32(trials),
            seed=np.int32(seed),
            bundle=np.array(run_label),
        )
        print(f"[precomputed] saved arrays to {out_npz}", flush=True)

    # Probability / loss-estimate heatmaps (mean over trials), mirroring
    # ``scripts/learning.py``. Same filename stem as the cumulative-loss figure.
    tinf_probs_path = run_dir / f"{stem}_tinf_action_probs.png"
    plot_action_probability_heatmap(
        np.mean(tinf_action_probs, axis=1),
        tinf_probs_path,
        title=f"{run_label} tinf: arm selection probability over time",
    )
    print(f"[precomputed] wrote action-probability heatmap to {tinf_probs_path}", flush=True)

    exp3_probs_path = run_dir / f"{stem}_exp3_action_probs.png"
    plot_action_probability_heatmap(
        np.mean(exp3_action_probs, axis=1),
        exp3_probs_path,
        title=f"{run_label} exp3: arm selection probability over time",
    )
    print(f"[precomputed] wrote action-probability heatmap to {exp3_probs_path}", flush=True)

    tspec_probs_path = run_dir / f"{stem}_tspec_action_probs.png"
    plot_action_probability_heatmap(
        np.mean(tspec_action_probs, axis=1),
        tspec_probs_path,
        title=f"{run_label} tsallis-spectral: arm selection probability over time",
    )
    print(f"[precomputed] wrote action-probability heatmap to {tspec_probs_path}", flush=True)

    tinf_loss_est_path = run_dir / f"{stem}_tinf_loss_estimates.png"
    plot_loss_estimate_heatmap(
        np.mean(tinf_loss_estimates, axis=1),
        tinf_loss_est_path,
        title=f"{run_label} tinf: per-arm normalized cumulative loss $L_i$ over time",
    )
    print(f"[precomputed] wrote loss-estimate heatmap to {tinf_loss_est_path}", flush=True)

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
    ax.plot(np.mean(np.cumsum(tspec_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2)
    ax.set_xlabel("total loss (cumulative)", fontsize=14)
    ax.set_ylabel("instances remaining", fontsize=14)
    if plot_all_arms:
        # A legend with K entries is unreadable; keep a compact legend and annotate the rest.
        ax.legend(["baselines (all arms)", "Tsallis-INF", "Exp3-Spectral", "Tsallis-Spectral"], fontsize=10)
    else:
        leg = [
            f"arm {int(baseline_arm_indices[j])} (grid={grid[int(baseline_arm_indices[j])]:.4g})"
            for j in range(baseline_arm_indices.size)
        ]
        leg += ["Tsallis-INF", "Exp3-Spectral", "Tsallis-Spectral"]
        ax.legend(leg, fontsize=10)
    # Title = run filename (stem) with the leading date stripped, plus the actual
    # hyperparameters of *both* spectral learners so a plot is self-describing.
    # Exp3-Spectral and Tsallis-Spectral each read their own ``exp3_*`` /
    # ``tsallis_spectral_*`` config fields; a sweep over one learner leaves the
    # other at its config defaults, so we label both to avoid confusion.
    title_base = re.sub(r"^\d{8}_\d{4}_", "", stem)
    hp_exp3 = (
        f"Exp3:  $\\eta$={cfg.exp3_eta:g}  $\\gamma$={cfg.exp3_gamma:g}  "
        f"$\\mu$={cfg.exp3_mu:g}  smooth={cfg.exp3_smoothness:g}"
    )
    hp_tspec = (
        f"Tsallis-Spec:  $\\eta$={cfg.tsallis_spectral_eta:g}  "
        f"$\\gamma$={cfg.tsallis_spectral_gamma:g}  $\\mu$={cfg.tsallis_spectral_mu:g}  "
        f"smooth={cfg.tsallis_spectral_smoothness:g}  $\\alpha$={cfg.tsallis_spectral_alpha:g}"
    )
    ax.set_title(f"{title_base}\n{hp_exp3}\n{hp_tspec}  seed={seed}", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / png_name, dpi=256)
    plt.close(fig)
    print(f"[precomputed] saved figure {run_dir / png_name}", flush=True)


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
        help=(
            "Optional K×K .npy custom graph. Auto-detects Laplacian vs edge weights "
            "(force with --similarity-is-laplacian). e.g. graphs/pitzDaily_full_info_sim_graph.npy"
        ),
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
        help="Force treating --path-similarity as a Laplacian (skips auto-detection).",
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
    p.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory to write the per-run artifact folder into (default: plots/).",
    )
    args = p.parse_args()

    cfg = (
        LearningExperimentConfig.from_json_file(args.config)
        if args.config
        else LearningExperimentConfig()
    )
    if args.run_name is not None:
        cfg.run_name = args.run_name

    # Short, filename-safe label for the graph actually used, so artifacts are
    # self-describing (e.g. full-info vs double_chain vs a graph embedded in the npz).
    if args.similarity_kind:
        graph_tag = args.similarity_kind
    elif args.path_similarity:
        graph_tag = Path(args.path_similarity).stem.replace("_sim_graph", "")
    else:
        graph_tag = "npzgraph"
    graph_tag = re.sub(r"[^0-9A-Za-z._-]+", "-", graph_tag).strip("-")

    bundles = load_precomputed_npz(
        args.data,
        path_similarity=args.path_similarity,
        similarity_kind=args.similarity_kind,
        similarity_is_laplacian=args.similarity_is_laplacian,
    )
    plots_dir = Path(args.plots_dir)
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
            graph_tag=graph_tag,
            save_loss_npz=args.save_loss_npz,
        )


if __name__ == "__main__":
    main()
