from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Stabilize native stack defaults for repeatable CLI runs.
# These apply only if the user has not already set them.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from ltr.bench.config import LearningExperimentConfig
from ltr.bench.similarity import similarity_spectrum
from ltr.domains import delsq_numgrid
from ltr.learners.exp3_spectral import Exp3Spectral
from ltr.learners.tsallis_inf import TsallisINF
from ltr.solvers.sor import precalc_lower_diag, sor
from ltr.solvers.ssor_pcg import ssor_pcg
from ltr.utils.random import truncated_normal

import datetime


def _one_trial_worker(
    *,
    T: int,
    omegas: np.ndarray,
    grid: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    A: sp.csr_matrix,
    n: int,
    epsilon: float,
    dist_a: float,
    dist_b: float,
    trial_seed: int,
    trial: int | None = None,
    trials: int | None = None,
    benchmark_solver: bool = False,
    benchmark_sor_detail: bool = False,
    exp3_eta: float = 0.05,
    exp3_gamma: float = 0.1,
    exp3_mu: float = 1e-2,
    exp3_smoothness: float = 1.0,
    solver: str = "sor",
    L: sp.csr_matrix | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float, dict[str, float]]
):
    if trial is not None and trials is not None:
        print(f"trial {trial+1}/{trials}: starting")
    rng = np.random.default_rng(trial_seed)
    tinf = TsallisINF(grid, T=T)
    exp3 = Exp3Spectral(
        grid=grid,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        eta=exp3_eta,
        gamma=exp3_gamma,
        mu=exp3_mu,
        smoothness=exp3_smoothness,
        L=L,
    )
    learners: dict[str, Any] = {
        "tinf": tinf,
        "exp3": exp3,
    }
    omega_costs_local = np.zeros((T, omegas.size))
    tinf_costs_local = np.zeros(T)
    exp3_costs_local = np.zeros(T)
    learner_costs_local: dict[str, np.ndarray] = {
        "tinf": tinf_costs_local,
        "exp3": exp3_costs_local,
    }
    learner_probs_local: dict[str, np.ndarray] = {
        name: np.zeros((T, grid.size), dtype=np.float64) for name in learners
    }

    # Reuse sparsity pattern: At = A + c I only shifts the diagonal (same as per-step eye sum).
    At = A.copy().tocsr()
    base_diag = A.diagonal().copy()

    solver_wall_s = 0.0
    sor_timings: defaultdict[str, float] | None = defaultdict(float) if benchmark_sor_detail else None

    def solve_iters(omega: float, L_csr: sp.csr_matrix | None, D_vec: np.ndarray | None) -> int:
        nonlocal solver_wall_s
        if solver == "ssor_pcg":
            if benchmark_solver:
                t0 = time.perf_counter()
                k = ssor_pcg(At, bt, None, omega, epsilon).iterations
                solver_wall_s += time.perf_counter() - t0
                return k
            return ssor_pcg(At, bt, None, omega, epsilon).iterations
        assert L_csr is not None and D_vec is not None
        if benchmark_solver:
            t0 = time.perf_counter()
            k = sor(
                At,
                bt,
                None,
                omega,
                epsilon,
                lower_triangle=L_csr,
                diagonal=D_vec,
                timings=sor_timings,
            ).iterations
            solver_wall_s += time.perf_counter() - t0
            return k
        return sor(
            At,
            bt,
            None,
            omega,
            epsilon,
            lower_triangle=L_csr,
            diagonal=D_vec,
        ).iterations

    def _extract_action_probs(learner: Any, n_arms: int) -> np.ndarray | None:
        """Best-effort extraction of the learner's latest arm distribution."""
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

    for t in range(T):
        if trial is not None and trials is not None and (t + 1) % 100 == 0:
            print(f"trial {trial+1}/{trials}: step {t+1}/{T}")
        c = -0.15 + 0.6 * float(rng.beta(dist_a, dist_b))
        At.setdiag(base_diag + float(c))
        bt = truncated_normal(n, rng=rng)
        if solver == "sor":
            L_at, D_at = precalc_lower_diag(At)
        else:
            L_at, D_at = None, None

        for learner_name, learner in learners.items():
            action = learner.predict(rng=rng)
            probs = _extract_action_probs(learner, grid.size)
            if probs is not None:
                learner_probs_local[learner_name][t, :] = probs
            loss = solve_iters(action, L_at, D_at)
            learner_costs_local[learner_name][t] = loss
            learner.update(loss)

        for i, om in enumerate(omegas):
            omega_costs_local[t, i] = solve_iters(float(om), L_at, D_at)

    if benchmark_solver:
        if benchmark_sor_detail:
            assert sor_timings is not None
            return (
                omega_costs_local,
                tinf_costs_local,
                exp3_costs_local,
                learner_probs_local,
                solver_wall_s,
                dict(sor_timings),
            )
        return omega_costs_local, tinf_costs_local, exp3_costs_local, learner_probs_local, solver_wall_s
    return omega_costs_local, tinf_costs_local, exp3_costs_local, learner_probs_local


def ensure_plots_dir() -> Path:
    p = Path("plots")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _print_sor_breakdown(label: str, merged: dict[str, float]) -> None:
    total = sum(merged.values())
    if total <= 0:
        print(f"[benchmark] {label} sor_detail: no timed samples")
        return
    parts = [f"{k}={merged[k]:.3f}s ({100.0 * merged[k] / total:.1f}%)" for k in sorted(merged)]
    print(f"[benchmark] {label} sor_detail sum over trials (~CPU·s): " + " | ".join(parts))


def _plot_action_probability_heatmap(prob_matrix: np.ndarray, out_path: Path, *, title: str) -> None:
    """Plot arm-choice probability heatmap with y=arm and x=time."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = np.asarray(prob_matrix, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"expected 2D matrix (T, arms), got shape {p.shape}")

    T, n_arms = p.shape
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        p.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=max(1e-12, float(np.max(p))),
    )
    ax.set_xlabel("time step", fontsize=12)
    ax.set_ylabel("arm index", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, max(0, T - 1))
    ax.set_ylim(0, max(0, n_arms - 1))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("selection probability", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def _run_trials(
    *,
    T: int,
    trials: int,
    jobs: int,
    omegas: np.ndarray,
    grid: np.ndarray,
    U: np.ndarray,
    lam: np.ndarray,
    A: sp.csr_matrix,
    n: int,
    epsilon: float,
    dist_a: float,
    dist_b: float,
    seeds: list[int],
    benchmark_solver: bool,
    benchmark_sor_detail: bool,
    cfg: LearningExperimentConfig,
    L: sp.csr_matrix,
) -> list[
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float, dict[str, float]]
]:
    """Run all trials without subprocesses.

    jobs == 1 runs serially in the current process.
    jobs <= 0 uses a thread pool sized to available CPUs.
    jobs > 1 runs trial workers on a thread pool with that many workers.
    """
    if jobs <= 0:
        max_workers = max(1, os.cpu_count() or 1)
    else:
        max_workers = max(1, jobs)
    worker_kwargs = dict(
        T=T,
        omegas=omegas,
        grid=grid,
        eigenvectors=U,
        eigenvalues=lam,
        A=A,
        n=n,
        epsilon=epsilon,
        dist_a=dist_a,
        dist_b=dist_b,
        benchmark_solver=benchmark_solver,
        benchmark_sor_detail=benchmark_sor_detail,
        exp3_eta=cfg.exp3_eta,
        exp3_gamma=cfg.exp3_gamma,
        exp3_mu=cfg.exp3_mu,
        exp3_smoothness=cfg.exp3_smoothness,
        solver=cfg.solver,
        L=L,
    )

    results: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]
        | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float]
        | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float, dict[str, float]]
    ] = [None] * trials
    if max_workers == 1:
        for trial in range(trials):
            results[trial] = _one_trial_worker(
                trial_seed=seeds[trial],
                trial=trial,
                trials=trials,
                **worker_kwargs,
            )
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_trial = {
            ex.submit(
                _one_trial_worker,
                trial_seed=seeds[trial],
                trial=trial,
                trials=trials,
                **worker_kwargs,
            ): trial
            for trial in range(trials)
        }
        for fut in as_completed(fut_to_trial):
            trial = fut_to_trial[fut]
            results[trial] = fut.result()
    return results


def run(
    cfg: LearningExperimentConfig,
    *,
    benchmark_solver: bool,
    benchmark_sor_detail: bool,
) -> None:
    dom = delsq_numgrid("S", cfg.domain_s)
    A = dom.A
    n = A.shape[0]

    omegas = np.linspace(cfg.omega_start, cfg.omega_end, cfg.omega_count)
    grid = np.linspace(cfg.grid_start, cfg.grid_end, cfg.grid_points)
    U, lam, L = similarity_spectrum(grid.size, cfg.similarity_kind)

    T, trials, seed, jobs = cfg.T, cfg.trials, cfg.seed, cfg.jobs
    epsilon = cfg.epsilon

    print(
        f"[experiment] similarity_kind={cfg.similarity_kind!r} "
        f"solver={cfg.solver!r} arms={grid.size} T={T} trials={trials} jobs={jobs} "
        f"plot_prefix={cfg.plot_prefix()!r}",
        flush=True,
    )

    omega_costs = np.zeros((T, trials, omegas.size))
    tinf_costs = np.zeros((T, trials))
    exp3_costs = np.zeros((T, trials))
    learner_prob_histories: dict[str, np.ndarray] = {
        "tinf": np.zeros((T, trials, grid.size), dtype=np.float64),
        "exp3": np.zeros((T, trials, grid.size), dtype=np.float64),
    }

    rng_master = np.random.default_rng(seed)

    now = datetime.datetime.now()
    prefix = cfg.plot_prefix()
    # # High-variance (disabled)
    # filename = f"{prefix}{now.strftime('%Y%m%d_%H%M')}_trials{trials}_learning_high_variance.png"
    # seeds = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(trials)]
    # t_block = time.perf_counter()
    # solver_sum_block = 0.0
    # sor_detail_sum: defaultdict[str, float] = defaultdict(float)
    # with ProcessPoolExecutor(max_workers=(None if jobs <= 0 else jobs)) as ex:
    #     futures = [
    #         ex.submit(
    #             _one_trial_worker,
    #             T=T,
    #             omegas=omegas,
    #             grid=grid,
    #             eigenvectors=U,
    #             eigenvalues=lam,
    #             A=A,
    #             n=n,
    #             epsilon=epsilon,
    #             dist_a=cfg.high_var_dist_a,
    #             dist_b=cfg.high_var_dist_b,
    #             trial_seed=seeds[trial],
    #             trial=trial,
    #             trials=trials,
    #             benchmark_solver=benchmark_solver,
    #             benchmark_sor_detail=benchmark_sor_detail,
    #             exp3_eta=cfg.exp3_eta,
    #             exp3_gamma=cfg.exp3_gamma,
    #             exp3_mu=cfg.exp3_mu,
    #             exp3_smoothness=cfg.exp3_smoothness,
    #             solver=cfg.solver,
    #         )
    #         for trial in range(trials)
    #     ]
    #     for trial, fut in enumerate(futures):
    #         out = fut.result()
    #         if benchmark_sor_detail:
    #             oc, tc, ec, solver_sec, bd = out
    #             solver_sum_block += solver_sec
    #             for key, val in bd.items():
    #                 sor_detail_sum[key] += val
    #         elif benchmark_solver:
    #             oc, tc, ec, solver_sec = out
    #             solver_sum_block += solver_sec
    #         else:
    #             oc, tc, ec = out
    #         omega_costs[:, trial, :] = oc
    #         tinf_costs[:, trial] = tc
    #         exp3_costs[:, trial] = ec
    # if benchmark_solver:
    #     wall = time.perf_counter() - t_block
    #     calls_per_trial = T * (2 + omegas.size)
    #     print(
    #         f"[benchmark] high_variance: parallel block wall {wall:.3f}s | "
    #         f"sum solver time over trials {solver_sum_block:.3f}s "
    #         f"(solver={cfg.solver!r}; mean {solver_sum_block / trials:.3f}s/trial | "
    #         f"{calls_per_trial} solves/trial)"
    #     )
    # if benchmark_sor_detail:
    #     _print_sor_breakdown("high_variance", dict(sor_detail_sum))
    #
    # plots = ensure_plots_dir()
    # fig, ax = plt.subplots(figsize=(7, 5))
    # for i, om in enumerate(omegas):
    #     ax.plot(np.mean(np.cumsum(omega_costs[:, :, i], axis=0), axis=1), T - np.arange(1, T + 1), lw=2, ls="--")
    # ax.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2, color="black")
    # ax.plot(np.mean(np.cumsum(exp3_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2)
    # ax.set_xlabel("total solver iterations", fontsize=14)
    # ax.set_ylabel("instances remaining", fontsize=14)
    # ax.legend([f"$\\omega={om:.1f}$" for om in omegas] + ["Tsallis-INF", "Exp3-Spectral"], fontsize=12)
    # fig.tight_layout()
    # fig.savefig(plots / filename, dpi=256)
    # plt.close(fig)

    # Low-variance
    omega_costs[:] = 0
    tinf_costs[:] = 0
    exp3_costs[:] = 0
    seeds = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(trials)]
    t_block = time.perf_counter()
    solver_sum_block = 0.0
    sor_detail_sum = defaultdict(float)
    if jobs <= 0:
        print(
            "[experiment] jobs<=0: running in-process thread pool "
            f"with {max(1, os.cpu_count() or 1)} workers (no subprocesses)",
            flush=True,
        )
    out_by_trial = _run_trials(
        T=T,
        trials=trials,
        jobs=jobs,
        omegas=omegas,
        grid=grid,
        U=U,
        lam=lam,
        A=A,
        n=n,
        epsilon=epsilon,
        dist_a=cfg.low_var_dist_a,
        dist_b=cfg.low_var_dist_b,
        seeds=seeds,
        benchmark_solver=benchmark_solver,
        benchmark_sor_detail=benchmark_sor_detail,
        cfg=cfg,
        L=L,
    )
    for trial, out in enumerate(out_by_trial):
        if benchmark_sor_detail:
            oc, tc, ec, learner_probs, solver_sec, bd = out
            solver_sum_block += solver_sec
            for key, val in bd.items():
                sor_detail_sum[key] += val
        elif benchmark_solver:
            oc, tc, ec, learner_probs, solver_sec = out
            solver_sum_block += solver_sec
        else:
            oc, tc, ec, learner_probs = out
        omega_costs[:, trial, :] = oc
        tinf_costs[:, trial] = tc
        exp3_costs[:, trial] = ec
        for name, arr in learner_probs.items():
            if name not in learner_prob_histories:
                learner_prob_histories[name] = np.zeros((T, trials, grid.size), dtype=np.float64)
            learner_prob_histories[name][:, trial, :] = arr
    if benchmark_solver:
        wall = time.perf_counter() - t_block
        calls_per_trial = T * (2 + omegas.size)
        print(
            f"[benchmark] low_variance: trial block wall {wall:.3f}s | "
            f"sum solver time over trials {solver_sum_block:.3f}s "
            f"(solver={cfg.solver!r}; mean {solver_sum_block / trials:.3f}s/trial | "
            f"{calls_per_trial} solves/trial)"
        )
    if benchmark_sor_detail:
        _print_sor_breakdown("low_variance", dict(sor_detail_sum))

    plots = ensure_plots_dir()
    filename = f"{prefix}{now.strftime('%Y%m%d_%H%M')}_trials{trials}_learning_low_variance.png"
    losses_path = plots / Path(filename).with_suffix(".npz")
    np.savez_compressed(
        losses_path,
        tinf_costs=tinf_costs.astype(np.int64, copy=False),
        exp3_costs=exp3_costs.astype(np.int64, copy=False),
        omega_costs=omega_costs.astype(np.int64, copy=False),
        tinf_action_probs=learner_prob_histories["tinf"].astype(np.float32, copy=False),
        exp3_action_probs=learner_prob_histories["exp3"].astype(np.float32, copy=False),
        omegas=np.asarray(omegas, dtype=np.float64),
        T=np.int32(T),
        trials=np.int32(trials),
        seed=np.int32(seed),
        trial_seeds=np.asarray(seeds, dtype=np.int64),
        variance_regime=np.array("low_variance"),
    )
    print(
        f"[experiment] saved per-step iteration counts (Tsallis-INF, Exp3, fixed omegas) to {losses_path!s}",
        flush=True,
    )

    for learner_name, probs in learner_prob_histories.items():
        # Aggregate across trials for a single interpretable heatmap per learner.
        mean_probs = np.mean(probs, axis=1)
        heatmap_path = plots / f"{Path(filename).stem}_{learner_name}_action_probs.png"
        _plot_action_probability_heatmap(
            mean_probs,
            heatmap_path,
            title=f"{learner_name}: arm selection probability over time",
        )
        print(f"[experiment] wrote action-probability heatmap to {heatmap_path!s}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, om in enumerate(omegas):
        ax.plot(np.mean(np.cumsum(omega_costs[:, :, i], axis=0), axis=1), T - np.arange(1, T + 1), lw=2, ls="--")
    ax.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2, color="black")
    ax.plot(np.mean(np.cumsum(exp3_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2)
    ax.set_xlabel("total solver iterations", fontsize=14)
    ax.set_ylabel("instances remaining", fontsize=14)
    ax.legend([f"$\\omega={om:.1f}$" for om in omegas] + ["Tsallis-INF", "Exp3-Spectral"], fontsize=12)
    fig.tight_layout()
    fig.savefig(plots / filename, dpi=256)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Learning experiment driver. Use --config for JSON presets; CLI flags override "
            "loaded values when passed."
        ),
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON experiment config (see configs/bench/).",
    )
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of trial workers (1 runs serially; <=0 uses CPU-count threads; >1 uses that many threads).",
    )
    p.add_argument(
        "--similarity-kind",
        type=str,
        default=None,
        help=(
            "Graph Laplacian over arms for Exp3-Spectral (overrides config file); "
            "see ltr.bench.similarity.SIMILARITY_KINDS."
        ),
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Prefix for plot filenames (overrides config file).",
    )
    p.add_argument(
        "--solver",
        type=str,
        choices=("sor", "ssor_pcg"),
        default=None,
        help='Linear solver for feedback (default from config JSON or "sor").',
    )
    p.add_argument(
        "--benchmark-solver",
        action="store_true",
        help=(
            "Print rough solver timings: wall time for each parallel trial block vs "
            "sum of per-trial time inside the chosen solver (useful with small T/trials)."
        ),
    )
    p.add_argument(
        "--benchmark-sor-detail",
        action="store_true",
        help=(
            "SOR only: break down time inside sor() (build M, triangular solve, "
            "matvec residual, norms); implies --benchmark-solver."
        ),
    )
    args = p.parse_args()
    benchmark_sor_detail = bool(args.benchmark_sor_detail)
    benchmark_solver = bool(args.benchmark_solver) or benchmark_sor_detail

    cfg = (
        LearningExperimentConfig.from_json_file(args.config)
        if args.config
        else LearningExperimentConfig()
    )
    if args.T is not None:
        cfg.T = args.T
    if args.trials is not None:
        cfg.trials = args.trials
    if args.seed is not None:
        cfg.seed = args.seed
    if args.jobs is not None:
        cfg.jobs = args.jobs
    if args.similarity_kind is not None:
        cfg.similarity_kind = args.similarity_kind
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.solver is not None:
        cfg.solver = args.solver

    if benchmark_sor_detail and cfg.solver != "sor":
        p.error("--benchmark-sor-detail applies only with solver sor")

    run(
        cfg,
        benchmark_solver=benchmark_solver,
        benchmark_sor_detail=benchmark_sor_detail,
    )


if __name__ == "__main__":
    main()
 
