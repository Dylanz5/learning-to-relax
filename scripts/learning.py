from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.stats import beta as beta_dist

from ltr.bench.config import LearningExperimentConfig
from ltr.bench.similarity import similarity_spectrum
from ltr.domains import delsq_numgrid
from ltr.learners.exp3_spectral import Exp3Spectral
from ltr.learners.tsallis_inf import TsallisINF
from ltr.solvers.sor import precalc_lower_diag, sor
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
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, float]
    | tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]
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
    )
    omega_costs_local = np.zeros((T, omegas.size))
    tinf_costs_local = np.zeros(T)
    exp3_costs_local = np.zeros(T)

    # Reuse sparsity pattern: At = A + c I only shifts the diagonal (same as per-step eye sum).
    At = A.copy().tocsr()
    base_diag = A.diagonal().copy()

    sor_wall_s = 0.0
    sor_timings: defaultdict[str, float] | None = defaultdict(float) if benchmark_sor_detail else None

    def sor_iters(omega: float, L_csr: sp.csr_matrix, D_vec: np.ndarray) -> int:
        nonlocal sor_wall_s
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
            sor_wall_s += time.perf_counter() - t0
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

    for t in range(T):
        if trial is not None and trials is not None and (t + 1) % 100 == 0:
            print(f"trial {trial+1}/{trials}: step {t+1}/{T}")
        c = -0.15 + 0.6 * beta_dist.rvs(dist_a, dist_b, random_state=rng)
        At.setdiag(base_diag + float(c))
        bt = truncated_normal(n, rng=rng)
        L_at, D_at = precalc_lower_diag(At)

        w = tinf.predict(rng=rng)
        tinf_costs_local[t] = sor_iters(w, L_at, D_at)
        tinf.update(tinf_costs_local[t])

        w2 = exp3.predict(rng=rng)
        exp3_costs_local[t] = sor_iters(w2, L_at, D_at)
        exp3.update(exp3_costs_local[t])

        for i, om in enumerate(omegas):
            omega_costs_local[t, i] = sor_iters(float(om), L_at, D_at)

    if benchmark_solver:
        if benchmark_sor_detail:
            assert sor_timings is not None
            return (
                omega_costs_local,
                tinf_costs_local,
                exp3_costs_local,
                sor_wall_s,
                dict(sor_timings),
            )
        return omega_costs_local, tinf_costs_local, exp3_costs_local, sor_wall_s
    return omega_costs_local, tinf_costs_local, exp3_costs_local


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
    U, lam = similarity_spectrum(grid.size, cfg.similarity_kind)

    T, trials, seed, jobs = cfg.T, cfg.trials, cfg.seed, cfg.jobs
    epsilon = cfg.epsilon

    omega_costs = np.zeros((T, trials, omegas.size))
    tinf_costs = np.zeros((T, trials))
    exp3_costs = np.zeros((T, trials))

    rng_master = np.random.default_rng(seed)

    now = datetime.datetime.now()
    prefix = cfg.plot_prefix()
    filename = f"{prefix}{now.strftime('%Y%m%d_%H%M')}_trials{trials}_learning_high_variance.png"
    # High-variance
    seeds = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(trials)]
    t_block = time.perf_counter()
    sor_sum_block = 0.0
    sor_detail_sum: defaultdict[str, float] = defaultdict(float)
    with ProcessPoolExecutor(max_workers=(None if jobs <= 0 else jobs)) as ex:
        futures = [
            ex.submit(
                _one_trial_worker,
                T=T,
                omegas=omegas,
                grid=grid,
                eigenvectors=U,
                eigenvalues=lam,
                A=A,
                n=n,
                epsilon=epsilon,
                dist_a=cfg.high_var_dist_a,
                dist_b=cfg.high_var_dist_b,
                trial_seed=seeds[trial],
                trial=trial,
                trials=trials,
                benchmark_solver=benchmark_solver,
                benchmark_sor_detail=benchmark_sor_detail,
                exp3_eta=cfg.exp3_eta,
                exp3_gamma=cfg.exp3_gamma,
                exp3_mu=cfg.exp3_mu,
                exp3_smoothness=cfg.exp3_smoothness,
            )
            for trial in range(trials)
        ]
        for trial, fut in enumerate(futures):
            out = fut.result()
            if benchmark_sor_detail:
                oc, tc, ec, sor_sec, bd = out
                sor_sum_block += sor_sec
                for key, val in bd.items():
                    sor_detail_sum[key] += val
            elif benchmark_solver:
                oc, tc, ec, sor_sec = out
                sor_sum_block += sor_sec
            else:
                oc, tc, ec = out
            omega_costs[:, trial, :] = oc
            tinf_costs[:, trial] = tc
            exp3_costs[:, trial] = ec
    if benchmark_solver:
        wall = time.perf_counter() - t_block
        calls_per_trial = T * (2 + omegas.size)
        print(
            f"[benchmark] high_variance: parallel block wall {wall:.3f}s | "
            f"sum SOR time over trials {sor_sum_block:.3f}s "
            f"(~CPU·s in sor; mean {sor_sum_block / trials:.3f}s/trial | "
            f"{calls_per_trial} sor calls/trial)"
        )
    if benchmark_sor_detail:
        _print_sor_breakdown("high_variance", dict(sor_detail_sum))

    plots = ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, om in enumerate(omegas):
        ax.plot(np.mean(np.cumsum(omega_costs[:, :, i], axis=0), axis=1), T - np.arange(1, T + 1), lw=2, ls="--")
    ax.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2, color="black")
    ax.plot(np.mean(np.cumsum(exp3_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2)
    ax.set_xlabel("total iterations", fontsize=14)
    ax.set_ylabel("instances remaining", fontsize=14)
    ax.legend([f"$\\omega={om:.1f}$" for om in omegas] + ["Tsallis-INF", "Exp3-Spectral"], fontsize=12)
    fig.tight_layout()
    fig.savefig(plots / filename, dpi=256)
    plt.close(fig)

    # Low-variance
    omega_costs[:] = 0
    tinf_costs[:] = 0
    exp3_costs[:] = 0
    seeds = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(trials)]
    t_block = time.perf_counter()
    sor_sum_block = 0.0
    sor_detail_sum = defaultdict(float)
    with ProcessPoolExecutor(max_workers=(None if jobs <= 0 else jobs)) as ex:
        futures = [
            ex.submit(
                _one_trial_worker,
                T=T,
                omegas=omegas,
                grid=grid,
                eigenvectors=U,
                eigenvalues=lam,
                A=A,
                n=n,
                epsilon=epsilon,
                dist_a=cfg.low_var_dist_a,
                dist_b=cfg.low_var_dist_b,
                trial_seed=seeds[trial],
                trial=trial,
                trials=trials,
                benchmark_solver=benchmark_solver,
                benchmark_sor_detail=benchmark_sor_detail,
                exp3_eta=cfg.exp3_eta,
                exp3_gamma=cfg.exp3_gamma,
                exp3_mu=cfg.exp3_mu,
                exp3_smoothness=cfg.exp3_smoothness,
            )
            for trial in range(trials)
        ]
        for trial, fut in enumerate(futures):
            out = fut.result()
            if benchmark_sor_detail:
                oc, tc, ec, sor_sec, bd = out
                sor_sum_block += sor_sec
                for key, val in bd.items():
                    sor_detail_sum[key] += val
            elif benchmark_solver:
                oc, tc, ec, sor_sec = out
                sor_sum_block += sor_sec
            else:
                oc, tc, ec = out
            omega_costs[:, trial, :] = oc
            tinf_costs[:, trial] = tc
            exp3_costs[:, trial] = ec
    if benchmark_solver:
        wall = time.perf_counter() - t_block
        calls_per_trial = T * (2 + omegas.size)
        print(
            f"[benchmark] low_variance: parallel block wall {wall:.3f}s | "
            f"sum SOR time over trials {sor_sum_block:.3f}s "
            f"(~CPU·s in sor; mean {sor_sum_block / trials:.3f}s/trial | "
            f"{calls_per_trial} sor calls/trial)"
        )
    if benchmark_sor_detail:
        _print_sor_breakdown("low_variance", dict(sor_detail_sum))



    filename = f"{prefix}{now.strftime('%Y%m%d_%H%M')}_trials{trials}_learning_low_variance.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, om in enumerate(omegas):
        ax.plot(np.mean(np.cumsum(omega_costs[:, :, i], axis=0), axis=1), T - np.arange(1, T + 1), lw=2, ls="--")
    ax.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2, color="black")
    ax.plot(np.mean(np.cumsum(exp3_costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2)
    ax.set_xlabel("total iterations", fontsize=14)
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
        help="Number of parallel trial workers (<=0 uses all cores).",
    )
    p.add_argument(
        "--similarity-kind",
        type=str,
        default=None,
        choices=["chain", "ring"],
        help="Graph Laplacian over arms for Exp3-Spectral (overrides config file).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Prefix for plot filenames (overrides config file).",
    )
    p.add_argument(
        "--benchmark-solver",
        action="store_true",
        help=(
            "Print rough SOR timings: wall time for each parallel trial block vs "
            "sum of per-trial time inside sor() (useful with small T/trials)."
        ),
    )
    p.add_argument(
        "--benchmark-sor-detail",
        action="store_true",
        help=(
            "Break down time inside sor() (build M, triangular solve, matvec residual, norms); "
            "implies --benchmark-solver."
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

    run(
        cfg,
        benchmark_solver=benchmark_solver,
        benchmark_sor_detail=benchmark_sor_detail,
    )


if __name__ == "__main__":
    main()
 
