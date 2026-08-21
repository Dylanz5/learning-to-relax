from __future__ import annotations

import argparse
import csv
import os
import re
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
from ltr.bench.prob_plots import (
    extract_action_probs,
    plot_action_probability_heatmap,
    plot_loss_estimate_heatmap,
)
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
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], float]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], float, dict[str, float]]
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
    # Per-arm normalized cumulative loss L_i = k_i/scale for learners that expose
    # it (Tsallis-INF). Piggybacked on the same history dict under a distinct key
    # so it flows through the existing capture/plotting path.
    learner_loss_est_keys: dict[str, str] = {}
    for _name, _learner in learners.items():
        if callable(getattr(_learner, "loss_estimates", None)):
            _key = f"{_name}_loss_estimate"
            learner_probs_local[_key] = np.zeros((T, grid.size), dtype=np.float64)
            learner_loss_est_keys[_name] = _key

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
            probs = extract_action_probs(learner, grid.size)
            if probs is not None:
                learner_probs_local[learner_name][t, :] = probs
            le_key = learner_loss_est_keys.get(learner_name)
            if le_key is not None:
                try:
                    l_vals = np.asarray(learner.loss_estimates(), dtype=float).reshape(-1)
                    if l_vals.shape == (grid.size,) and np.all(np.isfinite(l_vals)):
                        learner_probs_local[le_key][t, :] = l_vals
                except Exception:
                    pass
            loss = solve_iters(action, L_at, D_at)
            learner_costs_local[learner_name][t] = loss
            learner.update(loss)

        for i, om in enumerate(omegas):
            omega_costs_local[t, i] = solve_iters(float(om), L_at, D_at)

    # Tsallis-INF diagnostics (logging only). Emitted alongside the usual
    # per-step arrays so the driver can archive them without altering the run.
    tinf_diag_local: dict[str, Any] = {
        "diag": list(getattr(tinf, "diag", [])),
        "k_hist": (
            np.asarray(tinf.k_hist, dtype=np.float64)
            if getattr(tinf, "k_hist", None)
            else np.zeros((0, grid.size), dtype=np.float64)
        ),
        "p_hist": (
            np.asarray(tinf.p_hist, dtype=np.float64)
            if getattr(tinf, "p_hist", None)
            else np.zeros((0, grid.size), dtype=np.float64)
        ),
    }

    if benchmark_solver:
        if benchmark_sor_detail:
            assert sor_timings is not None
            return (
                omega_costs_local,
                tinf_costs_local,
                exp3_costs_local,
                learner_probs_local,
                tinf_diag_local,
                solver_wall_s,
                dict(sor_timings),
            )
        return (
            omega_costs_local,
            tinf_costs_local,
            exp3_costs_local,
            learner_probs_local,
            tinf_diag_local,
            solver_wall_s,
        )
    return omega_costs_local, tinf_costs_local, exp3_costs_local, learner_probs_local, tinf_diag_local


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
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], float]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], float, dict[str, float]]
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
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]
        | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], float]
        | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], float, dict[str, float]]
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


_DIAG_FIELDS: list[tuple[str, str]] = [
    ("trial", "i8"),
    ("t", "i8"),
    ("arm", "i8"),
    ("p_norm", "f8"),
    ("prob_raw", "f8"),
    ("loss", "f8"),
    ("run_mean", "f8"),
    ("increment", "f8"),
    ("k_arm_before", "f8"),
    ("k_arm_after", "f8"),
    ("scale_before", "f8"),
    ("scale_after", "f8"),
    ("x", "f8"),
    ("L_min", "f8"),
    ("argmax_arm", "i8"),
    ("p_max", "f8"),
    ("entropy", "f8"),
]


def _build_diag_table(tinf_diags: list[dict[str, Any]]) -> np.ndarray:
    """Flatten per-trial Tsallis-INF diag rows into one structured array.

    A leading ``trial`` column disambiguates rows when ``trials > 1``.
    """
    dtype = np.dtype(_DIAG_FIELDS)
    rows: list[tuple] = []
    for trial, d in enumerate(tinf_diags):
        if not d:
            continue
        for r in d.get("diag", []):
            rows.append(tuple(trial if name == "trial" else r[name] for name, _ in _DIAG_FIELDS))
    return np.array(rows, dtype=dtype)


def _stack_hist(tinf_diags: list[dict[str, Any]], key: str, T: int, K: int) -> np.ndarray:
    """Stack per-trial (T, K) history arrays into (trials, T, K)."""
    out = np.zeros((len(tinf_diags), T, K), dtype=np.float64)
    for trial, d in enumerate(tinf_diags):
        if not d:
            continue
        arr = np.asarray(d.get(key), dtype=np.float64)
        if arr.ndim == 2 and arr.size:
            rows = min(T, arr.shape[0])
            cols = min(K, arr.shape[1])
            out[trial, :rows, :cols] = arr[:rows, :cols]
    return out


def _write_diag_csv(diag_table: np.ndarray, csv_path: Path) -> None:
    """Write the scalar diag table to CSV (same stem as the run's .npz)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [name for name, _ in _DIAG_FIELDS]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(field_names)
        for row in diag_table:
            writer.writerow([row[name] for name in field_names])


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
    # filename = f"{prefix}{cfg.param_tag()}_{now.strftime('%Y%m%d_%H%M')}_trials{trials}_hv.png"
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
    tinf_diags: list[dict[str, Any]] = [None] * trials  # type: ignore[list-item]
    for trial, out in enumerate(out_by_trial):
        if benchmark_sor_detail:
            oc, tc, ec, learner_probs, tinf_diag, solver_sec, bd = out
            solver_sum_block += solver_sec
            for key, val in bd.items():
                sor_detail_sum[key] += val
        elif benchmark_solver:
            oc, tc, ec, learner_probs, tinf_diag, solver_sec = out
            solver_sum_block += solver_sec
        else:
            oc, tc, ec, learner_probs, tinf_diag = out
        tinf_diags[trial] = tinf_diag
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
    # Timestamp first so runs sort chronologically by name.
    filename = f"{now.strftime('%Y%m%d_%H%M')}_{prefix}{cfg.param_tag()}_trials{trials}_lv.png"
    losses_path = plots / Path(filename).with_suffix(".npz")

    # Tsallis-INF diagnostics (logging only): flatten the scalar rows into a
    # structured table and stack the per-round k/p vectors as (trials, T, K).
    diag_table = _build_diag_table(tinf_diags)
    tinf_k_hist = _stack_hist(tinf_diags, "k_hist", T, grid.size)
    tinf_p_hist = _stack_hist(tinf_diags, "p_hist", T, grid.size)

    # Graph artifacts for Exp3-Spectral. The graph is a *fixed* input (never
    # updated during a run) and identical across trials: save the Laplacian, the
    # recovered similarity matrix W (L = D - W => W = diag(diag(L)) - L), the
    # eigenvalues, and the D-optimal exploration design q derived from the graph.
    laplacian = np.asarray(L, dtype=np.float64)
    similarity = np.diag(np.diag(laplacian)) - laplacian
    eigenvalues = np.asarray(lam, dtype=np.float64)
    exp3_exploration = np.asarray(
        Exp3Spectral(
            grid=grid,
            eigenvectors=U,
            eigenvalues=lam,
            eta=cfg.exp3_eta,
            gamma=cfg.exp3_gamma,
            mu=cfg.exp3_mu,
            smoothness=cfg.exp3_smoothness,
            L=L,
        ).q,
        dtype=np.float64,
    )

    np.savez_compressed(
        losses_path,
        tinf_costs=tinf_costs.astype(np.int64, copy=False),
        exp3_costs=exp3_costs.astype(np.int64, copy=False),
        omega_costs=omega_costs.astype(np.int64, copy=False),
        tinf_action_probs=learner_prob_histories["tinf"].astype(np.float32, copy=False),
        exp3_action_probs=learner_prob_histories["exp3"].astype(np.float32, copy=False),
        tinf_loss_estimates=learner_prob_histories.get(
            "tinf_loss_estimate", np.zeros((T, trials, grid.size), dtype=np.float32)
        ).astype(np.float32, copy=False),
        tinf_diag=diag_table,
        tinf_k_hist=tinf_k_hist.astype(np.float64, copy=False),
        tinf_p_hist=tinf_p_hist.astype(np.float64, copy=False),
        laplacian=laplacian,
        similarity=similarity,
        eigenvalues=eigenvalues,
        exp3_exploration=exp3_exploration,
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

    diag_csv_path = losses_path.with_suffix(".csv")
    _write_diag_csv(diag_table, diag_csv_path)
    print(
        f"[experiment] saved Tsallis-INF diagnostics ({diag_table.shape[0]} rows) to {diag_csv_path!s}",
        flush=True,
    )

    for learner_name, arr_hist in learner_prob_histories.items():
        # Aggregate across trials for a single interpretable heatmap per learner.
        mean_arr = np.mean(arr_hist, axis=1)
        if learner_name.endswith("_loss_estimate"):
            base = learner_name[: -len("_loss_estimate")]
            heatmap_path = plots / f"{Path(filename).stem}_{base}_loss_estimates.png"
            plot_loss_estimate_heatmap(
                mean_arr,
                heatmap_path,
                title=f"{base}: per-arm normalized cumulative loss $L_i$ over time",
            )
            print(f"[experiment] wrote loss-estimate heatmap to {heatmap_path!s}", flush=True)
        else:
            heatmap_path = plots / f"{Path(filename).stem}_{learner_name}_action_probs.png"
            plot_action_probability_heatmap(
                mean_arr,
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
    # Title = run filename (stem) with the leading date stripped, plus the actual
    # Exp3-Spectral hyperparameters so a plot is self-describing.
    title_base = re.sub(r"^\d{8}_\d{4}_", "", Path(filename).stem)
    hp = (
        f"$\\eta$={cfg.exp3_eta:g}  $\\gamma$={cfg.exp3_gamma:g}  "
        f"$\\mu$={cfg.exp3_mu:g}  smooth={cfg.exp3_smoothness:g}  seed={seed}"
    )
    ax.set_title(f"{title_base}\n{hp}", fontsize=10)
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
 
