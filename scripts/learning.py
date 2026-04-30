from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.stats import beta as beta_dist

from ltr.domains import delsq_numgrid
from ltr.learners.exp3_spectral import Exp3Spectral, chain_laplacian
from ltr.learners.tsallis_inf import TsallisINF
from ltr.solvers.sor import sor
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if trial is not None and trials is not None:
        print(f"trial {trial+1}/{trials}: starting")
    rng = np.random.default_rng(trial_seed)
    tinf = TsallisINF(grid, T=T)
    exp3 = Exp3Spectral(
        grid=grid,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        # conservative defaults; tweak if you want more/less exploration
        eta=0.05,
        gamma=0.1,
        mu=1e-2,
        smoothness=1.0,
    )
    omega_costs_local = np.zeros((T, omegas.size))
    tinf_costs_local = np.zeros(T)
    exp3_costs_local = np.zeros(T)

    for t in range(T):
        if trial is not None and trials is not None and (t + 1) % 100 == 0:
            print(f"trial {trial+1}/{trials}: step {t+1}/{T}")
        c = -0.15 + 0.6 * beta_dist.rvs(dist_a, dist_b, random_state=rng)
        At = A + float(c) * sp.eye(n, format="csr")
        bt = truncated_normal(n, rng=rng)

        w = tinf.predict(rng=rng)
        tinf_costs_local[t] = sor(At, bt, np.zeros(n), w, epsilon).iterations
        tinf.update(tinf_costs_local[t])

        w2 = exp3.predict(rng=rng)
        exp3_costs_local[t] = sor(At, bt, np.zeros(n), w2, epsilon).iterations
        exp3.update(exp3_costs_local[t])

        for i, om in enumerate(omegas):
            omega_costs_local[t, i] = sor(At, bt, np.zeros(n), float(om), epsilon).iterations

    return omega_costs_local, tinf_costs_local, exp3_costs_local


def ensure_plots_dir() -> Path:
    p = Path("plots")
    p.mkdir(parents=True, exist_ok=True)
    return p


def run(T: int, trials: int, seed: int, jobs: int) -> None:
    dom = delsq_numgrid("S", 12)
    A = dom.A
    n = A.shape[0]

    epsilon = 1e-8
    omegas = np.linspace(1.0, 1.8, 5)
    grid = np.linspace(1.0, 1.95, 20)
    L = chain_laplacian(grid.size)
    lam, U = np.linalg.eigh(L)

    omega_costs = np.zeros((T, trials, omegas.size))
    tinf_costs = np.zeros((T, trials))
    exp3_costs = np.zeros((T, trials))

    rng_master = np.random.default_rng(seed)

    now = datetime.datetime.now()
    filename = f"{now.strftime('%Y%m%d_%H%M')}_trials{trials}_learning_high_variance.png"
    # High-variance
    seeds = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(trials)]
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
                dist_a=0.5,
                dist_b=1.5,
                trial_seed=seeds[trial],
                trial=trial,
                trials=trials,
            )
            for trial in range(trials)
        ]
        for trial, fut in enumerate(futures):
            oc, tc, ec = fut.result()
            omega_costs[:, trial, :] = oc
            tinf_costs[:, trial] = tc
            exp3_costs[:, trial] = ec

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
                dist_a=2.0,
                dist_b=6.0,
                trial_seed=seeds[trial],
                trial=trial,
                trials=trials,
            )
            for trial in range(trials)
        ]
        for trial, fut in enumerate(futures):
            oc, tc, ec = fut.result()
            omega_costs[:, trial, :] = oc
            tinf_costs[:, trial] = tc
            exp3_costs[:, trial] = ec



    filename = f"{now.strftime('%Y%m%d_%H%M')}_trials{trials}_learning_low_variance.png"

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
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=5000)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of parallel trial workers (<=0 uses all cores).",
    )
    args = p.parse_args()
    run(T=args.T, trials=args.trials, seed=args.seed, jobs=args.jobs)


if __name__ == "__main__":
    main()
 
