from __future__ import annotations

import numpy as np

from ltr.graphs import chain_laplacian
from ltr.learners import Exp3Spectral, TsallisINF, TsallisSpectral


def test_exp3_spectral_runs() -> None:
    rng = np.random.default_rng(0)
    grid = np.linspace(1.0, 1.95, 7)
    L = chain_laplacian(grid.size)
    lam, U = np.linalg.eigh(L)
    alg = Exp3Spectral(
        grid=grid, eigenvectors=U, eigenvalues=lam, eta=0.05, gamma=0.2, mu=1e-2, smoothness=1.0, L=L
    )
    for _ in range(5):
        _ = alg.predict(rng=rng)
        alg.update(float(rng.integers(1, 100)))


def test_tsallis_spectral_runs() -> None:
    rng = np.random.default_rng(0)
    grid = np.linspace(1.0, 1.95, 7)
    L = chain_laplacian(grid.size)
    lam, U = np.linalg.eigh(L)
    alg = TsallisSpectral(
        grid=grid, eigenvectors=U, eigenvalues=lam, eta=0.05, gamma=0.2, mu=1e-2, smoothness=1.0, alpha=0.5, L=L
    )
    for _ in range(5):
        action = alg.predict(rng=rng)
        assert grid.min() <= action <= grid.max()
        alg.update(float(rng.integers(1, 100)))


def test_tsallis_spectral_q_solver_is_valid_distribution() -> None:
    grid = np.linspace(1.0, 1.95, 9)
    L = chain_laplacian(grid.size)
    lam, U = np.linalg.eigh(L)
    rng = np.random.default_rng(1)
    for alpha in (0.25, 0.5, 0.75, 0.9):
        alg = TsallisSpectral(
            grid=grid, eigenvectors=U, eigenvalues=lam, eta=0.1, gamma=0.1, mu=1e-2, alpha=alpha, L=L
        )
        F = rng.normal(size=grid.size)
        q = alg._solve_q(F, eta=alg.eta)
        assert np.all(q >= 0.0)
        assert np.all(np.isfinite(q))
        assert abs(float(np.sum(q)) - 1.0) < 1e-6
        assert int(np.argmax(q)) == int(np.argmin(F))


def test_tsallis_inf_runs() -> None:
    rng = np.random.default_rng(0)
    alg = TsallisINF(np.linspace(1.0, 1.95, 7), T=10)
    for _ in range(5):
        _ = alg.predict(rng=rng)
        alg.update(float(rng.integers(1, 100)))


def test_tsallis_inf_regression_action_sequence() -> None:
    rng = np.random.default_rng(0)
    alg = TsallisINF(np.linspace(1.0, 1.95, 7), T=0)
    actions: list[int] = []
    for _ in range(20):
        _ = alg.predict(rng=rng)
        assert alg.index is not None
        actions.append(int(alg.index))
        alg.update(10.0)
    assert actions == [4, 1, 0, 2, 6, 5, 3, 3, 5, 6, 3, 0, 3, 2, 5, 3, 5, 3, 1, 5]
