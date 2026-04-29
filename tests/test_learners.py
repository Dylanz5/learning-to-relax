from __future__ import annotations

import numpy as np

from ltr.learners.chebcb import ChebCB
from ltr.learners.exp3_spectral import Exp3Spectral, chain_laplacian
from ltr.learners.tsallis_inf import TsallisINF
from ltr.learners.tsallis_inf_cb import TsallisINFCB


def test_exp3_spectral_runs() -> None:
    rng = np.random.default_rng(0)
    grid = np.linspace(1.0, 1.95, 7)
    L = chain_laplacian(grid.size)
    lam, U = np.linalg.eigh(L)
    alg = Exp3Spectral(grid=grid, eigenvectors=U, eigenvalues=lam, eta=0.05, gamma=0.2, mu=1e-2, smoothness=1.0)
    for _ in range(5):
        _ = alg.predict(rng=rng)
        alg.update(float(rng.integers(1, 100)))


def test_tsallis_inf_runs() -> None:
    rng = np.random.default_rng(0)
    alg = TsallisINF(np.linspace(1.0, 1.95, 7), T=10)
    for _ in range(5):
        _ = alg.predict(rng=rng)
        alg.update(float(rng.integers(1, 100)))
 
 
def test_tsallis_inf_regression_action_sequence() -> None:
    """Coarse regression guard for Tsallis-INF port.

    This is not meant to match MATLAB bit-for-bit, but it prevents accidental
    semantic changes from silently drifting behavior.
    """

    rng = np.random.default_rng(0)
    alg = TsallisINF(np.linspace(1.0, 1.95, 7), T=0)

    actions: list[int] = []
    for _ in range(20):
        _ = alg.predict(rng=rng)
        assert alg.index is not None
        assert np.isfinite(alg.prob) and alg.prob > 0
        actions.append(int(alg.index))
        alg.update(10.0)

    assert actions == [4, 1, 0, 0, 6, 5, 3, 4, 2, 6, 5, 0, 5, 1, 3, 2, 5, 2, 2, 3]


def test_tsallis_inf_cb_runs() -> None:
    rng = np.random.default_rng(0)
    alg = TsallisINFCB(np.linspace(1.0, 1.95, 7), m=3, a=-0.15, b=0.65)
    for _ in range(5):
        _ = alg.predict(float(rng.uniform(-0.15, 0.65)), rng=rng)
        alg.update(float(rng.integers(1, 100)))
 
 
def test_chebcb_runs() -> None:
    rng = np.random.default_rng(0)
    alg = ChebCB(np.linspace(1.0, 1.95, 7), T=20, m=6, a=-0.15, b=0.65)
    for _ in range(10):
        _ = alg.predict(float(rng.uniform(-0.15, 0.65)), rng=rng)
        alg.update(float(rng.integers(1, 100)))
 
