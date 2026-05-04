from __future__ import annotations

import numpy as np

from ltr.domains import delsq_numgrid
from ltr.solvers.sor import precalc_lower_diag, sor
from ltr.solvers.ssor_pcg import ssor_pcg


def test_sor_converges_on_laplacian() -> None:
     dom = delsq_numgrid("S", 12)
     A = dom.A
     n = A.shape[0]
     b = np.ones(n)
     res = sor(A, b, np.zeros(n), omega=1.5, tol=1e-8, maxiter=10_000)
     assert res.iterations > 0
     r = b - A @ res.x
     assert np.linalg.norm(r) / np.linalg.norm(b) < 1e-6


def test_sor_cached_lower_diag_matches_naive() -> None:
    dom = delsq_numgrid("S", 12)
    A = dom.A.tocsr()
    n = A.shape[0]
    b = np.random.default_rng(0).standard_normal(n)
    omega = 1.35
    tol = 1e-8

    naive = sor(A, b, None, omega=omega, tol=tol, maxiter=10_000)
    L, D = precalc_lower_diag(A)
    cached = sor(
        A,
        b,
        None,
        omega=omega,
        tol=tol,
        maxiter=10_000,
        lower_triangle=L,
        diagonal=D,
    )
    assert naive.iterations == cached.iterations
    np.testing.assert_allclose(naive.x, cached.x, rtol=0.0, atol=1e-12)


def test_ssor_pcg_converges_on_laplacian() -> None:
     dom = delsq_numgrid("S", 12)
     A = dom.A
     n = A.shape[0]
     b = np.ones(n)
     res = ssor_pcg(A, b, np.zeros(n), omega=1.5, tol=1e-8, maxiter=5_000)
     assert res.iterations > 0
     r = b - A @ res.x
     assert np.linalg.norm(r) / np.linalg.norm(b) < 1e-6
 
