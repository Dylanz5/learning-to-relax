from __future__ import annotations

import time
from collections.abc import MutableMapping
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class SorResult:
    iterations: int
    x: np.ndarray


def precalc_lower_diag(A: sp.spmatrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Factors shared across SOR calls with the same ``A`` but different ``omega``.

    MATLAB uses ``M = D/omega + tril(A,-1)``. For fixed ``A``, only ``D/omega``
    depends on ``omega``; the strictly lower triangle is reused.
    """
    Acsr = A.tocsr()
    L = sp.tril(Acsr, k=-1).tocsr()
    D = Acsr.diagonal().copy()
    return L, D


def sor(
    A: sp.spmatrix,
    b: np.ndarray,
    x0: np.ndarray | None,
    omega: float,
    tol: float,
    maxiter: int = 10_000,
    *,
    lower_triangle: sp.csr_matrix | None = None,
    diagonal: np.ndarray | None = None,
    timings: MutableMapping[str, float] | None = None,
) -> SorResult:
    """Successive over-relaxation iteration count to reach relative residual tol.

    Port of MATLAB `solvers/sor.m`. Uses the same update:

        M = D/omega + tril(A, -1)
        x <- x + M \\ (b - A x)
    """

    if omega <= 0:
        raise ValueError("omega must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    A = A.tocsr()
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.shape[0]
    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()

    if x.shape[0] != n:
        raise ValueError("x0 has incompatible shape")
    if (lower_triangle is None) ^ (diagonal is None):
        raise ValueError("pass both lower_triangle and diagonal, or neither")

    if timings is not None:
        t_b = time.perf_counter()
    if lower_triangle is not None and diagonal is not None:
        Dvec = np.asarray(diagonal, dtype=float).reshape(-1)
        if Dvec.shape[0] != n:
            raise ValueError("diagonal length must match A.shape[0]")
        L = sp.csr_matrix(lower_triangle)
    else:
        Dvec = A.diagonal()
        L = sp.tril(A, k=-1).tocsr()

    # Build diagonal as explicit csr_matrix to avoid sparse-array/matrix mixing.
    idx = np.arange(n, dtype=np.int64)
    Dmat = sp.csr_matrix((Dvec / omega, (idx, idx)), shape=(n, n))
    M = (L + Dmat).tocsr()
    if timings is not None:
        timings["build_m"] = timings.get("build_m", 0.0) + time.perf_counter() - t_b

    if timings is not None:
        t_r = time.perf_counter()
    r = b - A @ x
    if timings is not None:
        timings["matvec_residual"] = timings.get("matvec_residual", 0.0) + time.perf_counter() - t_r
    if timings is not None:
        t_n = time.perf_counter()
    norm0 = np.linalg.norm(r)
    if timings is not None:
        timings["norm"] = timings.get("norm", 0.0) + time.perf_counter() - t_n
    if norm0 == 0:
        return SorResult(iterations=0, x=x)

    for k in range(1, maxiter + 1):
        if timings is not None:
            t_s = time.perf_counter()
        try:
            dx = spla.spsolve_triangular(M, r, lower=True)
        except Exception:
            # Fallback for intermittent SciPy sparse-internal failures.
            dx = spla.spsolve(M, r)
        if timings is not None:
            timings["triangular_solve"] = timings.get("triangular_solve", 0.0) + time.perf_counter() - t_s
        x = x + dx
        if timings is not None:
            t_r = time.perf_counter()
        r = b - A @ x
        if timings is not None:
            timings["matvec_residual"] = timings.get("matvec_residual", 0.0) + time.perf_counter() - t_r
        if timings is not None:
            t_n = time.perf_counter()
        nr = np.linalg.norm(r)
        if timings is not None:
            timings["norm"] = timings.get("norm", 0.0) + time.perf_counter() - t_n
        if nr / norm0 < tol:
            return SorResult(iterations=k, x=x)

    return SorResult(iterations=maxiter, x=x)
 
