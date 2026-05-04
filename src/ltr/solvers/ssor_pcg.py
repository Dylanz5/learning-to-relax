from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class PcqResult:
    iterations: int
    x: np.ndarray
    info: int


def _ssor_preconditioner(A: sp.spmatrix, omega: float) -> spla.LinearOperator:
    """LinearOperator applying the SSOR-preconditioner inverse.

    MATLAB reference (`solvers/ssor_pcg.m`):
      D = diag(diag(A));
      L = tril(A, -1);
      X = D + omega * L;
      pcg(A, b, tol, 10000, X'*inv(D), X/omega/(2-omega), x);

    SciPy's CG takes a single preconditioner M approximating A^{-1}.
    We implement M^{-1} = M2^{-1} M1^{-1} with the same split:
      M1 = X' * inv(D)
      M2 = X / (omega*(2-omega))

    Solve steps avoid explicit inverses using triangular solves + diagonal scaling.
    """

    if omega <= 0 or omega >= 2:
        # SSOR is typically defined for 0 < omega < 2
        raise ValueError("omega must be in (0, 2) for SSOR")

    A = A.tocsr()
    D = A.diagonal().astype(float)
    if np.any(D == 0):
        raise ValueError("A has zero diagonal entries; SSOR undefined")

    L = sp.tril(A, k=-1).tocsr()
    X = (sp.diags(D, format="csr") + omega * L).tocsr()  # lower triangular with diagonal

    scale = omega * (2.0 - omega)

    def apply(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(-1)
        # y = M1^{-1} v
        # M1 = X.T * D^{-1}  (upper triangular)
        # Solve X.T w = v, then y = D * w
        w = spla.spsolve_triangular(X.T, v, lower=False)
        y = D * w
        # z = M2^{-1} y, M2 = X / scale  => z = scale * X^{-1} y
        z = scale * spla.spsolve_triangular(X, y, lower=True)
        return z

    n = A.shape[0]
    return spla.LinearOperator((n, n), matvec=apply, dtype=float)


def ssor_pcg(
    A: sp.spmatrix,
    b: np.ndarray,
    x0: np.ndarray | None,
    omega: float,
    tol: float,
    maxiter: int = 10_000,
) -> PcqResult:
    """SSOR-preconditioned conjugate gradient.

    Port of MATLAB `solvers/ssor_pcg.m` using SciPy's `cg`.
    Returns the iteration count (k) and final iterate.
    """

    A = A.tocsr()
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.shape[0]
    x0v = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)

    M = _ssor_preconditioner(A, omega=omega)
    it = 0

    def cb(_xk: np.ndarray) -> None:
        nonlocal it
        it += 1

    x, info = spla.cg(A, b, x0=x0v, rtol=tol, atol=0.0, maxiter=maxiter, M=M, callback=cb)
    return PcqResult(iterations=it, x=x, info=int(info))
