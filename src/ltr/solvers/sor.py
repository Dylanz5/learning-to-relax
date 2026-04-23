from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class SorResult:
    iterations: int
    x: np.ndarray


def sor(
     A: sp.spmatrix,
     b: np.ndarray,
     x0: np.ndarray | None,
     omega: float,
     tol: float,
     maxiter: int = 10_000,
 ) -> SorResult:
     """Successive over-relaxation iteration count to reach relative residual tol.
 
     Port of MATLAB `solvers/sor.m`. Uses the same update:
 
         M = D/omega + tril(A, -1)
         x <- x + M \\ (b - A x)
 
     Parameters
     ----------
     A:
         Sparse matrix (typically SPD).
     b:
         Right-hand side vector.
     x0:
         Initial guess (defaults to zeros).
     omega:
         Relaxation parameter.
     tol:
         Relative residual tolerance (||r||/||r0|| < tol).
     maxiter:
         Maximum iterations (MATLAB code uses 10000).
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
 
     D = A.diagonal()
     # M = D/omega + tril(A, -1)
     M = sp.tril(A, k=-1).tocsr()
     M = M + sp.diags(D / omega, format="csr")
 
     r = b - A @ x
     norm0 = np.linalg.norm(r)
     if norm0 == 0:
         return SorResult(iterations=0, x=x)
 
     for k in range(1, maxiter + 1):
         dx = spla.spsolve_triangular(M, r, lower=True)
         x = x + dx
         r = b - A @ x
         if np.linalg.norm(r) / norm0 < tol:
             return SorResult(iterations=k, x=x)
 
     return SorResult(iterations=maxiter, x=x)
 
