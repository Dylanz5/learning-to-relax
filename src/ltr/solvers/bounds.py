from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def omega_opt(A: sp.spmatrix) -> float:
     """Asymptotically optimal omega for SOR (port of `omega_opt.m`)."""
 
     beta = rho_jacobi(A)
     return float(1.0 + (beta / (1.0 + math.sqrt(max(0.0, 1.0 - beta * beta)))) ** 2)
 
 
 def omega_grid(A: sp.spmatrix, omega_min: float, omega_max: float, step: float) -> np.ndarray:
     """Evenly spaced omegas plus omega_opt(A) (port of `omega_grid.m`)."""
 
     opt = omega_opt(A)
     before = int(math.floor((opt - omega_min) / step) + 1)
     after = int(math.floor((omega_max - opt) / step) + 1)
 
     grid = np.empty(before + 1 + after, dtype=float)
     omega = omega_min
     for i in range(before):
         grid[i] = omega
         omega += step
     grid[before] = opt
     for i in range(before + 1, before + after + 1):
         grid[i] = omega
         omega += step
     return grid
 
 
 def rho_jacobi(A: sp.spmatrix) -> float:
     """Spectral radius of Jacobi iteration matrix (port of `rho_jacobi.m`)."""
 
     A = A.tocsr()
     n = A.shape[0]
     D = A.diagonal().astype(float)
     if np.any(D == 0):
         raise ValueError("A has zero diagonal entries; Jacobi undefined")
 
     Dinv = sp.diags(1.0 / D, format="csr")
     B = (sp.eye(n, format="csr") - Dinv @ A).tocsr()
 
     # largest magnitude eigenvalue (B is generally non-symmetric)
     lam = spla.eigs(B, k=1, which="LM", return_eigenvectors=False)[0]
     return float(abs(lam))
 
 
 def energy_norm_bound(A: sp.spmatrix, omega: float) -> float:
     """Bound on energy norm of SOR iteration matrix (port of `energy_norm.m`)."""
 
     A = A.tocsr()
     D = A.diagonal().astype(float)
     if np.any(D == 0):
         raise ValueError("A has zero diagonal entries")
 
     Omega = (2.0 - omega) / (2.0 * omega)
     invD = sp.diags(1.0 / D, format="csr")
     L = sp.tril(A, k=-1).tocsr()
 
     # gamma = 1 - max(abs(eig(invD*(L+L'))))
     S = (invD @ (L + L.T)).tocsr()
     lamS = spla.eigs(S, k=1, which="LM", return_eigenvectors=False)[0]
     gamma = 1.0 - abs(lamS)
 
     # max(abs(eig(invD*L*invD*L')))
     T = (invD @ L @ invD @ L.T).tocsr()
     lamT = spla.eigs(T, k=1, which="LM", return_eigenvectors=False)[0]
     maxT = abs(lamT)
 
     denom = (Omega * Omega + gamma / omega + maxT - 0.25)
     val = 1.0 - 2.0 * Omega * gamma / denom
     return float(math.sqrt(max(0.0, val)))
 
 
 def cgbound(A: sp.spmatrix, omegas: np.ndarray, epsilon: float) -> np.ndarray:
     """Axelsson-style SSOR-PCG iteration bound (port of `cgbound.m`).
 
     Parameters
     ----------
     A:
         SPD matrix.
     omegas:
         Array of omega values.
     epsilon:
         Target tolerance.
 
     Returns
     -------
     np.ndarray
         Bound for each omega.
     """
 
     A = A.tocsc()
     n = A.shape[0]
     Ddiag = A.diagonal().astype(float)
     if np.any(Ddiag == 0):
         raise ValueError("A has zero diagonal entries")
     D = sp.diags(Ddiag, format="csc")
     invD = sp.diags(1.0 / Ddiag, format="csc")
     L = sp.tril(A, k=-1).tocsc()
 
     solveA = spla.factorized(A)
 
     # kappa = lambda_max(A) / lambda_min(A)
     lam_max = spla.eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]
     lam_min = spla.eigsh(A, k=1, which="SA", return_eigenvectors=False)[0]
     kappa = float(lam_max / lam_min)
 
     # alpha = largest eigenvalue of D*A^{-1} (similar to SPD)
     def alpha_mv(v: np.ndarray) -> np.ndarray:
         return Ddiag * solveA(v)
 
     alpha_op = spla.LinearOperator((n, n), matvec=alpha_mv, dtype=float)
     alpha = float(np.real(spla.eigs(alpha_op, k=1, which="LR", return_eigenvectors=False)[0]))
 
     # beta = max eigenvalue of (L*inv(D)*L' - .25*D) * A^{-1}
     B = (L @ invD @ L.T - 0.25 * D).tocsc()
 
     def beta_mv(v: np.ndarray) -> np.ndarray:
         return B @ solveA(v)
 
     beta_op = spla.LinearOperator((n, n), matvec=beta_mv, dtype=float)
     beta = float(np.real(spla.eigs(beta_op, k=1, which="LR", return_eigenvectors=False)[0]))
 
     omega = np.asarray(omegas, dtype=float)
     tmo = 2.0 - omega
     tmooo = tmo / omega
     ic = tmo / (1.0 + 0.25 * tmo * tmooo * alpha + beta * omega)
 
     # bound = 1 + log(sqrt(kappa)/eps + sqrt(kappa/eps^2-1)) / -log(1 - 2/(1 + sqrt(1/ic)))
     num = np.log(np.sqrt(kappa) / epsilon + np.sqrt(kappa / (epsilon * epsilon) - 1.0))
     denom = -np.log(1.0 - 2.0 / (1.0 + np.sqrt(1.0 / ic)))
     return 1.0 + num / denom
 
