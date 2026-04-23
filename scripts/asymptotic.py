from __future__ import annotations
 
 import argparse
 from pathlib import Path
 
 import matplotlib.pyplot as plt
 import numpy as np
 import scipy.sparse as sp
 import scipy.sparse.linalg as spla
 
 from ltr.domains import delsq_numgrid
 from ltr.solvers.bounds import energy_norm_bound, omega_grid
 from ltr.solvers.sor import sor
 from ltr.utils.random import truncated_normal
 
 
 def ensure_plots_dir() -> Path:
     p = Path("plots")
     p.mkdir(parents=True, exist_ok=True)
     return p
 
 
 def _iteration_matrix(A: sp.spmatrix, omega: float) -> np.ndarray:
     A = A.tocsr()
     n = A.shape[0]
     D = A.diagonal()
     L = sp.tril(A, k=-1).tocsr()
     M = L + sp.diags(D / omega, format="csr")
     # C = I - A * inv(M)
     # Form dense inv(M) by solving M X = I
     I = np.eye(n)
     X = spla.spsolve(M.tocsc(), I)
     C = np.eye(n) - (A.toarray() @ X)
     return C
 
 
 def run(seed: int) -> None:
     dom = delsq_numgrid("S", 12)
     A = dom.A
     n = A.shape[0]
     b = truncated_normal(n, rng=np.random.default_rng(seed))
     epsilon = 1e-8
 
     omegas = omega_grid(A, 1.0, 1.99, 0.01)
     actual = np.zeros_like(omegas)
     radius = np.zeros_like(omegas)
     errs = np.zeros_like(omegas)
     energy = np.zeros_like(omegas)
 
     for i, om in enumerate(omegas):
         k = sor(A, b, np.zeros(n), float(om), epsilon).iterations
         C = _iteration_matrix(A, float(om))
         eigvals = np.linalg.eigvals(C)
         rad = float(np.max(np.abs(eigvals)))
         radius[i] = rad
         if k > 1:
             Cp = np.linalg.matrix_power(C, k - 1)
             errs[i] = float(np.linalg.norm(Cp, 2) ** (1.0 / (k - 1)) - rad)
         else:
             errs[i] = 0.0
         actual[i] = k
         energy[i] = energy_norm_bound(A, float(om))
 
     tau = float(np.max(errs / np.maximum(1e-12, 1.0 - radius)))
 
     plots = ensure_plots_dir()
     fig, ax = plt.subplots(figsize=(7, 5))
     ax.semilogy(omegas, actual, lw=2, label="actual cost")
     ax.semilogy(omegas, np.log(epsilon) / np.log(radius), lw=2, label="asymptotic estimate")
     ax.semilogy(omegas, np.log(epsilon / (2 * np.sqrt(np.linalg.cond(A.toarray())))) / np.log(energy), lw=2, label="energy bound")
     ax.semilogy(omegas, np.log(epsilon) / np.log(radius + tau * (1 - radius)), lw=2, label="near-asymptotic bound")
     ax.set_xlabel(r"$\\omega$", fontsize=14)
     ax.set_ylabel("iterations", fontsize=14)
     ax.legend(fontsize=11, loc="upper left")
     fig.tight_layout()
     fig.savefig(plots / "bound_comparison.png", dpi=256)
     plt.close(fig)
 
     fig, ax = plt.subplots(figsize=(7, 5))
     ax.plot(omegas, errs, lw=2, label=r"$||C^k||^{1/k} - \\rho(C)$")
     ax.plot(omegas, tau * (1 - radius), lw=2, label=r"$\\tau(1-\\rho(C))$")
     ax.set_xlabel(r"$\\omega$", fontsize=14)
     ax.legend(fontsize=11, loc="upper left")
     fig.tight_layout()
     fig.savefig(plots / "asymptocity.png", dpi=256)
     plt.close(fig)
 
 
 def main() -> None:
     p = argparse.ArgumentParser()
     p.add_argument("--seed", type=int, default=0)
     args = p.parse_args()
     run(seed=args.seed)
 
 
 if __name__ == "__main__":
     main()
 
