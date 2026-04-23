from __future__ import annotations
 
 import argparse
 from pathlib import Path
 
 import matplotlib.pyplot as plt
 import numpy as np
 import scipy.sparse as sp
 import scipy.sparse.linalg as spla
 
 from ltr.domains import delsq_numgrid
 from ltr.solvers.bounds import omega_grid
 from ltr.solvers.sor import sor
 from ltr.utils.random import truncated_normal
 
 
 def ensure_plots_dir() -> Path:
     p = Path("plots")
     p.mkdir(parents=True, exist_ok=True)
     return p
 
 
 def run(seed: int) -> None:
     dom = delsq_numgrid("S", 12)
     A = dom.A
     n = A.shape[0]
     epsilon = 1e-8
 
     omegas = omega_grid(A, 1.0, 1.9, 0.001)
 
     # iteration matrix at omega=1.4
     D = A.diagonal()
     L = sp.tril(A, k=-1).tocsr()
     M = L + sp.diags(D / 1.4, format="csr")
     I = np.eye(n)
     X = spla.spsolve(M.tocsc(), I)
     C = np.eye(n) - (A.toarray() @ X)
 
     # smallest eigenvector of C (MATLAB picks column 99; we pick smallest magnitude)
     eigvals, eigvecs = np.linalg.eig(C)
     idx = int(np.argmin(np.abs(eigvals)))
     degen = np.real(eigvecs[:, idx])
 
     degen_cost = np.zeros_like(omegas)
     gauss_cost = np.zeros_like(omegas)
     gauss_min = np.zeros_like(omegas)
     gauss_max = np.zeros_like(omegas)
 
     rng = np.random.default_rng(seed)
     for i, om in enumerate(omegas):
         degen_cost[i] = sor(A, degen, np.zeros(n), float(om), epsilon).iterations
         costs = np.array([sor(A, truncated_normal(n, rng=rng), np.zeros(n), float(om), epsilon).iterations for _ in range(40)])
         gauss_cost[i] = float(np.mean(costs))
         gauss_max[i] = float(np.max(costs))
         gauss_min[i] = float(np.min(costs))
 
     plots = ensure_plots_dir()
     fig, ax = plt.subplots(figsize=(7, 5))
     ax.fill_between(omegas, gauss_min, gauss_max, color=(1.0, 0.8, 0.8), linewidth=0)
     ax.plot(omegas, degen_cost, lw=2, label="degenerate cost")
     ax.plot(omegas, gauss_cost, lw=2, label="mean cost")
     ax.set_xlabel(r"$\\omega$", fontsize=14)
     ax.set_ylabel("iterations", fontsize=14)
     ax.set_xlim(1.0, 1.9)
     ax.legend(fontsize=11, loc="upper center")
     fig.tight_layout()
     fig.savefig(plots / "degenerate.png", dpi=256)
     plt.close(fig)
 
 
 def main() -> None:
     p = argparse.ArgumentParser()
     p.add_argument("--seed", type=int, default=0)
     args = p.parse_args()
     run(seed=args.seed)
 
 
 if __name__ == "__main__":
     main()
 
