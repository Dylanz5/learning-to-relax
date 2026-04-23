from __future__ import annotations
 
 import argparse
 from pathlib import Path
 
 import matplotlib.pyplot as plt
 import numpy as np
 import scipy.sparse as sp
 import scipy.sparse.linalg as spla
 from scipy.stats import beta as beta_dist
 
 from ltr.domains import delsq_numgrid
 from ltr.solvers.bounds import omega_grid
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
     I = np.eye(n)
     X = spla.spsolve(M.tocsc(), I)
     return np.eye(n) - (A.toarray() @ X)
 
 
 def run(trials: int, seed: int) -> None:
     dom = delsq_numgrid("S", 12)
     A0 = dom.A
     n = A0.shape[0]
     epsilon = 1e-8
     omegas = omega_grid(A0, 1.0, 1.99, 0.01)
 
     rng = np.random.default_rng(seed)
     plots = ensure_plots_dir()
 
     def evaluate(dist_a: float, dist_b: float, outfile: str) -> None:
         cs = -0.15 + 0.6 * beta_dist.rvs(dist_a, dist_b, size=trials, random_state=rng)
 
         actual = np.zeros_like(omegas)
         predicted = np.zeros_like(omegas)
         dynamic_actual = 0.0
         dynamic_predicted = 0.0
 
         for c in cs:
             Ac = A0 + float(c) * sp.eye(n, format="csr")
             b = truncated_normal(n, rng=rng)
 
             radius = np.zeros_like(omegas)
             errs = np.zeros_like(omegas)
             current_best = np.inf
 
             for j, om in enumerate(omegas):
                 k = sor(Ac, b, np.zeros(n), float(om), epsilon).iterations
                 C = _iteration_matrix(Ac, float(om))
                 rad = float(np.max(np.abs(np.linalg.eigvals(C))))
                 radius[j] = rad
                 if k > 1:
                     Cp = np.linalg.matrix_power(C, k - 1)
                     errs[j] = float(np.linalg.norm(Cp, 2) ** (1.0 / (k - 1)) - rad)
                 actual[j] += k
                 current_best = min(current_best, k)
 
             dynamic_actual += current_best
             tau = float(np.max(errs / np.maximum(1e-12, 1.0 - radius)))
             current_pred = 1.0 + (np.log(epsilon) / np.log(radius + tau * (1.0 - radius)))
             dynamic_predicted += float(np.min(current_pred))
             predicted += current_pred
 
         fig, ax = plt.subplots(figsize=(7, 5))
         ax.semilogy(omegas, actual / trials, lw=2, label="actual cost")
         ax.semilogy(omegas, (dynamic_actual / trials) * np.ones_like(omegas), lw=2, ls="--", label="(instance-optimal)")
         ax.semilogy(omegas, predicted / trials, lw=2, label="near-asymptotic bound")
         ax.semilogy(omegas, (dynamic_predicted / trials) * np.ones_like(omegas), lw=2, ls="--", label="(instance-optimal)")
         ax.set_xlabel(r"$\\omega$", fontsize=14)
         ax.set_ylabel("iterations", fontsize=14)
         ax.legend(fontsize=11, loc="upper center")
         fig.tight_layout()
         fig.savefig(plots / outfile, dpi=256)
         plt.close(fig)
 
     evaluate(2.0, 6.0, "low_variance.png")
     evaluate(0.5, 1.5, "high_variance.png")
 
 
 def main() -> None:
     p = argparse.ArgumentParser()
     p.add_argument("--trials", type=int, default=40)
     p.add_argument("--seed", type=int, default=0)
     args = p.parse_args()
     run(trials=args.trials, seed=args.seed)
 
 
 if __name__ == "__main__":
     main()
 
