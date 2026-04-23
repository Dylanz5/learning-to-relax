from __future__ import annotations
 
 import argparse
 from pathlib import Path
 
 import matplotlib.pyplot as plt
 import numpy as np
import scipy.sparse as sp
 
 from ltr.domains import delsq_numgrid
 from ltr.solvers.bounds import cgbound
 from ltr.solvers.ssor_pcg import ssor_pcg
 from ltr.utils.random import truncated_normal
 
 
 def ensure_plots_dir() -> Path:
     p = Path("plots")
     p.mkdir(parents=True, exist_ok=True)
     return p
 
 
 def run(seed: int) -> None:
     plots = ensure_plots_dir()
     rng = np.random.default_rng(seed)
 
     for region in ("L", "S"):
         for s in (12, 32):
             for offset in (0.0, 0.5):
                 dom = delsq_numgrid(region, s)
                 A = dom.A
                 n = A.shape[0]
                 if offset != 0.0:
                     A = A + offset * sp.eye(n, format="csr")
 
                 b = truncated_normal(n, rng=rng)
                 K = 100
                 epsilon = 1e-8
                 omegas = np.linspace(2.0 * np.sqrt(2.0) - 2.0, 1.9, K)
 
                 costs = np.zeros_like(omegas)
                 bounds = cgbound(A, omegas, epsilon) - 1.0
 
                 for i, om in enumerate(omegas):
                     costs[i] = ssor_pcg(A, b, np.zeros(n), float(om), epsilon).iterations
 
                 tau = float(np.max((costs - 1.0) / np.maximum(1e-12, bounds)))
 
                 fig, ax = plt.subplots(figsize=(7, 5))
                 ax.plot(omegas, costs, lw=2, label="actual cost")
                 ax.plot(omegas, 1.0 + tau * bounds, lw=2, ls="--", label="upper bound")
                 ax.set_title(f"{region}-shaped domain: size={n}, offset={offset}", fontsize=14)
                 ax.set_xlabel(r"$\\omega$", fontsize=14)
                 ax.set_ylabel("iterations", fontsize=14)
                 ax.legend(fontsize=11, loc="upper center")
                 fig.tight_layout()
                 fig.savefig(plots / f"cgbound-{region}-{n}-{offset}.png", dpi=256)
                 plt.close(fig)
 
 
 def main() -> None:
     p = argparse.ArgumentParser()
     p.add_argument("--seed", type=int, default=0)
     args = p.parse_args()
     run(seed=args.seed)
 
 
 if __name__ == "__main__":
     main()
 
