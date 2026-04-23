from __future__ import annotations
 
 import argparse
 from pathlib import Path
 
 import matplotlib.pyplot as plt
 import numpy as np
import scipy.sparse as sp
 from scipy.stats import beta as beta_dist
 
 from ltr.domains import delsq_numgrid
 from ltr.learners.chebcb import ChebCB
 from ltr.learners.tsallis_inf import TsallisINF
 from ltr.learners.tsallis_inf_cb import TsallisINFCB
 from ltr.solvers.bounds import omega_opt
 from ltr.solvers.sor import sor
 from ltr.utils.random import truncated_normal
 
 
 def ensure_plots_dir() -> Path:
     p = Path("plots")
     p.mkdir(parents=True, exist_ok=True)
     return p
 
 
 def run(T: int, trials: int, seed: int) -> None:
     dom = delsq_numgrid("S", 12)
     A = dom.A
     n = A.shape[0]
 
     epsilon = 1e-8
     rng_master = np.random.default_rng(seed)
 
     def one_trial(dist_a: float, dist_b: float, trial_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
         rng = np.random.default_rng(trial_seed)
         tinf = TsallisINF(np.linspace(1.0, 1.95, 20), T=T)
         cheb = ChebCB(np.linspace(1.0, 1.95, 20), T=T, m=6, a=-0.15, b=0.65)
         tinfcb = TsallisINFCB(np.linspace(1.0, 1.95, 20), m=5, a=-0.15, b=0.65)
 
         tinf_costs = np.zeros(T)
         cheb_costs = np.zeros(T)
         tinfcb_costs = np.zeros(T)
         opt_costs = np.zeros(T)
         omega_costs = np.zeros(T)
 
         for t in range(T):
             c = -0.15 + 0.6 * beta_dist.rvs(dist_a, dist_b, random_state=rng)
             At = A + c * sp.eye(n, format="csr")
             bt = truncated_normal(n, rng=rng)
 
             tinf_costs[t] = sor(At, bt, np.zeros(n), tinf.predict(rng=rng), epsilon).iterations
             tinf.update(tinf_costs[t])
 
             cheb_costs[t] = sor(At, bt, np.zeros(n), cheb.predict(c, rng=rng), epsilon).iterations
             cheb.update(cheb_costs[t])
 
             tinfcb_costs[t] = sor(At, bt, np.zeros(n), tinfcb.predict(c, rng=rng), epsilon).iterations
             tinfcb.update(tinfcb_costs[t])
 
             opt_costs[t] = sor(At, bt, np.zeros(n), omega_opt(At), epsilon).iterations
             omega_costs[t] = sor(At, bt, np.zeros(n), 1.8, epsilon).iterations
 
         return tinf_costs, omega_costs, tinfcb_costs, cheb_costs, opt_costs
 
     def plot(cost_arrays: list[np.ndarray], labels: list[str], outfile: str) -> None:
         plots = ensure_plots_dir()
         fig, ax = plt.subplots(figsize=(7, 5))
         for costs, lab in zip(cost_arrays, labels):
             ax.plot(np.mean(np.cumsum(costs, axis=0), axis=1), T - np.arange(1, T + 1), lw=2, label=lab)
         ax.set_xlabel("total iterations", fontsize=14)
         ax.set_ylabel("instances remaining", fontsize=14)
         ax.legend(fontsize=11)
         fig.tight_layout()
         fig.savefig(plots / outfile, dpi=256)
         plt.close(fig)
 
     # High-variance
     tinf = np.zeros((T, trials))
     omg = np.zeros((T, trials))
     tinfcb = np.zeros((T, trials))
     cheb = np.zeros((T, trials))
     opt = np.zeros((T, trials))
 
     for trial in range(trials):
         tc, oc, tcc, cc, opc = one_trial(0.5, 1.5, int(rng_master.integers(0, 2**32 - 1)))
         tinf[:, trial] = tc
         omg[:, trial] = oc
         tinfcb[:, trial] = tcc
         cheb[:, trial] = cc
         opt[:, trial] = opc
 
     plot([tinf, omg, tinfcb, cheb, opt], ["Tsallis-INF", r"$\\omega=1.8$", "Tsallis-INF-CB", "ChebCB", "Instance-Optimal"], "contextual_high_variance.png")
 
     # Low-variance
     tinf[:] = 0
     omg[:] = 0
     tinfcb[:] = 0
     cheb[:] = 0
     opt[:] = 0
     for trial in range(trials):
         tc, oc, tcc, cc, opc = one_trial(2.0, 6.0, int(rng_master.integers(0, 2**32 - 1)))
         tinf[:, trial] = tc
         omg[:, trial] = oc
         tinfcb[:, trial] = tcc
         cheb[:, trial] = cc
         opt[:, trial] = opc
 
     plot([tinf, omg, tinfcb, cheb, opt], ["Tsallis-INF", r"$\\omega=1.6$", "Tsallis-INF-CB", "ChebCB", "Instance-Optimal"], "contextual_low_variance.png")
 
 
 def main() -> None:
     p = argparse.ArgumentParser()
     p.add_argument("--T", type=int, default=5000)
     p.add_argument("--trials", type=int, default=40)
     p.add_argument("--seed", type=int, default=0)
     args = p.parse_args()
     run(T=args.T, trials=args.trials, seed=args.seed)
 
 
 if __name__ == "__main__":
     main()
 
