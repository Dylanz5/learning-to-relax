from __future__ import annotations
 
 import argparse
 import time
 from pathlib import Path
 
 import numpy as np
 import scipy.sparse.linalg as spla
 
 from ltr.learners.chebcb import ChebCB
 from ltr.learners.tsallis_inf import TsallisINF
 from ltr.learners.tsallis_inf_cb import TsallisINFCB
 from ltr.pde.heat2d import Heat2D
 from ltr.solvers.bounds import omega_opt
 from ltr.solvers.ssor_pcg import ssor_pcg
 from ltr.utils.bump import bump
 from ltr.utils.golden_section import golden_section
 
 
 def ensure_plots_dir() -> Path:
     p = Path("plots")
     p.mkdir(parents=True, exist_ok=True)
     return p
 
 
 def run(T: int, max_level: int, trials: int, seed: int) -> None:
     # Mirrors `scripts/h2d.m` but with CLI knobs to keep runtime manageable.
     stoptime = 5.0
     dt = stoptime / T
     nxs = 25 * 2 ** np.arange(0, max_level + 1)
     epsilon = 1e-8
     omegas = np.array([1.0, 1.3, 1.5, 1.75, 1.95], dtype=float)
 
     def coefficient(t: float) -> float:
         return max(0.1 * np.sin(2.0 * np.pi * t), -10.0 * np.sin(2.0 * np.pi * t))
 
     def forcing(t: float, x: np.ndarray) -> np.ndarray:
         center = np.array([0.5 + np.cos(16.0 * np.pi * t) / 4.0, 0.5 + np.sin(16.0 * np.pi * t) / 4.0])
         return 32.0 * bump(x, center=center, radius=0.125)
 
     def initial(x: np.ndarray) -> np.ndarray:
         return bump(x, center=np.array([0.5, 0.5]), radius=0.25)
 
     rng = np.random.default_rng(seed)
 
     # This script is primarily a performance evaluation; for the port we
     # implement the main solver loops and print progress, but we do not
     # replicate every MATLAB stored artifact (contours/actions arrays).
     for nx in nxs:
         print(f"[nx={nx}] starting")
 
         # Baseline exact solve (spsolve) each step
         pde = Heat2D(coefficient, forcing, initial, int(nx), float(dt))
         t0 = time.time()
         for i in range(T):
             A, b = pde.crank_nicolson_system()
             u = spla.spsolve(A.tocsc(), b)
             pde.update(u)
         wall = time.time() - t0
         print(f"[nx={nx}] A\\b wallclock={wall:.3f}s")
 
         # Unpreconditioned CG
         pde = Heat2D(coefficient, forcing, initial, int(nx), float(dt))
         t0 = time.time()
         niters = 0
         for _ in range(T):
             A, b = pde.crank_nicolson_system()
             u, info = spla.cg(A, b, x0=pde.u, rtol=epsilon, atol=0.0, maxiter=10_000)
             niters += max(0, info) if info > 0 else 0
             pde.update(u)
         wall = time.time() - t0
         print(f"[nx={nx}] CG wallclock={wall:.3f}s")
 
         # SSOR-PCG for fixed omegas
         for om in omegas:
             pde = Heat2D(coefficient, forcing, initial, int(nx), float(dt))
             niters = 0
             t0 = time.time()
             for _ in range(T):
                 A, b = pde.crank_nicolson_system()
                 res = ssor_pcg(A, b, pde.u, float(om), epsilon)
                 niters += res.iterations
                 pde.update(res.x)
             wall = time.time() - t0
             print(f"[nx={nx}] SSOR-PCG omega={om:.2f} wallclock={wall:.3f}s iters={niters}")
 
         # Learned omega via Tsallis-INF (demonstration; keep trials small)
         for trial in range(trials):
             pde = Heat2D(coefficient, forcing, initial, int(nx), float(dt))
             tinf = TsallisINF(np.linspace(1.0, 1.95, 20), T=T)
             niters = 0
             for _ in range(T):
                 A, b = pde.crank_nicolson_system()
                 om = tinf.predict(rng=rng)
                 res = ssor_pcg(A, b, pde.u, float(om), epsilon)
                 tinf.update(res.iterations)
                 niters += res.iterations
                 pde.update(res.x)
             print(f"[nx={nx}] Tsallis-INF trial={trial+1}/{trials} iters={niters}")
 
         # Contextual learners (also demonstration)
         for trial in range(trials):
             pde = Heat2D(coefficient, forcing, initial, int(nx), float(dt))
             cheb = ChebCB(np.linspace(1.0, 1.95, 20), T=T, m=6, a=-0.15, b=0.65)
             tinfcb = TsallisINFCB(np.linspace(1.0, 1.95, 20), m=5, a=-0.15, b=0.65)
             niters_cheb = 0
             niters_tinfcb = 0
             for _ in range(T):
                 A, b = pde.crank_nicolson_system()
                 # Use diffusion coefficient as a proxy context as in MATLAB offsets
                 c = float(coefficient(pde.t))
                 om_cheb = cheb.predict(c, rng=rng)
                 res = ssor_pcg(A, b, pde.u, float(om_cheb), epsilon)
                 cheb.update(res.iterations)
                 niters_cheb += res.iterations
                 pde.update(res.x)
 
             pde = Heat2D(coefficient, forcing, initial, int(nx), float(dt))
             for _ in range(T):
                 A, b = pde.crank_nicolson_system()
                 c = float(coefficient(pde.t))
                 om_cb = tinfcb.predict(c, rng=rng)
                 res = ssor_pcg(A, b, pde.u, float(om_cb), epsilon)
                 tinfcb.update(res.iterations)
                 niters_tinfcb += res.iterations
                 pde.update(res.x)
 
             print(f"[nx={nx}] ChebCB trial={trial+1}/{trials} iters={niters_cheb}")
             print(f"[nx={nx}] Tsallis-INF-CB trial={trial+1}/{trials} iters={niters_tinfcb}")
 
         # Instance-optimal omega (slow; optional demo for smaller nx)
         if nx <= 400:
             pde = Heat2D(coefficient, forcing, initial, int(nx), float(dt))
             for i in range(T):
                 A, b = pde.crank_nicolson_system()
                 eval_omega = lambda om: ssor_pcg(A, b, pde.u, float(om), epsilon).iterations
                 start = omegas
                 start_evals = np.array([eval_omega(om) for om in start], dtype=float)
                 og, ng = golden_section(eval_omega, start, start_evals, N=12)
                 best = og[int(np.argmin(ng))]
                 res = ssor_pcg(A, b, pde.u, float(best), epsilon)
                 pde.update(res.x)
                 if (i + 1) % 500 == 0:
                     print(f"[nx={nx}] instance-opt step {i+1}/{T}")
 
 
 def main() -> None:
     p = argparse.ArgumentParser()
     p.add_argument("--T", type=int, default=5000)
     p.add_argument("--max-level", type=int, default=2, help="0..4 in MATLAB; higher is much slower")
     p.add_argument("--trials", type=int, default=1)
     p.add_argument("--seed", type=int, default=0)
     args = p.parse_args()
     run(T=args.T, max_level=args.max_level, trials=args.trials, seed=args.seed)
 
 
 if __name__ == "__main__":
     main()
 
