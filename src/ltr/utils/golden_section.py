from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def golden_section(
     eval_func: Callable[[float], float],
     start_grid: np.ndarray,
     start_evals: np.ndarray,
     N: int,
 ) -> tuple[np.ndarray, np.ndarray]:
     """Golden section search bookkeeping (port of `utils/golden_section.m`).
 
     Returns the queried omega grid and corresponding evaluations, including
     the initial coarse grid.
     """
 
     start_grid = np.asarray(start_grid, dtype=float).reshape(-1)
     start_evals = np.asarray(start_evals, dtype=float).reshape(-1)
     n = int(start_grid.shape[0])
     if start_evals.shape[0] != n:
         raise ValueError("start_grid and start_evals must have same length")
     if N < n:
         raise ValueError("N must be >= len(start_grid)")
 
     grid = np.zeros(N, dtype=float)
     evals = np.zeros(N, dtype=float)
     grid[:n] = start_grid
     evals[:n] = start_evals
 
     best = float(np.min(start_evals))
     argmins = np.flatnonzero(start_evals == best)
     left = max(0, int(np.min(argmins)) - 1)
     right = min(n - 1, int(np.max(argmins)) + 1)
     oleft = float(start_grid[left])
     oright = float(start_grid[right])
 
     tau = (math.sqrt(5.0) - 1.0) / 2.0
 
     # Initialize o1,f1,o2,f2 as in MATLAB code paths
     if right - left == 2:
         ocenter = float(start_grid[left + 1])
         n += 1
         if (oright - ocenter) > (ocenter - oleft):
             o1, f1 = ocenter, float(start_evals[left + 1])
             o2 = oleft + tau * (oright - oleft)
             grid[n - 1] = o2
             f2 = float(eval_func(o2))
             evals[n - 1] = f2
         elif (oright - ocenter) < (ocenter - oleft):
             o1 = oleft + (1.0 - tau) * (oright - oleft)
             grid[n - 1] = o1
             f1 = float(eval_func(o1))
             evals[n - 1] = f1
             o2, f2 = ocenter, float(start_evals[left + 1])
         else:
             o1 = oleft + (1.0 - tau) * (oright - oleft)
             grid[n - 1] = o1
             f1 = float(eval_func(o1))
             evals[n - 1] = f1
             n += 1
             o2 = oleft + tau * (oright - oleft)
             grid[n - 1] = o2
             f2 = float(eval_func(o2))
             evals[n - 1] = f2
     elif right - left == 3:
         o1, f1 = float(start_grid[left + 1]), float(start_evals[left + 1])
         o2, f2 = float(start_grid[right - 1]), float(start_evals[right - 1])
     else:
         n += 1
         o1 = oleft + (1.0 - tau) * (oright - oleft)
         grid[n - 1] = o1
         f1 = float(eval_func(o1))
         evals[n - 1] = f1
         n += 1
         o2 = oleft + tau * (oright - oleft)
         grid[n - 1] = o2
         f2 = float(eval_func(o2))
         evals[n - 1] = f2
 
     # Main loop
     for i in range(n, N):
         # MATLAB parity rule: (f1 > f2) || (f1 == f2 && rem(i,2))
         if (f1 > f2) or (f1 == f2 and (i % 2 == 1)):
             oleft = o1
             o1, f1 = o2, f2
             o2 = oleft + tau * (oright - oleft)
             grid[i] = o2
             f2 = float(eval_func(o2))
             evals[i] = f2
         else:
             oright = o2
             o2, f2 = o1, f1
             o1 = oleft + (1.0 - tau) * (oright - oleft)
             grid[i] = o1
             f1 = float(eval_func(o1))
             evals[i] = f1
 
     return grid, evals
 
