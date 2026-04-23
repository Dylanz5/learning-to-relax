from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp


CoefficientFn = Callable[[float], float]
ForcingFn = Callable[[float, np.ndarray], np.ndarray]
InitialFn = Callable[[np.ndarray], np.ndarray]
 
 
 @dataclass
 class Heat2D:
     """Crank–Nicolson heat equation driver (port of `utils/Heat2D.m`)."""
 
     coefficient: CoefficientFn
     forcing: ForcingFn
     initial: InitialFn
     nx: int
     dt: float
 
     def __post_init__(self) -> None:
         self.nx = int(self.nx)
         if self.nx < 2:
             raise ValueError("nx must be >= 2")
 
         self.dt = float(self.dt)
         self.t = 0.0
 
         # interior unknowns are (nx-1)^2 in the MATLAB code
         n1 = self.nx - 1
         self.n = n1 * n1
 
         dx = 1.0 / self.nx
         self.dx = dx
 
         # 2D Laplacian on interior with Dirichlet boundaries, scaled by 1/dx^2
         o = np.ones(n1, dtype=float)
         T = sp.diags([o, -2.0 * o, o], [-1, 0, 1], shape=(n1, n1), format="csr")
         I = sp.eye(n1, format="csr")
         self.L = (sp.kron(I, T) + sp.kron(T, I)) / (dx * dx)
         self.I = sp.eye(self.n, format="csr")
 
         # coordinates of interior points (same ordering as vectorization below)
         xs = (np.arange(1, n1 + 1, dtype=float) * dx)
         X, Y = np.meshgrid(xs, xs, indexing="ij")
         self.ijdx = np.column_stack([X.reshape(-1), Y.reshape(-1)])
 
         self.u = np.asarray(self.initial(self.ijdx), dtype=float).reshape(-1)
         if self.u.shape[0] != self.n:
             raise ValueError("initial() returned wrong shape")
 
     def crank_nicolson_system(self) -> tuple[sp.csr_matrix, np.ndarray]:
         halfstep = self.t + 0.5 * self.dt
         CL = 0.5 * float(self.coefficient(halfstep)) * self.dt * self.L
         A = (self.I - CL).tocsr()
         b = (self.I + CL) @ self.u + self.dt * np.asarray(self.forcing(halfstep, self.ijdx), dtype=float).reshape(-1)
         return A, b
 
     def update(self, u: np.ndarray) -> None:
         self.u = np.asarray(u, dtype=float).reshape(-1)
         self.t += self.dt
 
