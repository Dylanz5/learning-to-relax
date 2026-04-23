from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tsallis_inf import TsallisINF


@dataclass
class TsallisINFCB:
     """Contextual bandit via discretization + Tsallis-INF per bin.
 
     Port of `learners/TsallisINFCB.m`.
     """
 
     grid: np.ndarray
     m: int
     a: float
     b: float
 
     def __post_init__(self) -> None:
         self.grid = np.asarray(self.grid, dtype=float).reshape(-1)
         self.handles = [TsallisINF(self.grid, T=0) for _ in range(int(self.m))]
         self.disc = self.a + (self.b - self.a) * np.linspace(0.5 / self.m, 1.0 - 0.5 / self.m, self.m)
         self.action: int | None = None
 
     def predict(self, context: float, rng: np.random.Generator | None = None) -> float:
         i = int(np.argmin(np.abs(self.disc - float(context))))
         self.action = i
         return self.handles[i].predict(rng=rng)
 
     def update(self, loss: float) -> None:
         if self.action is None:
             raise RuntimeError("predict() must be called before update()")
         self.handles[self.action].update(loss)
 
