from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TsallisINF:
    """Tsallis-INF bandit algorithm (port of `learners/TsallisINF.m`)."""

    grid: np.ndarray
    T: int = 0

    def __post_init__(self) -> None:
         self.grid = np.asarray(self.grid, dtype=float).reshape(-1)
         self.d = int(self.grid.shape[0])
         self.t = 1  # MATLAB-style 1-indexed time for eta schedule
         self.k = np.zeros(self.d, dtype=float)
 
         self.index: int | None = None
         self.prob: float = 1.0 / self.d
         self.scale: float = 1.0
 
         if self.T and self.T > 0:
             self.actions = np.zeros(self.T, dtype=int)
             self.losses = np.zeros(self.T, dtype=float)
         else:
             self.actions = np.zeros(0, dtype=int)
             self.losses = np.zeros(0, dtype=float)
 
    def _ensure_capacity(self) -> None:
        if self.t <= self.losses.shape[0]:
            return
        # grow arrays (amortized)
        new_len = max(16, 2 * self.losses.shape[0])
        self.losses = np.pad(self.losses, (0, new_len - self.losses.shape[0]))
        self.actions = np.pad(self.actions, (0, new_len - self.actions.shape[0]))

    def predict(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()

        # MATLAB reference (`learners/TsallisINF.m`):
        #   eta = 2 / sqrt(t)
        #   x = -1
        #   for i = 1:20
        #       probs = 4 * (eta*(k/scale - x)).^(-2)
        #       x = x - (sum(probs) - 1) / (eta * sum(probs.^1.5))
        #   end
        #   index = randsample(1:d, 1, true, probs)   % try/catch fallback to uniform
        eta = 2.0 / np.sqrt(float(self.t))
        x = -1.0

        probs: np.ndarray | None = None
        try:
            for _ in range(20):
                denom = eta * (self.k / self.scale - x)
                probs = 4.0 * (denom ** (-2.0))
                x = x - (float(np.sum(probs)) - 1.0) / (eta * float(np.sum(probs ** 1.5)))
        except Exception:
            probs = None

        # randsample(...) expects nonnegative weights; if invalid, sample uniformly.
        if probs is None or (not np.all(np.isfinite(probs))) or np.any(probs < 0) or float(np.sum(probs)) <= 0:
            p = np.ones(self.d, dtype=float) / self.d
        else:
            p = probs / float(np.sum(probs))

        idx = int(rng.choice(self.d, p=p))
        self.index = idx
        self.prob = float(p[idx])

        self._ensure_capacity()
        self.actions[self.t - 1] = idx
        return float(self.grid[idx])
 
    def update(self, loss: float) -> None:
        if self.index is None:
            raise RuntimeError("predict() must be called before update()")
 
        self._ensure_capacity()
        self.losses[self.t - 1] = float(loss)

        # MATLAB: scale = mean(losses(1:t)) - 1
        mu = float(np.mean(self.losses[: self.t]))
        self.scale = mu - 1.0
 
        # MATLAB updates k using the action probability.
        self.k[self.index] = self.k[self.index] + (float(loss) - 1.0) / self.prob
        self.t += 1
 
