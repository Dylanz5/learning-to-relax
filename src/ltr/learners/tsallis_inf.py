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
        # stored "probability" used in the importance-weighted update.
        # MATLAB code stores `probs(index)` where `probs` are the Newton-iteration
        # weights (not necessarily normalized to sum exactly to 1).
        self.prob: float = 1.0 / self.d
        self.scale: float = 1.0
        self._last_p = np.ones(self.d, dtype=float) / float(self.d)

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
        with np.errstate(all="ignore"):
            for _ in range(20):
                denom = eta * (self.k / self.scale - x)
                probs = 4.0 * (denom ** (-2.0))
                x = x - (float(np.sum(probs)) - 1.0) / (eta * float(np.sum(probs ** 1.5)))

        # MATLAB uses try/catch around randsample with weights; if that fails,
        # it samples uniformly. We emulate that behavior while keeping the
        # update safe under invalid numerical weights.
        idx: int
        prob_for_update: float
        try:
            if probs is None:
                raise ValueError("probs unavailable")
            s = float(np.sum(probs))
            if (not np.isfinite(s)) or s <= 0 or (not np.all(np.isfinite(probs))):
                raise ValueError("invalid weight vector")
            p = probs / s
            if np.any(p < 0) or (not np.isfinite(float(np.sum(p)))):
                raise ValueError("invalid sampling distribution")

            idx = int(rng.choice(self.d, p=p))
            self._last_p = p.copy()
            w = float(probs[idx])
            # MATLAB stores the raw weight; fall back to true probability if needed.
            if np.isfinite(w) and w > 0:
                prob_for_update = w
            else:
                prob_for_update = float(p[idx]) if float(p[idx]) > 0 else (1.0 / self.d)
        except Exception:
            idx = int(rng.integers(0, self.d))
            self._last_p = np.ones(self.d, dtype=float) / float(self.d)
            # MATLAB's implementation still assigns `prob = probs(index)` after the
            # catch block. To stay close while remaining numerically safe, use the
            # raw weight when it's valid; otherwise use the uniform probability.
            if probs is not None:
                w = float(probs[idx])
                prob_for_update = w if (np.isfinite(w) and w > 0) else (1.0 / self.d)
            else:
                prob_for_update = 1.0 / self.d

        self.index = idx
        self.prob = prob_for_update

        self._ensure_capacity()
        self.actions[self.t - 1] = idx
        return float(self.grid[idx])

    def action_probabilities(self) -> np.ndarray:
        """Return the last sampling distribution over arms."""
        return np.asarray(self._last_p, dtype=float).copy()
 
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
 
