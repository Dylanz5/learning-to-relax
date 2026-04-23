from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class ChebCB:
    """ChebCB contextual bandit (port of `learners/ChebCB.m`).

    Toolbox-free port using:
    - Chebyshev features via `numpy.polynomial.chebyshev.chebvander`
    - least squares via `numpy.linalg.lstsq`
    - bounded least squares via `scipy.optimize.lsq_linear`
    """

    grid: np.ndarray
    T: int
    m: int
    a: float
    b: float

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float).reshape(-1)
        self.d = int(self.grid.shape[0])
        self.T = int(self.T)
        self.m = int(self.m)
        self.t = 1  # MATLAB-style 1-indexed

        self.actions = np.zeros(self.T, dtype=int)
        self.theta = np.zeros((self.d, self.m), dtype=float)
        self.features = np.zeros((self.T, self.m), dtype=float)
        self.losses = np.zeros(self.T, dtype=float)

        self.eta = 0.0
        self.a = float(self.a)
        self.bma = float(self.b - self.a)
        if self.bma <= 0:
            raise ValueError("b must be > a")

        self.scale = 0.0
        self.predicted = np.zeros(self.T, dtype=float)
        self.contexts = np.zeros(self.T, dtype=float)

    def _cheb_features(self, context: float) -> np.ndarray:
        x = 2.0 / self.bma * (float(context) - self.a) - 1.0
        return np.polynomial.chebyshev.chebvander(np.array([x]), self.m - 1)[0]

    def predict(self, context: float, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()

        feature = self._cheb_features(context)
        self.features[self.t - 1, :] = feature

        yhat = self.theta @ feature
        istar = int(np.argmin(yhat))
        ystar = float(yhat[istar])

        probs = np.zeros(self.d, dtype=float)
        other = np.ones(self.d, dtype=bool)
        other[istar] = False
        denom = self.d + self.eta * (yhat[other] - ystar)
        denom = np.maximum(denom, 1e-12)
        probs[other] = 1.0 / denom
        probs[istar] = max(0.0, 1.0 - probs[other].sum())

        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            probs[:] = 1.0 / self.d
        else:
            probs /= s

        i = int(rng.choice(self.d, p=probs))
        self.actions[self.t - 1] = i

        self.predicted[self.t - 1] = float(self.theta[i, :] @ feature)
        self.contexts[self.t - 1] = float(context)
        return float(self.grid[i])

    def update(self, loss: float) -> None:
        self.losses[self.t - 1] = float(loss)

        K = float(np.max(self.losses[: self.t]))
        minL = float(np.min(self.losses[: self.t]))
        L = 0.0 if self.bma == 0 else (K - minL) / self.bma
        if K <= 0:
            K = 1.0

        N = 2.0 + 4.0 * self.bma * L / K * (1.0 + math.log(max(2, self.m)))
        update_all = self.scale != K * N
        self.scale = K * N

        ub = np.concatenate(([1.0 / N], (2.0 * self.bma * L / self.scale) / np.arange(1, self.m)))
        lb = -ub
        # SciPy requires strict lb < ub. When L==0, higher-order bounds become 0.
        tight = ub == 0.0
        lb = lb.copy()
        ub = ub.copy()
        lb[tight] = -1e-12
        ub[tight] = 1e-12

        resnorm = 0.0
        for i in range(self.d):
            if update_all or i == self.actions[self.t - 1]:
                idx = self.actions[: self.t] == i
                if np.any(idx):
                    A = self.features[: self.t][idx, :]
                    b = (self.losses[: self.t][idx] - 1.0) / self.scale
                    theta_i, *_ = np.linalg.lstsq(A, b, rcond=None)

                    if np.any(np.abs(theta_i) > ub):
                        sol = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol="auto", verbose=0)
                        theta_i = sol.x
                        resnorm = float(sol.cost * 2.0)

                    self.theta[i, :] = theta_i

        alpha = (math.pi + 2.0 / math.pi * math.log(2.0 * self.m + 1.0)) / (
            2.0 * self.scale * (self.m + 1.0)
        ) * self.bma * L
        R = float(np.sum((self.predicted[: self.t] - self.losses[: self.t] / (K * N)) ** 2))
        denom = max(1e-12, R - resnorm + 2.0 * alpha * alpha * self.t)
        self.eta = 2.0 * self.t * math.sqrt(self.d * self.t / denom)
        self.t += 1
 
