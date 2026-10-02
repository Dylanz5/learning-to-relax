from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy (bits) of a probability vector; ignores non-positive mass."""
    q = np.asarray(p, dtype=float)
    q = q[q > 0.0]
    if q.size == 0:
        return 0.0
    return float(-np.sum(q * np.log2(q)))


@dataclass
class TsallisINF:
    """Tsallis-INF bandit algorithm (port of `learners/TsallisINF.m`)."""

    grid: np.ndarray
    T: int = 0
    #: Multiplier on the ``2/sqrt(t)`` learning-rate schedule. The MATLAB port
    #: fixes this at 1.0; exposing it lets a sweep tune Tsallis-INF's
    #: exploration/exploitation trade-off (larger -> more aggressive updates).
    eta_scale: float = 1.0

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float).reshape(-1)
        self.eta_scale = float(self.eta_scale)
        self.d = int(self.grid.shape[0])
        self.t = 1  # MATLAB-style 1-indexed time for eta schedule
        self.k = np.zeros(self.d, dtype=float)

        self.index: int | None = None
        # stored "probability" used in the importance-weighted update.
        # MATLAB code stores `probs(index)` where `probs` are the Newton-iteration
        # weights (not necessarily normalized to sum exactly to 1).
        self.prob: float = 1.0 / self.d
        self._last_p = np.ones(self.d, dtype=float) / float(self.d)

        # Diagnostics (logging only; no effect on the algorithm). ``diag`` holds
        # one scalar row (dict) per predict/update cycle; ``k_hist``/``p_hist``
        # hold the full per-round (K,) vectors of ``k`` and the sampling
        # distribution ``p``. ``_diag_pending`` stashes predict-time quantities
        # until the matching ``update`` closes out the row.
        self.diag: list[dict[str, float]] = []
        self.k_hist: list[np.ndarray] = []
        self.p_hist: list[np.ndarray] = []
        self._diag_pending: dict[str, float] | None = None
        self._pending_p: np.ndarray | None = None

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
        #       probs = 4 * (eta*(k - x)).^(-2)
        #       x = x - (sum(probs) - 1) / (eta * sum(probs.^1.5))
        #   end
        #   index = randsample(1:d, 1, true, probs)   % try/catch fallback to uniform
        #
        # The weights depend only on the *difference* ``L_i - x``, so the whole
        # normalization problem is translation-invariant in ``x``. We solve
        # ``sum_i 4/(eta*(L_i - x))^2 = 1`` for the multiplier ``x < min_i L_i``.
        # ``F(x) = sum probs(x) - 1`` is convex and strictly increasing on
        # ``(-inf, min L_i)`` with a pole at ``x = min L_i``; Newton is only safe
        # when started on the right of the root (``F(x0) >= 0``), otherwise it can
        # overshoot the pole and stall against it, dumping all mass on the argmin
        # arm.
        #
        # The MATLAB port hard-codes ``x0 = -1``. That is only on the safe side
        # when the loss estimates sit near 0 (the [0,1]-loss setting). Here ``k``
        # accumulates raw importance weights ``loss/prob``, so a fixed ``x0 = -1``
        # can land far to the *left* of the root; Newton then overshoots past
        # ``min L_i`` and collapses the distribution onto a single arm. Anchoring
        # the start just below the smallest estimate keeps ``F(x0) >= 0`` for any
        # loss magnitude and matches the MATLAB init when ``min L_i == 0``.
        eta = self.eta_scale * 2.0 / np.sqrt(float(self.t))
        L = self.k
        L_min = float(np.min(L)) if np.all(np.isfinite(L)) else -1.0
        x = L_min - 1.0

        probs: np.ndarray | None = None
        with np.errstate(all="ignore"):
            for _ in range(20):
                denom = eta * (L - x)
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

        # --- diagnostics (logging only) ---------------------------------
        p_used = np.asarray(self._last_p, dtype=float)
        argmax_arm = int(np.argmax(p_used))
        self._diag_pending = {
            "t": int(self.t),
            "arm": int(idx),
            "p_norm": float(p_used[idx]),
            "prob_raw": float(self.prob),
            "x": float(x) if np.isfinite(x) else float("nan"),
            "L_min": float(np.min(self.k)) if np.all(np.isfinite(self.k)) else float("nan"),
            "argmax_arm": argmax_arm,
            "p_max": float(p_used[argmax_arm]),
            "entropy": _shannon_entropy(p_used),
        }
        self._pending_p = p_used.copy()
        # ----------------------------------------------------------------

        self._ensure_capacity()
        self.actions[self.t - 1] = idx
        return float(self.grid[idx])

    def action_probabilities(self) -> np.ndarray:
        """Return the last sampling distribution over arms."""
        return np.asarray(self._last_p, dtype=float).copy()

    def loss_estimates(self) -> np.ndarray:
        """Per-arm cumulative importance-weighted loss ``k``.

        ``predict`` sets ``p_i ∝ (k_i - x)**(-2)``, so the arm with the
        smallest ``k_i`` receives the most mass.
        """
        return np.asarray(self.k, dtype=float).copy()

    def update(self, loss: float) -> None:
        if self.index is None:
            raise RuntimeError("predict() must be called before update()")

        idx = int(self.index)
        k_arm_before = float(self.k[idx])
        increment = float(loss) / self.prob

        self._ensure_capacity()
        self.losses[self.t - 1] = float(loss)
        self.k[self.index] = self.k[self.index] + float(loss) / self.prob
        self.t += 1

        # --- diagnostics (logging only) ---------------------------------
        if self._diag_pending is not None:
            row = dict(self._diag_pending)
            row.update(
                {
                    "loss": float(loss),
                    "increment": float(increment),
                    "k_arm_before": k_arm_before,
                    "k_arm_after": float(self.k[idx]),
                }
            )
            self.diag.append(row)
        self.k_hist.append(self.k.copy())
        if self._pending_p is not None:
            self.p_hist.append(self._pending_p.copy())
        else:
            self.p_hist.append(np.asarray(self._last_p, dtype=float).copy())
        self._diag_pending = None
        self._pending_p = None
        # ----------------------------------------------------------------
 
