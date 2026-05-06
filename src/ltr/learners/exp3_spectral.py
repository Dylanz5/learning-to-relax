from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import scipy.sparse.linalg as spla


@dataclass
class Exp3Spectral:
    """Exp3-style algorithm for spectral bandits (minimal port of Alg. Exp3).

    - arms are a discrete grid of actions (e.g., omega candidates)
    - graph structure is a Laplacian over the arms, passed as an eigendecomposition
    - losses are scalar bandit feedback (e.g., solver iterations)
    """

    grid: np.ndarray
    eigenvectors: np.ndarray  # U, shape (K, K)
    eigenvalues: np.ndarray  # lam, shape (K,)
    smoothness: float = 1.0
    eta: float = 0.1  # learning rate
    gamma: float = 0.1  # exploration mixing
    mu: float = 1e-2  # regularization scale on eigenvalues
    exploration: np.ndarray | None = None  # q, shape (K,)
    L: sp.csr_matrix | None = None
    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float).reshape(-1)
        self.K = int(self.grid.shape[0])

        U = np.asarray(self.eigenvectors, dtype=float)
        lam = np.asarray(self.eigenvalues, dtype=float).reshape(-1)
        if U.shape != (self.K, self.K):
            raise ValueError(f"eigenvectors must have shape (K,K)=({self.K},{self.K})")
        if lam.shape != (self.K,):
            raise ValueError(f"eigenvalues must have shape (K,)=({self.K},)")

        self.U = U
        self.lam = lam
        self.smoothness = float(self.smoothness)
        self.eta = float(self.eta)
        self.gamma = float(self.gamma)
        self.mu = float(self.mu)

        if self.exploration is None:
            q = np.ones(self.K, dtype=float) / float(self.K)
        else:
            q = np.asarray(self.exploration, dtype=float).reshape(-1)
            if q.shape != (self.K,):
                raise ValueError(f"exploration must have shape (K,)=({self.K},)")
            s = float(np.sum(q))
            if (not np.isfinite(s)) or s <= 0:
                raise ValueError("exploration distribution must sum to a positive finite number")
            q = q / s
        self.q = q

        # In the eigenbasis, arm i is U^T e_i = U[i, :]^T.
        # So we can represent all arms as rows of U: arms[i] = U[i, :].
        self.arms = self.U.copy()

        # Cumulative scores S_i = sum_{s<t} (loss_hat_s(i) - bonus_s(i))
        self.S = np.zeros(self.K, dtype=float)

        # Last prediction state (used by update)
        self._last_p: np.ndarray | None = None
        self._last_i: int | None = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x - float(np.min(x))  # stable (min since we exponentiate -eta*S)
        ex = np.exp(-x)
        s = float(np.sum(ex))
        if (not np.isfinite(s)) or s <= 0:
            return np.ones_like(x) / float(x.size)
        return ex / s

    def predict(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()

        p_ftrl = self._softmax(self.eta * self.S)
        p = self.gamma * self.q + (1.0 - self.gamma) * p_ftrl

        # numerical guard
        p = np.maximum(p, 0.0)
        s = float(np.sum(p))
        if (not np.isfinite(s)) or s <= 0:
            p[:] = 1.0 / float(self.K)
        else:
            p /= s

        i = int(rng.choice(self.K, p=p))
        self._last_p = p
        self._last_i = i
        return float(self.grid[i])

    def update(self, loss: float) -> None:
        if self._last_p is None or self._last_i is None:
            raise RuntimeError("predict() must be called before update()")

        loss = float(loss)
        p = self._last_p
        it = int(self._last_i)

        # V = sum_i p(i) a_i a_i^T  where a_i are in eigenbasis (rows of U)
        # Efficiently: V = A^T diag(p) A, where A = arms (KxK).
        A = self.arms
        #V = A.T @ (p[:, None] * A)

        # M = mu*diag(lam) + V  (symmetric PSD)
        #M = V + np.diag(self.mu * self.lam)

        M = self.mu*self.L + np.diag(p)
        # loss_vec_hat = M^{-1} a_it * loss
        #a_it = A[it, :]
        hot = np.zeros(self.K, dtype=float)
        hot[it] = 1.0
        try:
            loss_vec_hat = spla.spsolve(M, hot * loss)
        except np.linalg.LinAlgError:
            # fallback: add tiny ridge if numerical issues
            ridge = 1e-9
            loss_vec_hat = spla.spsolve(M + ridge * np.eye(self.K), hot * loss)
        
        # loss_hat over arms: U @ loss_vec_hat
        loss_hat =loss_vec_hat

        # # bonus(i) = smoothness*sqrt(mu) * ||a_i||_{M^{-1}}
        # sqrt_mu = float(np.sqrt(max(self.mu, 0.0)))
        # bonus = np.zeros(self.K, dtype=float)
        # for i in range(self.K):
        #     hot = np.zeros(self.K, dtype=float)
        #     hot[i] = 1.0
        #     try:
        #         x = spla.spsolve(M, hot)
        #     except np.linalg.LinAlgError:
        #         x = spla.spsolve(M + 1e-9 * np.eye(self.K), hot)
        #     val = float(hot @ x)
        #     bonus[i] = self.smoothness * sqrt_mu * float(np.sqrt(max(val, 0.0)))

        self.S += loss_hat
