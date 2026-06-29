from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math


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
        
        self.gamma = float(self.gamma)
        self.mu = float(self.mu)

        if self.exploration is None:
           q, g = self.compute_d_optimal_fw(self.L, self.mu)
        else:
            q = np.asarray(self.exploration, dtype=float).reshape(-1)
            if q.shape != (self.K,):
                raise ValueError(f"exploration must have shape (K,)=({self.K},)")
            s = float(np.sum(q))
            if (not np.isfinite(s)) or s <= 0:
                raise ValueError("exploration distribution must sum to a positive finite number")
            q = q / s
        self.q = q

        dadv = self.adv_effective_dimension(self.eigenvalues, self.mu)

        #self.eta = np.sqrt(np.log(self.K)/(2000*(g+dadv)))
        #self.gamma = self.eta*(g+self.smoothness*np.sqrt(self.mu*g))

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

        loss = float(loss)/99.72476923076923
        p = self._last_p
        it = int(self._last_i)

        # V = sum_i p(i) a_i a_i^T  where a_i are in eigenbasis (rows of U)
        # Efficiently: V = A^T diag(p) A, where A = arms (KxK).
        A = self.arms
        #V = A.T @ (p[:, None] * A)

        # M = mu*diag(lam) + V  (symmetric PSD)
        #M = V + np.diag(self.mu * self.lam)

        if self.L is None:
            raise RuntimeError("Exp3Spectral requires L= (arm graph Laplacian) for update().")
        # Use sparse diag(p); np.diag(p) is a dense K×K and would densify sparse+sparse sums.
        L_sp = sp.csr_matrix(self.L) if isinstance(self.L, np.ndarray) else self.L.tocsr()
        D = sp.diags(np.asarray(p, dtype=float), offsets=0, shape=(self.K, self.K), format="csr")
        M = (self.mu * L_sp).tocsr() + D
        M = M.tocsr()

        
        # loss_vec_hat = M^{-1} a_it * loss
        #a_it = A[it, :]
        hot = np.zeros(self.K, dtype=float)
        hot[it] = 1.0
        try:
            loss_vec_hat = spla.spsolve(M, hot * loss)
        except np.linalg.LinAlgError:
            # fallback: add tiny ridge if numerical issues
            ridge = 1e-9
            M_ridge = M + ridge * (sp.eye(self.K, format="csr") if sp.issparse(M) else np.eye(self.K))
            if sp.issparse(M_ridge):
                M_ridge = M_ridge.tocsr()
            loss_vec_hat = spla.spsolve(M_ridge, hot * loss)
        
        # loss_hat over arms: U @ loss_vec_hat
        loss_hat =loss_vec_hat

        # bonus(i) = smoothness*sqrt(mu) * ||a_i||_{M^{-1}}
        sqrt_mu = float(np.sqrt(max(self.mu, 0.0)))
        bonus = np.zeros(self.K, dtype=float)
        for i in range(self.K):
            hot = np.zeros(self.K, dtype=float)
            hot[i] = 1.0
            try:
                x = spla.spsolve(M, hot)
            except np.linalg.LinAlgError:
                x = spla.spsolve(M + 1e-9 * np.eye(self.K), hot)
            val = float(x[i])
            bonus[i] = self.smoothness * sqrt_mu * float(np.sqrt(max(val, 0.0)))

        self.S += (loss_hat - bonus)

    def action_probabilities(self) -> np.ndarray:
        """Return the last sampling distribution over arms."""
        if self._last_p is None:
            return np.ones(self.K, dtype=float) / float(self.K)
        return np.asarray(self._last_p, dtype=float).copy()


    def gradient(self, x, L, mu):
        grad = np.zeros(self.K)
        for i in range(self.K):
            hot = np.zeros(self.K)
            hot[i] = 1
            grad[i] = -(spla.spsolve(mu * L + sp.eye(self.K), hot)[i])
        return grad


    def compute_d_optimal_fw(self, L, mu, tol=1e-2, default_step=False):
        """
            Compute D optimal design using Frank-Wolfe

            :param L: Laplacian matrix
            :param mu: mu parameter
            :param tol: tolerance
            :param default_step: whether to use the default step size
            :return: Lambda-regularized D-optimal design
            """
        N = L.shape[0]
        L = sp.csr_matrix(L)
        pi = np.ones(N) / N

        k = 0
        while True:
            grad = -self.gradient(pi, L, mu)
            i = np.argmax(grad)
            hot = np.zeros(L.shape[0])
            hot[i] = 1

            tr = np.dot(pi, grad)
            err = abs(grad[i] - tr) / tr
            if err <= tol:
                return pi, grad[i]
            #print(err)

            if default_step:
                step = 2.0 / (k + 2.0)
                k += 1
            else:
                step = max(0.0, min(1.0, (grad[i] / tr - 1.0) / max(grad[i] - 1.0, 1e-8)))
            pi = (1 - step) * pi + step * hot
        
    def adv_effective_dimension(self, eigenvalues, mu):
        N = len(eigenvalues)
        K = np.sum(np.isclose(eigenvalues, 0))


        # Lambda = eigenvalues + lambda_reg
        Lambda = eigenvalues  # using no lambda regularization

        # Find omega
        omega = 0
        lambda_sum = 0
        lambda_sqrt_sum = 0
        for i in range(K + 1, N + 1):
            omega = i
            lambda_sum += Lambda[i - 1]
            lambda_sqrt_sum += math.sqrt(Lambda[i - 1])
            if math.sqrt(Lambda[i - 1]) * ((1 + mu * lambda_sum) / lambda_sqrt_sum) - (mu * Lambda[i - 1]) <= 0:
                omega = i - 1
                lambda_sum -= Lambda[i - 1]
                lambda_sqrt_sum -= math.sqrt(Lambda[i - 1])
                break

        # Compute N-tuple p = (p_1, ... p_N)
        p = []
        for i in range(K + 1, omega + 1):
            p.append(math.sqrt(Lambda[i - 1]) * ((1 + mu * lambda_sum) / lambda_sqrt_sum) - (mu * Lambda[i - 1]))
        p = np.array(p)

        # Plug into effective dimension definition
        d = K + np.sum(p / (mu * Lambda[K:omega] + p))
        return d

