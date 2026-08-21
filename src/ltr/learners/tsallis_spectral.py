from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class TsallisSpectral:
    """FTRL with the Tsallis entropy for spectral bandits (Algorithm 2).

    This is the Tsallis-entropy counterpart of :class:`Exp3Spectral` (which uses
    the Shannon entropy / softmax). It shares all of the spectral machinery with
    Exp3-Spectral -- D-optimal exploration design ``pi``, the loss-estimate solve
    ``f_hat = (mu*L + diag(p))^{-1} e_at * y``, and the bonus
    ``b(i) = beta * ||x_i||_{(mu*L + V_t)^{-1}}`` -- but replaces the closed-form
    softmax FTRL step with the Tsallis-entropy regularizer

        psi_{TE,alpha}(q) = (1/alpha) * (1 - sum_i q(i)^alpha),   alpha in (0, 1).

    Because that FTRL objective has no closed form, ``q_t`` is obtained with
    Newton's method on the dual (normalization) variable -- Algorithm 4 in the
    paper -- rather than by explicit constrained minimization. At ``alpha = 1/2``
    this reduces to the Tsallis-INF solver in :class:`TsallisINF`.

    Interface matches the other learners (duck-typed by the drivers):
    ``predict(rng) -> float`` action, ``update(loss)``, ``action_probabilities()``
    and ``loss_estimates()``.
    """

    grid: np.ndarray
    eigenvectors: np.ndarray  # U, shape (K, K)
    eigenvalues: np.ndarray  # lam, shape (K,)
    smoothness: float = 1.0  # C in the paper (used for the bonus beta = C*sqrt(mu))
    eta: float = 0.05  # learning rate
    gamma: float = 0.1  # exploration mixing weight
    mu: float = 1e-2  # regularization scale on the graph Laplacian
    alpha: float = 0.5  # Tsallis entropy parameter in (0, 1)
    exploration: np.ndarray | None = None  # pi, shape (K,); default = D-optimal design
    L: sp.csr_matrix | None = None
    use_bonus: bool = True  # include the exploration bonus b_t(i) from Eq. (5)

    # Newton solver controls (Algorithm 4).
    newton_tol: float = 1e-10
    newton_max_iter: int = 50

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float).reshape(-1)
        self.K = int(self.grid.shape[0])

        U = np.asarray(self.eigenvectors, dtype=float)
        lam = np.asarray(self.eigenvalues, dtype=float).reshape(-1)
        if U.shape != (self.K, self.K):
            raise ValueError(f"eigenvectors must have shape (K,K)=({self.K},{self.K})")
        if lam.shape != (self.K,):
            raise ValueError(f"eigenvalues must have shape (K,)=({self.K},)")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"alpha must lie in (0, 1); got {self.alpha!r}")
        if float(self.eta) <= 0.0:
            raise ValueError("eta must be positive")

        self.U = U
        self.lam = lam
        self.smoothness = float(self.smoothness)
        self.eta = float(self.eta)
        self.gamma = float(self.gamma)
        self.mu = float(self.mu)
        self.alpha = float(self.alpha)

        # Exploration distribution pi: use the regularized D-optimal design by
        # default (same as Exp3-Spectral), otherwise accept a user-supplied pi.
        if self.exploration is None:
            pi, g = self.compute_d_optimal_fw(self.L, self.mu)
        else:
            pi = np.asarray(self.exploration, dtype=float).reshape(-1)
            if pi.shape != (self.K,):
                raise ValueError(f"exploration must have shape (K,)=({self.K},)")
            s = float(np.sum(pi))
            if (not np.isfinite(s)) or s <= 0:
                raise ValueError("exploration distribution must sum to a positive finite number")
            pi = pi / s
            g = float("nan")
        self.pi = pi
        self.g_pi = float(g)

        # Effective dimension d_adv (diagnostic; also used by theoretical params).
        self.dadv = self.adv_effective_dimension(self.eigenvalues, self.mu)

        # In the eigenbasis arm i is U[i, :]; but in the node basis the arm
        # feature is simply e_i, so V_t = sum_i p_t(i) e_i e_i^T = diag(p_t) and
        # everything below stays in the node basis (matching Exp3Spectral).
        self.arms = self.U.copy()

        # Cumulative losses F_t = sum_{s<t} (f_hat_s - b_s), the FTRL statistic.
        self.S = np.zeros(self.K, dtype=float)

        # Warm-start for Algorithm 4's dual variable omega (updated each round).
        self._omega: float = float("nan")

        # Running loss mean, used to normalize feedback to O(1) so the algorithm
        # is invariant to the absolute loss scale (mirrors Exp3Spectral).
        self._loss_running_sum: float = 0.0
        self._loss_running_count: int = 0

        # Last prediction state (consumed by update()).
        self._last_p: np.ndarray | None = None
        self._last_i: int | None = None

    # ------------------------------------------------------------------ #
    # Algorithm 4: Newton's method for q_t                               #
    # ------------------------------------------------------------------ #
    def _solve_q(self, F: np.ndarray) -> np.ndarray:
        """Solve for q_t via Newton's method on the dual variable (Algorithm 4).

        The FTRL/Tsallis stationarity condition gives, for a Lagrange multiplier
        ``omega`` enforcing ``sum_i q(i) = 1``,

            q(i) = (eta * (F(i) - omega))^{1/(alpha-1)}.

        Since ``1/(alpha-1) < 0`` for ``alpha in (0,1)``, every ``F(i) - omega``
        must stay strictly positive, i.e. ``omega < min_i F(i)`` (a pole sits at
        the min). ``phi(omega) = sum_i q(i) - 1`` is increasing and convex on
        ``(-inf, min F)``, so Newton is safe only when started to the *right* of
        the root (``phi >= 0``); anchoring just left of the pole guarantees this
        for any loss scale. The Newton step is

            omega <- omega - (1-alpha) * (sum_i q(i) - 1) / (eta * sum_i q(i)^{2-alpha}),

        which is exactly ``omega - phi(omega)/phi'(omega)``.
        """
        alpha, eta = self.alpha, self.eta
        exponent = 1.0 / (alpha - 1.0)  # negative
        Fmin = float(np.min(F))
        # Gap kept between omega and the pole when we have to reset/clamp; scaled
        # so it is meaningful regardless of the magnitude of F.
        gap = 1e-6 * (abs(Fmin) + 1.0)

        # Warm-start from the previous round, but never on/over the pole.
        omega = self._omega
        if (not np.isfinite(omega)) or omega >= Fmin:
            omega = Fmin - 1.0

        q = np.full(self.K, 1.0 / self.K, dtype=float)
        with np.errstate(all="ignore"):
            for _ in range(int(self.newton_max_iter)):
                diff = eta * (F - omega)  # strictly positive by construction
                q = diff ** exponent
                s = float(np.sum(q))
                if (not np.isfinite(s)) or s <= 0.0:
                    # Numerical blow-up: reset just left of the pole and retry.
                    omega = Fmin - gap
                    continue
                if abs(s - 1.0) <= self.newton_tol:
                    break
                denom = eta * float(np.sum(q ** (2.0 - alpha)))
                if (not np.isfinite(denom)) or denom <= 0.0:
                    break
                step = (1.0 - alpha) * (s - 1.0) / denom
                omega_new = omega - step
                # Never cross the pole; if a step would, snap just left of it.
                if (not np.isfinite(omega_new)) or omega_new >= Fmin:
                    omega_new = Fmin - gap
                omega = omega_new

        self._omega = omega

        # Recompute q at the final omega and normalize defensively.
        with np.errstate(all="ignore"):
            q = (eta * (F - omega)) ** exponent
        q = np.where(np.isfinite(q), q, 0.0)
        q = np.maximum(q, 0.0)
        total = float(np.sum(q))
        if (not np.isfinite(total)) or total <= 0.0:
            return np.ones(self.K, dtype=float) / float(self.K)
        return q / total

    # ------------------------------------------------------------------ #
    # Bandit interface                                                   #
    # ------------------------------------------------------------------ #
    def predict(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()

        q_t = self._solve_q(self.S)
        p = self.gamma * self.pi + (1.0 - self.gamma) * q_t

        # numerical guard
        p = np.maximum(p, 0.0)
        s = float(np.sum(p))
        if (not np.isfinite(s)) or s <= 0:
            p = np.ones(self.K, dtype=float) / float(self.K)
        else:
            p = p / s

        i = int(rng.choice(self.K, p=p))
        self._last_p = p
        self._last_i = i
        return float(self.grid[i])

    def update(self, loss: float) -> None:
        if self._last_p is None or self._last_i is None:
            raise RuntimeError("predict() must be called before update()")

        # Normalize feedback by its running mean (scale invariance).
        self._loss_running_sum += float(loss)
        self._loss_running_count += 1
        loss_scale = self._loss_running_sum / self._loss_running_count
        if (not np.isfinite(loss_scale)) or loss_scale <= 0.0:
            loss_scale = 1.0
        loss = float(loss) / loss_scale

        p = self._last_p
        it = int(self._last_i)

        if self.L is None:
            raise RuntimeError("TsallisSpectral requires L= (arm graph Laplacian) for update().")

        # M = mu*L + V_t, with V_t = diag(p) in the node basis (x_i = e_i).
        L_sp = sp.csr_matrix(self.L) if isinstance(self.L, np.ndarray) else self.L.tocsr()
        D = sp.diags(np.asarray(p, dtype=float), offsets=0, shape=(self.K, self.K), format="csr")
        M = ((self.mu * L_sp).tocsr() + D).tocsc()

        beta = self.smoothness * float(np.sqrt(max(self.mu, 0.0)))  # beta = C*sqrt(mu)

        if self.use_bonus and beta != 0:
            # Need the full inverse: f_hat = M^{-1} e_it * loss uses one column,
            # and the bonus b(i) = beta*sqrt([M^{-1}]_{ii}) uses the diagonal.
            # For the small K used here this single factorized solve for all
            # columns is cheaper than K separate solves.
            Minv = spla.spsolve(M, sp.eye(self.K, format="csc"))
            Minv = Minv.toarray() if sp.issparse(Minv) else np.asarray(Minv)
            loss_hat = Minv[:, it] * loss
            diag = np.clip(np.diag(Minv), 0.0, None)
            bonus = beta * np.sqrt(diag)
        else:
            hot = np.zeros(self.K, dtype=float)
            hot[it] = 1.0
            try:
                loss_hat = spla.spsolve(M, hot * loss)
            except np.linalg.LinAlgError:
                M_ridge = (M + 1e-9 * sp.eye(self.K, format="csc")).tocsc()
                loss_hat = spla.spsolve(M_ridge, hot * loss)
            bonus = np.zeros(self.K, dtype=float)

        self.S += (loss_hat - bonus)

    def action_probabilities(self) -> np.ndarray:
        """Return the last sampling distribution over arms."""
        if self._last_p is None:
            return np.ones(self.K, dtype=float) / float(self.K)
        return np.asarray(self._last_p, dtype=float).copy()

    def loss_estimates(self) -> np.ndarray:
        """Per-arm cumulative FTRL statistic F_t (loss estimates minus bonuses)."""
        return np.asarray(self.S, dtype=float).copy()

    # ------------------------------------------------------------------ #
    # Optional: theoretical parameter settings from Eq. (8) / Thm 5.4    #
    # ------------------------------------------------------------------ #
    def set_theoretical_parameters(self, T: int) -> None:
        """Set (alpha, eta, gamma, mu) to the theory-driven values in Eq. (8).

        These give the regret bound of Theorem 5.4. By default the constructor
        keeps explicit hyperparameters (like Exp3Spectral); call this to switch
        to the analysis-backed schedule for horizon ``T``. ``mu`` (hence d_adv,
        pi and g(pi)) is recomputed since mu changes.
        """
        T = int(T)
        C = self.smoothness
        N = self.K

        # mu := 1/(C^2 T) does not depend on the others; recompute pi/g/d_adv.
        self.mu = 1.0 / (C * C * T)
        self.pi, self.g_pi = self.compute_d_optimal_fw(self.L, self.mu)
        self.dadv = self.adv_effective_dimension(self.eigenvalues, self.mu)

        log_plus = max(math.log(max(N / max(self.dadv, 1e-12), 1e-12)), 1.0)
        self.alpha = 1.0 - 1.0 / (2.0 * log_plus)

        self.eta = 0.5 * math.sqrt(
            (1.0 - self.alpha) * (N ** (1.0 - self.alpha))
            / (2.0 * self.alpha * (T + 1) * (self.dadv ** self.alpha))
        )

        g = self.g_pi
        self.gamma = g * (
            (8.0 * self.eta / (1.0 - self.alpha)) * (1.0 + 1.0 / math.sqrt(g * T))
        ) ** (1.0 / self.alpha)

    # ------------------------------------------------------------------ #
    # Spectral helpers (mirrors Exp3Spectral)                            #
    # ------------------------------------------------------------------ #
    def gradient(self, x, L, mu):
        """Gradient of the regularized D-optimal objective (Lemma 5.1).

        grad_i = [(mu*L + diag(p))^{-1}]_{ii}, evaluated at the design ``x`` (=p).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        M = (mu * sp.csr_matrix(L) + sp.diags(x, offsets=0, shape=(self.K, self.K))).tocsc()
        grad = np.zeros(self.K)
        for i in range(self.K):
            hot = np.zeros(self.K)
            hot[i] = 1.0
            grad[i] = -(spla.spsolve(M, hot)[i])
        return grad

    def compute_d_optimal_fw(self, L, mu, tol=1e-2, default_step=False):
        """Regularized D-optimal exploration design via Frank-Wolfe."""
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

            if default_step:
                step = 2.0 / (k + 2.0)
                k += 1
            else:
                step = max(0.0, min(1.0, (grad[i] / tr - 1.0) / max(grad[i] - 1.0, 1e-8)))
            pi = (1 - step) * pi + step * hot

    def adv_effective_dimension(self, eigenvalues, mu):
        N = len(eigenvalues)
        K = np.sum(np.isclose(eigenvalues, 0))

        Lambda = eigenvalues  # no lambda regularization

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

        p = []
        for i in range(K + 1, omega + 1):
            p.append(math.sqrt(Lambda[i - 1]) * ((1 + mu * lambda_sum) / lambda_sqrt_sum) - (mu * Lambda[i - 1]))
        p = np.array(p)

        d = K + np.sum(p / (mu * Lambda[K:omega] + p))
        return d
