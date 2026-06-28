"""Load archived full-information loss tensors for offline bandit replays.

Expected layout per experiment: losses ``(T, K, trials)`` where entry
``(t, i, k)`` is the loss (e.g. iteration count) for arm *i* at time *t* in
trial *k*. A nonnegative symmetric **edge-weight** matrix ``W`` (path similarity /
adjacency) yields a combinatorial Laplacian ``L = diag(sum(W)) - W`` for
Exp3-Spectral.

``.npz`` key conventions (any one loss array can be omitted if you only need one regime):

- ``losses_low`` / ``losses_high`` — preferred names matching ``learning.py``
- ``losses_a`` / ``losses_b`` — generic pair
- ``losses`` — single tensor

Graph (exactly one of):

- ``path_similarity`` / ``adjacency`` / ``similarity`` — nonnegative weights ``W_ij``
- ``laplacian`` — use directly (symmetrized)

Optional:

- ``grid`` — shape ``(K,)`` arm labels (default ``0..K-1``)

You can also pass a single ``.npy`` file whose only array has shape ``(T, K, trials)``.
Then you must supply a graph via ``--path-similarity`` / ``--similarity-kind`` on the CLI
(see ``scripts/run_learning_precomputed.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ltr.bench.similarity import SIMILARITY_KINDS, graph_laplacian


@dataclass
class PrecomputedLossBundle:
    """One regime: full loss tensor plus graph spectrum for Exp3."""

    losses: np.ndarray  # (T, K, trials), float
    U: np.ndarray
    lam: np.ndarray
    grid: np.ndarray
    laplacian: np.ndarray

    @property
    def T(self) -> int:
        return int(self.losses.shape[0])

    @property
    def K(self) -> int:
        return int(self.losses.shape[1])

    @property
    def trials(self) -> int:
        return int(self.losses.shape[2])


def _symmetrize_nonneg(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    W = np.maximum(W, 0.0)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def laplacian_from_edge_weights(W: np.ndarray) -> np.ndarray:
    """Combinatorial Laplacian ``D - W`` with zero diagonal on ``W``."""
    W = _symmetrize_nonneg(W)
    d = W.sum(axis=1)
    return np.diag(d) - W


def laplacian_from_bundle_arrays(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "laplacian" in data:
        L = np.asarray(data["laplacian"], dtype=float)
        return 0.5 * (L + L.T)
    for key in ("path_similarity", "adjacency", "similarity"):
        if key in data:
            return laplacian_from_edge_weights(np.asarray(data[key], dtype=float))
    raise KeyError(
        "npz must contain one of: laplacian, path_similarity, adjacency, similarity",
    )


def _resolve_laplacian(
    K: int,
    *,
    path_matrix: str | Path | None,
    similarity_kind: str | None,
    matrix_is_laplacian: bool,
) -> np.ndarray:
    """Build ``K×K`` Laplacian from a sidecar matrix file and/or a named graph."""
    if path_matrix is not None:
        M = np.load(path_matrix, allow_pickle=False)
        M = np.asarray(M, dtype=float)
        if M.shape != (K, K):
            raise ValueError(
                f"sidecar matrix has shape {M.shape}, expected ({K}, {K}) for this loss tensor",
            )
        if matrix_is_laplacian:
            return 0.5 * (M + M.T)
        return laplacian_from_edge_weights(M)
    if similarity_kind is not None:
        return graph_laplacian(K, similarity_kind)
    raise ValueError(
        "Missing graph: use a .npz with laplacian/path_similarity, or pass path_matrix= "
        f"(.npy weights or Laplacian) or similarity_kind= one of {SIMILARITY_KINDS}",
    )


def _pick_loss_arrays(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in ("losses_low", "losses_high", "losses_a", "losses_b", "losses"):
        if key in data:
            out[key] = np.asarray(data[key], dtype=float)
    return out


def _bundle_from_losses(
    name: str,
    losses: np.ndarray,
    L: np.ndarray,
    grid: np.ndarray | None,
) -> tuple[str, PrecomputedLossBundle]:
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 3:
        raise ValueError(f"{name} must have shape (T, K, trials), got {losses.shape}")
    _, k, _ = losses.shape
    if L.shape != (k, k):
        raise ValueError(f"{name} has K={k} but Laplacian is {L.shape}")
    lam, U = np.linalg.eigh(L)
    if grid is None:
        g = np.arange(k, dtype=float)
    else:
        g = np.asarray(grid, dtype=float).reshape(-1)
        if g.shape[0] != k:
            raise ValueError(f"grid has length {g.shape[0]}, expected K={k}")
    return name, PrecomputedLossBundle(
        losses=losses,
        U=U,
        lam=lam,
        grid=g,
        laplacian=L,
    )


def load_precomputed_npz(
    path: str | Path,
    *,
    path_similarity: str | Path | None = None,
    similarity_kind: str | None = None,
    similarity_is_laplacian: bool = False,
) -> dict[str, PrecomputedLossBundle]:
    """Load loss tensors from a ``.npz`` or a single ``.npy`` `(T, K, trials)` array.

    For ``.npz``, the graph normally lives in the same archive. If those keys are
    absent, ``path_similarity`` / ``similarity_kind`` are used (same as for ``.npy``).

    For a plain ``.npy``, you must pass ``path_similarity`` (a ``K×K`` ``.npy``) and/or
    ``similarity_kind`` (e.g. ``\"chain\"``). The returned dict has one entry named
    like the file stem (e.g. ``boxTurb32``).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            loss_map = _pick_loss_arrays(data)
            if not loss_map:
                raise KeyError(
                    f"{path}: no loss array found; expected one of "
                    "losses_low, losses_high, losses_a, losses_b, losses",
                )
            grid_np = np.asarray(data["grid"], dtype=float).reshape(-1) if "grid" in data else None
            k0 = int(next(iter(loss_map.values())).shape[1])
            try:
                L = laplacian_from_bundle_arrays(data)
            except KeyError:
                L = _resolve_laplacian(
                    k0,
                    path_matrix=path_similarity,
                    similarity_kind=similarity_kind,
                    matrix_is_laplacian=similarity_is_laplacian,
                )
            bundles: dict[str, PrecomputedLossBundle] = {}
            for name, losses in loss_map.items():
                _, b = _bundle_from_losses(name, losses, L, grid_np)
                bundles[name] = b
            return bundles

    if suffix in (".npy",):
        losses = np.load(path, allow_pickle=False)
        losses = np.asarray(losses, dtype=float)
        if losses.ndim != 3:
            raise ValueError(
                f"{path}: expected a single array of shape (T, K, trials), got {losses.shape}",
            )
        K = int(losses.shape[1])
        L = _resolve_laplacian(
            K,
            path_matrix=path_similarity,
            similarity_kind=similarity_kind,
            matrix_is_laplacian=similarity_is_laplacian,
        )
        name, bundle = _bundle_from_losses(path.stem, losses, L, None)
        return {name: bundle}

    raise ValueError(
        f"{path}: unsupported extension {suffix!r}; use .npz (archive) or .npy (one loss tensor)",
    )
