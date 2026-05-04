"""Similarity graphs over the discrete arm grid for Exp3-Spectral (graph Laplacian + eigh)."""

from __future__ import annotations

import numpy as np

from ltr.learners.exp3_spectral import chain_laplacian

#: Kinds understood by ``similarity_spectrum`` (extend here as you add graphs).
SIMILARITY_KINDS: tuple[str, ...] = ("chain", "ring")


def ring_laplacian(k: int) -> np.ndarray:
    """Unweighted cycle-graph Laplacian on k nodes."""
    k = int(k)
    if k <= 1:
        return np.zeros((k, k), dtype=float)
    L = np.zeros((k, k), dtype=float)
    for i in range(k):
        L[i, i] = 2.0
        L[i, (i - 1) % k] = -1.0
        L[i, (i + 1) % k] = -1.0
    return L


def graph_laplacian(num_arms: int, kind: str) -> np.ndarray:
    kind = kind.strip().lower()
    if kind == "chain":
        return chain_laplacian(num_arms)
    if kind == "ring":
        return ring_laplacian(num_arms)
    raise ValueError(
        f"unknown similarity_kind {kind!r}; expected one of {SIMILARITY_KINDS}",
    )


def similarity_spectrum(num_arms: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvectors U, eigenvalues lam) of the graph Laplacian for ``kind``."""
    L = graph_laplacian(num_arms, kind)
    lam, U = np.linalg.eigh(L)
    return U, lam
