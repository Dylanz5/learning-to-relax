"""Similarity graphs over the discrete arm grid for Exp3-Spectral (graph Laplacian + eigh)."""

from __future__ import annotations

import numpy as np

#: Kinds understood by ``similarity_spectrum`` (extend here as you add graphs).
SIMILARITY_KINDS: tuple[str, ...] = ("chain", "ring", "double_chain")

def create_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute the Laplacian matrix from adjacency matrix."""
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    return L


def chain_laplacian(k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return np.zeros((k, k), dtype=float)
    adj = np.zeros((k, k), dtype=float)
    for i in range(k):
        adj[i,i] = 1.0
        if i - 1 >= 0:
            adj[i, i - 1] = 1.0
        if i + 1 < k:
            adj[i, i + 1] = 1.0
    return create_laplacian(adj)


def double_chain_laplacian(k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return np.zeros((k, k), dtype=float)
    adj = np.zeros((k, k), dtype=float)
    for i in range(k):
        if i - 1 >= 0:
            adj[i, i - 1] = 1.0
        if i + 1 < k:
            adj[i, i + 1] = 1.0
        if i - 2 >= 0:
            adj[i, i - 2] = .5
        if i + 2 < k:
            adj[i, i + 2] = .5
    return create_laplacian(adj)


def ring_laplacian(k: int) -> np.ndarray:
    """Unweighted cycle-graph Laplacian on k nodes."""
    k = int(k)
    if k <= 1:
        return np.zeros((k, k), dtype=float)
    adj = np.zeros((k, k), dtype=float)
    for i in range(k):
        adj[i, (i - 1) % k] = 1.0
        adj[i, (i + 1) % k] = 1.0
    return create_laplacian(adj)


def graph_laplacian(num_arms: int, kind: str) -> np.ndarray:
    kind = kind.strip().lower()
    if kind == "chain":
        return chain_laplacian(num_arms)
    if kind == "double_chain":
        return double_chain_laplacian(num_arms)
    if kind == "ring":
        return ring_laplacian(num_arms)
    raise ValueError(
        f"unknown similarity_kind {kind!r}; expected one of {SIMILARITY_KINDS}",
    )


def similarity_spectrum(num_arms: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvectors U, eigenvalues lam) of the graph Laplacian for ``kind``."""
    L = graph_laplacian(num_arms, kind)
    lam, U = np.linalg.eigh(L)
    return U, lam, L
