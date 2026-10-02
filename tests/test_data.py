from __future__ import annotations

import numpy as np

from ltr.data import load_losses
from ltr.graphs import SIMILARITY_KINDS, chain_laplacian, similarity_spectrum


def test_similarity_chain_matches_direct_eigh() -> None:
    k = 20
    U, lam, Lspec = similarity_spectrum(k, "chain")
    L = chain_laplacian(k)
    np.testing.assert_allclose(Lspec, L, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(U @ np.diag(lam) @ U.T, L, rtol=1e-12, atol=1e-12)


def test_similarity_kinds_nonempty() -> None:
    assert "chain" in SIMILARITY_KINDS
    assert "ring" in SIMILARITY_KINDS


def test_load_npz_roundtrip(tmp_path) -> None:
    K, T, trials = 5, 7, 3
    W = np.zeros((K, K), dtype=float)
    for i in range(K - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    losses = np.random.default_rng(0).random((T, K, trials))
    path = tmp_path / "b.npz"
    np.savez(path, losses_low=losses, path_similarity=W)
    bundles = load_losses(path)
    assert "losses_low" in bundles
    b = bundles["losses_low"]
    assert b.losses.shape == (T, K, trials)
    assert b.U.shape == (K, K)


def test_load_npy_with_chain(tmp_path) -> None:
    K, T, trials = 4, 6, 2
    losses = np.random.default_rng(1).random((T, K, trials))
    path = tmp_path / "run.npy"
    np.save(path, losses)
    bundles = load_losses(path, similarity_kind="chain")
    assert "run" in bundles
    assert bundles["run"].K == K


def test_load_csv_losses(tmp_path) -> None:
    path = tmp_path / "loss.csv"
    path.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n", encoding="utf-8")
    bundles = load_losses(path, similarity_kind="chain")
    b = bundles["loss"]
    assert b.losses.shape == (2, 3, 1)
    assert b.K == 3
