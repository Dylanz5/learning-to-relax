from __future__ import annotations

import numpy as np
import pytest

from ltr.bench.config import LearningExperimentConfig
from ltr.bench.precomputed import load_precomputed_npz
from ltr.bench.similarity import SIMILARITY_KINDS, chain_laplacian, similarity_spectrum


def test_similarity_chain_matches_direct_eigh() -> None:
    k = 20
    U, lam = similarity_spectrum(k, "chain")
    L = chain_laplacian(k)
    np.testing.assert_allclose(U @ np.diag(lam) @ U.T, L, rtol=1e-12, atol=1e-12)


def test_similarity_kinds_nonempty() -> None:
    assert "chain" in SIMILARITY_KINDS
    assert "ring" in SIMILARITY_KINDS


def test_config_roundtrip_json(tmp_path) -> None:
    cfg = LearningExperimentConfig(similarity_kind="ring", run_name="test", trials=3)
    path = tmp_path / "c.json"
    cfg.to_json_file(path)
    loaded = LearningExperimentConfig.from_json_file(path)
    assert loaded == cfg


def test_config_roundtrip_ssor_pcg_solver(tmp_path) -> None:
    cfg = LearningExperimentConfig(similarity_kind="chain", solver="ssor_pcg", trials=2)
    path = tmp_path / "cg.json"
    cfg.to_json_file(path)
    assert LearningExperimentConfig.from_json_file(path) == cfg


def test_config_invalid_solver_raises() -> None:
    with pytest.raises(ValueError, match="solver"):
        LearningExperimentConfig(solver="gmres")


def test_load_precomputed_npz_roundtrip(tmp_path) -> None:
    K, T, trials = 5, 7, 3
    W = np.eye(K, dtype=float) * 0.0
    for i in range(K - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    losses = np.random.default_rng(0).random((T, K, trials))
    path = tmp_path / "b.npz"
    np.savez(path, losses_low=losses, path_similarity=W)
    bundles = load_precomputed_npz(path)
    assert "losses_low" in bundles
    b = bundles["losses_low"]
    assert b.losses.shape == (T, K, trials)
    assert b.U.shape == (K, K)
    assert b.laplacian.shape == (K, K)


def test_load_precomputed_npy_with_chain(tmp_path) -> None:
    K, T, trials = 4, 6, 2
    losses = np.random.default_rng(1).random((T, K, trials))
    path = tmp_path / "run.npy"
    np.save(path, losses)
    bundles = load_precomputed_npz(path, similarity_kind="chain")
    assert "run" in bundles
    assert bundles["run"].K == K
