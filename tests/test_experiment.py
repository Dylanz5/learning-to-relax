from __future__ import annotations

from pathlib import Path

import numpy as np

from ltr.data import load_losses
from ltr.experiment import Algo, live_sor, offline, paper_algos, run, spectrum_for
from ltr.learners import TsallisINF


def test_run_offline_csv_writes_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "loss.csv"
    csv_path.write_text("1.0,2.0,3.0,4.0\n4.0,3.0,2.0,1.0\n2.0,2.0,2.0,2.0\n", encoding="utf-8")
    bundle = load_losses(csv_path, similarity_kind="chain")["loss"]
    data = offline(bundle, path=str(csv_path), graph_desc="chain")
    grid = bundle.grid
    U, lam, L = bundle.U, bundle.lam, bundle.laplacian
    algos = paper_algos(grid, U, lam, L, T=bundle.T, which=("tinf", "exp3"))
    result = run(
        algos, data, seed=0, out_dir=tmp_path / "runs", name="csv_replay", plot=False, wandb_project=""
    )
    run_dir = result.run_dir
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "summary.csv").is_file()
    assert (run_dir / "run.txt").is_file()
    assert (run_dir / "arrays.npz").is_file()
    text = (run_dir / "run.txt").read_text(encoding="utf-8")
    assert "tinf" in text
    assert "chain" in text
    z = np.load(run_dir / "arrays.npz")
    assert "tinf_costs" in z
    assert "exp3_costs" in z
    assert z["tinf_costs"].shape == (bundle.T, bundle.trials)
    assert result.totals["tinf_total"] > 0.0


def test_run_accepts_factory_dict(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    T, K, trials = 5, 4, 2
    losses = rng.random((T, K, trials))
    from ltr.graphs import chain_laplacian

    L = chain_laplacian(K)
    lam, U = np.linalg.eigh(L)
    from ltr.data import LossBundle

    bundle = LossBundle(losses=losses, U=U, lam=lam, grid=np.arange(K, dtype=float), laplacian=L)
    data = offline(bundle, graph_desc="chain")
    algos = [
        {
            "name": "tinf",
            "factory": lambda: TsallisINF(bundle.grid, T=T),
        }
    ]
    result = run(
        algos, data, seed=1, out_dir=tmp_path / "runs", name="dict_algo", plot=False, wandb_project=""
    )
    assert result.costs["tinf"].shape == (T, trials)
    assert "TsallisINF" in (result.run_dir / "run.txt").read_text(encoding="utf-8")


def test_run_live_sor_tiny(tmp_path: Path) -> None:
    grid = np.linspace(1.0, 1.9, 4)
    data = live_sor(T=2, trials=1, grid=grid, domain_s=8, omega_count=2)
    U, lam, L = spectrum_for(grid, kind="chain")
    algos = paper_algos(grid, U, lam, L, T=2, which=("tinf",))
    result = run(
        algos, data, seed=0, out_dir=tmp_path / "runs", name="live_tiny", plot=False, wandb_project=""
    )
    assert result.costs["tinf"].shape == (2, 1)
    assert result.omega_costs is not None
    assert result.omega_costs.shape[-1] == 2
    assert "live SOR" in (result.run_dir / "run.txt").read_text(encoding="utf-8")
