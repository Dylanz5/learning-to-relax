"""Replay Tsallis-INF / Exp3-Spectral / Tsallis-Spectral on pitzDaily losses."""

from __future__ import annotations

from pathlib import Path

from ltr.experiment import REPLAY_HPS, offline_from_path, paper_algos, run

DATA = Path("data/fullinfo/pitzDaily.npy")
GRAPH = Path("graphs/pitzDaily_full_info_sim_graph.npy")
SEED = 0


def main() -> None:
    label, data = offline_from_path(
        DATA,
        path_similarity=GRAPH,
        similarity_is_laplacian=True,
    )
    b = data.bundle
    algos = paper_algos(b.grid, b.U, b.lam, b.laplacian, T=b.T, hps=REPLAY_HPS)
    run(
        algos,
        data,
        seed=SEED,
        name=f"replay_{label}",
        notes={"graph": str(GRAPH)},
    )


if __name__ == "__main__":
    main()
