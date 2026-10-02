"""Live SOR on a chain graph: Tsallis-INF / Exp3-Spectral / Tsallis-Spectral."""

from __future__ import annotations

import numpy as np

from ltr.experiment import LIVE_HPS, live_sor, paper_algos, run, spectrum_for

T = 100
TRIALS = 1
SEED = 0
K = 16
GRAPH = "chain"


def main() -> None:
    grid = np.linspace(1.0, 1.9, K)
    data = live_sor(T=T, trials=TRIALS, grid=grid, domain_s=12)
    U, lam, L = spectrum_for(grid, kind=GRAPH)
    algos = paper_algos(grid, U, lam, L, T=T, hps=LIVE_HPS)
    run(
        algos,
        data,
        seed=SEED,
        name="live_sor",
        notes={"graph": GRAPH},
    )


if __name__ == "__main__":
    main()
