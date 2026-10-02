"""Tsallis-Spectral alpha sweep on archived pitzDaily losses."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from ltr.experiment import REPLAY_HPS, offline_from_path, paper_algos, run

DATA = Path("data/fullinfo/pitzDaily.npy")
GRAPH = Path("graphs/pitzDaily_full_info_sim_graph.npy")
ALPHAS = (0.25, 0.5, 0.75, 0.9)
SEED = 0


def main() -> None:
    label, data = offline_from_path(
        DATA,
        path_similarity=GRAPH,
        similarity_is_laplacian=True,
    )
    b = data.bundle
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("plots/sweeps") / f"{stamp}_alpha"
    rows: list[dict] = []
    for alpha in ALPHAS:
        hps = dict(REPLAY_HPS)
        hps["tspec_alpha"] = float(alpha)
        algos = paper_algos(b.grid, b.U, b.lam, b.laplacian, T=b.T, hps=hps, which=("tspec",))
        tag = str(alpha).replace(".", "p")
        result = run(
            algos,
            data,
            seed=SEED,
            name=f"sweep_a{tag}_{label}",
            out_dir=root / "runs",
            notes={"graph": str(GRAPH), "alpha": alpha},
        )
        row = {
            "alpha": alpha,
            "tspec_total": result.totals["tspec_total"],
            "npz": str(result.run_dir / "arrays.npz"),
        }
        rows.append(row)
        print(f"[sweep] alpha={alpha:g}  tspec={row['tspec_total']:.1f}", flush=True)

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["alpha", "tspec_total", "npz"])
        w.writeheader()
        w.writerows(rows)
    (root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
