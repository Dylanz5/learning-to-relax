"""Live SOR sweep over the number of ω-grid points (K)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ltr.experiment import LIVE_HPS, live_sor, paper_algos, run, spectrum_for
from ltr.plot import plot_regret_vs_k

KS = (8, 16, 32)
T = 40
TRIALS = 1
SEED = 0
GRAPH = "chain"


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("plots/sweeps") / f"{stamp}_k"
    rows: list[dict] = []
    for k in KS:
        grid = np.linspace(1.0, 1.9, int(k))
        data = live_sor(T=T, trials=TRIALS, grid=grid, domain_s=12)
        U, lam, L = spectrum_for(grid, kind=GRAPH)
        algos = paper_algos(grid, U, lam, L, T=T, hps=LIVE_HPS)
        result = run(
            algos,
            data,
            seed=SEED,
            name=f"sweep_K{k}",
            out_dir=root / "runs",
            notes={"graph": GRAPH, "K": k},
        )
        row = {
            "K": k,
            "tinf_total": result.totals["tinf_total"],
            "exp3_total": result.totals["exp3_total"],
            "tspec_total": result.totals["tspec_total"],
            "best_fixed_total": result.totals["best_fixed_total"],
            "npz": str(result.run_dir / "arrays.npz"),
        }
        rows.append(row)
        print(
            f"[sweep] K={k}  tspec={row['tspec_total']:.1f}  "
            f"tinf={row['tinf_total']:.1f}  exp3={row['exp3_total']:.1f}",
            flush=True,
        )

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["K", "tinf_total", "exp3_total", "tspec_total", "best_fixed_total", "npz"],
        )
        w.writeheader()
        w.writerows(rows)
    (root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ks_arr = np.array([r["K"] for r in rows], dtype=float)
    series = {
        "Tsallis-INF": np.array([r["tinf_total"] for r in rows]),
        "Exp3-Spectral": np.array([r["exp3_total"] for r in rows]),
        "Tsallis-Spectral": np.array([r["tspec_total"] for r in rows]),
        "best fixed ω": np.array([r["best_fixed_total"] for r in rows]),
    }
    plot_regret_vs_k(ks_arr, series, root / "regret_vs_k.png", title="regret vs number of configs")
    print(f"[sweep] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
