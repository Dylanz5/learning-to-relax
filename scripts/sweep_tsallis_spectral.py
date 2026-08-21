#!/usr/bin/env python
"""Hyperparameter sweep: can Tsallis-Spectral beat Tsallis-INF on pitzDaily?

For each point in a grid of Tsallis-Spectral (Algorithm 2) hyperparameters
(eta, gamma, mu, smoothness, alpha) x seeds, this launches
``scripts/run_learning_precomputed.py`` as a subprocess (so every run's plots +
``.npz`` are saved as usual), then reads the saved per-round costs and aggregates
the *final cumulative loss* of each learner into a ranked ``results.csv`` /
``results.json``.

Key framing: Tsallis-INF does not depend on the Tsallis-Spectral knobs, so for a
fixed seed its total is a constant baseline that every Tsallis-Spectral config is
compared against. "improvement" = (tinf_total - tspec_total); positive means
Tsallis-Spectral wins.

Usage (from repo root, inside the project's conda env)::

    # default grid, 4 parallel workers (small pitzDaily)
    python scripts/sweep_tsallis_spectral.py --jobs 4

    # pitzDailyLarge (or any other tensor) via --data / --path-similarity
    python scripts/sweep_tsallis_spectral.py \\
        --data data/fullinfo/pitzDailyLarge_wallclock_even.npy \\
        --path-similarity graphs/pitzDailyLarge_laplacian_hand_crafted.npy \\
        --jobs 8

    # custom grid
    python scripts/sweep_tsallis_spectral.py \\
        --etas 0.05,0.1,0.3 --gammas 0.01,0.1 --mus 0.001,0.01,0.1 \\
        --smooths 0,1 --alphas 0.25,0.5,0.75 --seeds 0,1 --jobs 6

    # just print the commands, don't run anything
    python scripts/sweep_tsallis_spectral.py --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO / "scripts" / "run_learning_precomputed.py"
DATA = REPO / "data" / "fullinfo" / "pitzDaily.npy"
GRAPH = REPO / "graphs" / "pitzDaily_full_info_sim_graph.npy"

DEFAULT_GRID = {
    "eta": [.05, .1, .2],
    "gamma": [0],
    "mu": [.0001, .001],
    "smoothness": [0.0],
    "alpha": [.9],
}
DEFAULT_SEEDS = [0]

# ``run_learning_precomputed.py`` prints: "[precomputed] saved arrays to <path>".
NPZ_RE = re.compile(r"saved arrays to (\S+\.npz)")


def _fmt(x: float) -> str:
    """Filename-safe number: 0.05 -> 0p05, -1 -> m1."""
    return f"{x:g}".replace(".", "p").replace("-", "m")


def cfg_tag(c: dict) -> str:
    return (
        f"et{_fmt(c['eta'])}_g{_fmt(c['gamma'])}_mu{_fmt(c['mu'])}"
        f"_sm{_fmt(c['smoothness'])}_a{_fmt(c['alpha'])}_s{c['seed']}"
    )


def build_command(
    c: dict,
    cfg_path: Path,
    graph_args: list[str],
    runs_dir: Path,
    data_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(RUN_SCRIPT),
        "--data", str(data_path),
        *graph_args,
        "--config", str(cfg_path),
        "--seed", str(c["seed"]),
        "--save-loss-npz",
        "--baselines", "5",
        "--plots-dir", str(runs_dir),
    ]


def write_config(c: dict, sweep_dir: Path) -> Path:
    tag = cfg_tag(c)
    cfg_path = sweep_dir / "configs" / f"{tag}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(
            {
                "run_name": f"sweep_{tag}",
                "tsallis_spectral_eta": c["eta"],
                "tsallis_spectral_gamma": c["gamma"],
                "tsallis_spectral_mu": c["mu"],
                "tsallis_spectral_smoothness": c["smoothness"],
                "tsallis_spectral_alpha": c["alpha"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return cfg_path


def metrics_from_npz(npz_path: Path) -> dict:
    z = np.load(npz_path)
    tinf = np.asarray(z["tinf_costs"], dtype=float)  # (T, trials)
    tspec = np.asarray(z["tspec_costs"], dtype=float)  # (T, trials)
    tinf_tot = tinf.sum(axis=0)  # per-trial totals -> (trials,)
    tspec_tot = tspec.sum(axis=0)
    diff = tinf_tot - tspec_tot  # >0 => tspec better
    n = diff.size
    ddof = 1 if n > 1 else 0
    diff_std = float(diff.std(ddof=ddof))
    # Paired t-statistic on the per-trial differences (rough significance signal).
    tstat = float(diff.mean() / (diff_std / sqrt(n))) if diff_std > 0 and n > 1 else float("nan")
    # Best fixed-arm baseline (context only), if baselines were saved.
    best_base = float("nan")
    if "omega_costs" in z:
        base_tot = np.asarray(z["omega_costs"], dtype=float).sum(axis=0).mean(axis=0)  # (B,)
        if base_tot.size:
            best_base = float(base_tot.min())
    return {
        "trials": int(n),
        "tinf_total_mean": float(tinf_tot.mean()),
        "tinf_total_std": float(tinf_tot.std(ddof=ddof)),
        "tspec_total_mean": float(tspec_tot.mean()),
        "tspec_total_std": float(tspec_tot.std(ddof=ddof)),
        "improvement_mean": float(diff.mean()),  # tinf - tspec
        "improvement_std": diff_std,
        "improvement_pct": float(100.0 * diff.mean() / tinf_tot.mean()) if tinf_tot.mean() else float("nan"),
        "tspec_win_rate": float((tspec_tot < tinf_tot).mean()),
        "paired_tstat": tstat,
        "best_baseline_total": best_base,
        "npz": str(npz_path),
    }


def run_one(
    c: dict,
    sweep_dir: Path,
    graph_args: list[str],
    runs_dir: Path,
    data_path: Path,
) -> dict:
    tag = cfg_tag(c)
    cfg_path = write_config(c, sweep_dir)
    cmd = build_command(c, cfg_path, graph_args, runs_dir, data_path)
    log_path = sweep_dir / "logs" / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Pin native thread pools so parallel workers don't oversubscribe cores.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    log_path.write_text(proc.stdout + "\n===STDERR===\n" + proc.stderr, encoding="utf-8")

    row = {**c, "tag": tag, "seconds": round(elapsed, 1), "status": "ok"}
    if proc.returncode != 0:
        row["status"] = f"FAILED(rc={proc.returncode})"
        return row
    m = NPZ_RE.search(proc.stdout)
    if not m:
        row["status"] = "no_npz_in_output"
        return row
    npz_path = Path(m.group(1))
    if not npz_path.is_absolute():
        npz_path = REPO / npz_path
    try:
        row.update(metrics_from_npz(npz_path))
    except Exception as exc:  # noqa: BLE001 - record and continue the sweep
        row["status"] = f"metric_error: {exc}"
    return row


CSV_COLS = [
    "tag", "eta", "gamma", "mu", "smoothness", "alpha", "seed", "status", "seconds",
    "trials", "tspec_total_mean", "tspec_total_std", "tinf_total_mean", "tinf_total_std",
    "improvement_mean", "improvement_std", "improvement_pct", "tspec_win_rate",
    "paired_tstat", "best_baseline_total", "npz",
]


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_list(s: str, typ) -> list:
    return [typ(x) for x in s.split(",") if x.strip() != ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etas", type=str, default=None, help="comma list, e.g. 0.05,0.1,0.3")
    ap.add_argument("--gammas", type=str, default=None)
    ap.add_argument("--mus", type=str, default=None)
    ap.add_argument("--smooths", type=str, default=None)
    ap.add_argument("--alphas", type=str, default=None, help="Tsallis entropy params in (0,1), e.g. 0.25,0.5,0.75")
    ap.add_argument("--seeds", type=str, default=None, help="comma list of int seeds")
    ap.add_argument(
        "--data",
        type=str,
        default=str(DATA),
        help="Loss tensor .npy with shape (T, K, trials). Default: small pitzDaily.npy.",
    )
    ap.add_argument(
        "--similarity-kind",
        type=str,
        default=None,
        choices=["chain", "ring", "double_chain"],
        help="Use a built-in graph (e.g. double_chain = 'two chains'); overrides --path-similarity.",
    )
    ap.add_argument(
        "--path-similarity",
        type=str,
        default=str(GRAPH),
        help="K x K .npy graph file (default: pitzDaily full-info Laplacian). Ignored if --similarity-kind is set.",
    )
    ap.add_argument(
        "--similarity-is-laplacian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat --path-similarity as a Laplacian (our saved graph files are). Use --no-similarity-is-laplacian for edge weights.",
    )
    ap.add_argument("--jobs", type=int, default=4, help="parallel subprocesses")
    ap.add_argument("--limit", type=int, default=0, help="cap number of configs (0 = all)")
    ap.add_argument("--out", type=str, default=None, help="sweep output dir (default plots/sweeps/<ts>)")
    ap.add_argument("--dry-run", action="store_true", help="write configs + commands.txt, run nothing")
    args = ap.parse_args()

    grid = {
        "eta": parse_list(args.etas, float) if args.etas else DEFAULT_GRID["eta"],
        "gamma": parse_list(args.gammas, float) if args.gammas else DEFAULT_GRID["gamma"],
        "mu": parse_list(args.mus, float) if args.mus else DEFAULT_GRID["mu"],
        "smoothness": parse_list(args.smooths, float) if args.smooths else DEFAULT_GRID["smoothness"],
        "alpha": parse_list(args.alphas, float) if args.alphas else DEFAULT_GRID["alpha"],
    }
    seeds = parse_list(args.seeds, int) if args.seeds else DEFAULT_SEEDS

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (REPO / data_path).resolve()
    if not data_path.exists():
        raise SystemExit(f"missing data file: {data_path}")
    data_shape = np.load(data_path, mmap_mode="r").shape
    if len(data_shape) != 3:
        raise SystemExit(f"expected loss tensor (T, K, trials), got shape {data_shape}")
    T, K, trials = (int(x) for x in data_shape)
    data_desc = data_path.stem

    # Resolve the graph source: built-in kind (e.g. double_chain = "two chains")
    # or a K x K .npy file (default: the pitzDaily full-info Laplacian).
    if args.similarity_kind:
        graph_args = ["--similarity-kind", args.similarity_kind]
        graph_desc = args.similarity_kind
    else:
        graph_path = Path(args.path_similarity)
        if not graph_path.is_absolute():
            graph_path = (REPO / graph_path).resolve()
        if not graph_path.exists():
            raise SystemExit(f"missing graph file: {graph_path}")
        graph_args = ["--path-similarity", str(graph_path)]
        if args.similarity_is_laplacian:
            graph_args.append("--similarity-is-laplacian")
        graph_desc = graph_path.stem

    configs = [
        {"eta": e, "gamma": g, "mu": m, "smoothness": s, "alpha": a, "seed": sd}
        for e, g, m, s, a in itertools.product(
            grid["eta"], grid["gamma"], grid["mu"], grid["smoothness"], grid["alpha"]
        )
        for sd in seeds
    ]
    if args.limit > 0:
        configs = configs[: args.limit]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = (
        Path(args.out)
        if args.out
        else (REPO / "plots" / "sweeps" / f"{ts}_tspec_{data_desc}_{graph_desc}")
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    # Each run's per-config artifact folder goes here, inside the sweep folder.
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Record every command up front so the sweep is reproducible by hand.
    cmd_lines = []
    for c in configs:
        cfg_path = write_config(c, sweep_dir)
        cmd_lines.append(" ".join(build_command(c, cfg_path, graph_args, runs_dir, data_path)))
    (sweep_dir / "commands.txt").write_text("\n".join(cmd_lines) + "\n", encoding="utf-8")

    print(f"[sweep] {len(configs)} configs -> {sweep_dir}", flush=True)
    print(f"[sweep] data:  {data_path}  shape=(T={T}, K={K}, trials={trials})", flush=True)
    print(f"[sweep] graph: {graph_desc}  ({' '.join(graph_args)})", flush=True)
    print(f"[sweep] grid: {grid} seeds={seeds}", flush=True)
    # Rough cost model: small-pitz (K=33,T=2440,trials=10) ~2.5min. Scale by K^2 * T * trials.
    # Tsallis-Spectral computes a full K x K inverse per round (bonus), so it is a
    # bit heavier than Exp3; treat this as a lower bound.
    ref_cost = 33 ** 2 * 2440 * 10
    run_min = 2.5 * ((K ** 2) * T * trials) / ref_cost
    est_min = len(configs) * run_min / max(1, args.jobs)
    print(
        f"[sweep] ~{est_min:.0f} min wall estimate at ~{run_min:.1f}min/run "
        f"(scaled from K=33 baseline, lower bound), jobs={args.jobs}",
        flush=True,
    )

    if args.dry_run:
        print(f"[sweep] dry-run: wrote {sweep_dir/'commands.txt'} and configs/, running nothing.", flush=True)
        return

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = {
            ex.submit(run_one, c, sweep_dir, graph_args, runs_dir, data_path): c
            for c in configs
        }
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            if row.get("status") == "ok":
                print(
                    f"[sweep] {done}/{len(configs)} {row['tag']}: "
                    f"tspec={row['tspec_total_mean']:.1f} vs tinf={row['tinf_total_mean']:.1f} "
                    f"({row['improvement_pct']:+.1f}%  win={row['tspec_win_rate']:.0%})  "
                    f"[{row['seconds']:.0f}s]",
                    flush=True,
                )
            else:
                print(f"[sweep] {done}/{len(configs)} {row['tag']}: {row['status']} [{row['seconds']:.0f}s]", flush=True)
            # Persist incrementally so partial progress is never lost.
            write_csv(rows, sweep_dir / "results.csv")
            (sweep_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["tspec_total_mean"])  # lower total loss is better
    write_csv(rows, sweep_dir / "results.csv")
    (sweep_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n===== SWEEP SUMMARY (best Tsallis-Spectral configs by total loss) =====", flush=True)
    if ok:
        tinf_ref = ok[0]["tinf_total_mean"]
        print(f"Tsallis-INF baseline total (seed set): ~{tinf_ref:.1f}\n", flush=True)
        header = f"{'rank':>4}  {'tspec_total':>12}  {'improv%':>8}  {'win':>5}  {'tstat':>6}  config"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for i, r in enumerate(ok[:15], 1):
            print(
                f"{i:>4}  {r['tspec_total_mean']:>12.1f}  {r['improvement_pct']:>+7.1f}%  "
                f"{r['tspec_win_rate']:>4.0%}  {r['paired_tstat']:>6.2f}  "
                f"eta={r['eta']} gamma={r['gamma']} mu={r['mu']} sm={r['smoothness']} "
                f"alpha={r['alpha']} seed={r['seed']}",
                flush=True,
            )
        winners = [r for r in ok if r["improvement_mean"] > 0]
        print(
            f"\n{len(winners)}/{len(ok)} configs beat Tsallis-INF on mean total loss.",
            flush=True,
        )
        if winners:
            b = winners[0] if winners[0] is ok[0] else min(winners, key=lambda r: r["tspec_total_mean"])
            print(
                f"Best: {b['tag']}  ->  {b['improvement_pct']:+.1f}% vs Tsallis  "
                f"(win rate {b['tspec_win_rate']:.0%}, paired t={b['paired_tstat']:.2f})",
                flush=True,
            )
    else:
        print("No successful runs. Check logs in", sweep_dir / "logs", flush=True)

    failed = [r for r in rows if r.get("status") != "ok"]
    if failed:
        print(f"\n{len(failed)} runs did not produce metrics; see {sweep_dir/'logs'}.", flush=True)
    print(f"\n[sweep] results.csv + results.json written to {sweep_dir}", flush=True)


if __name__ == "__main__":
    main()
