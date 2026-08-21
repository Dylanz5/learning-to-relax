"""Report on Tsallis-INF diagnostics captured by ``scripts/learning.py``.

The learning driver writes, next to each run's plot:

* ``<stem>.npz`` containing (among the usual arrays):
    - ``tinf_diag``   structured scalar table, one row per predict/update cycle
    - ``tinf_k_hist`` (trials, T, K) per-round copies of ``k``
    - ``tinf_p_hist`` (trials, T, K) per-round sampling distributions ``p``
* ``<stem>.csv``  the same scalar table as plain text.

This script loads those and prints four diagnostics:

  (a) the 20 largest ``increment`` rows (the amounts added to ``k[arm]``);
  (b) a per-arm slope check comparing ``k_i(T)/T`` against
      ``mean observed loss of arm i - 1`` (these should track each other);
  (c) the rounds where ``scale_after/scale_before`` moved by more than a
      threshold (default 5%) — these line up with the vertical stripes in the
      ``*_tinf_loss_estimates.png`` heatmap;
  (d) for a chosen arm, its pull times and its ``p`` value at the pull, one
      step later, and midway to its next pull.

Note on time indexing: the ``t`` column is the learner's 1-indexed clock (the
value used for the ``eta = 2/sqrt(t)`` schedule that round). The corresponding
0-indexed round -- i.e. the column of the heatmap and the row of
``k_hist``/``p_hist`` -- is ``t - 1``.

Usage (from repo root)::

    python scripts/diag_report.py --npz plots/<stem>.npz --arm 12
    python scripts/diag_report.py --npz plots/<stem>.npz --trial 0 --arm 12 --top 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_SCALAR_FIELDS: list[tuple[str, type]] = [
    ("trial", int),
    ("t", int),
    ("arm", int),
    ("p_norm", float),
    ("prob_raw", float),
    ("loss", float),
    ("run_mean", float),
    ("increment", float),
    ("k_arm_before", float),
    ("k_arm_after", float),
    ("scale_before", float),
    ("scale_after", float),
    ("x", float),
    ("L_min", float),
    ("argmax_arm", int),
    ("p_max", float),
    ("entropy", float),
]


def _load_scalar_table(npz_path: Path | None, csv_path: Path | None) -> np.ndarray:
    """Load the structured diag table, preferring an explicit CSV then the npz."""
    if csv_path is not None and csv_path.exists():
        return _load_csv(csv_path)
    if npz_path is not None and npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            if "tinf_diag" in data.files:
                return data["tinf_diag"]
    # Last resort: derive the CSV path next to the npz.
    if npz_path is not None:
        guess = npz_path.with_suffix(".csv")
        if guess.exists():
            return _load_csv(guess)
    raise SystemExit("could not find a diag table (pass --csv or an --npz containing 'tinf_diag')")


def _load_csv(csv_path: Path) -> np.ndarray:
    dtype = np.dtype([(name, "i8" if typ is int else "f8") for name, typ in _SCALAR_FIELDS])
    rows: list[tuple] = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(
                tuple((int if typ is int else float)(r[name]) for name, typ in _SCALAR_FIELDS)
            )
    return np.array(rows, dtype=dtype)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))
    line = "  ".join(h.rjust(widths[j]) for j, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[j] for j in range(len(headers))))
    for row in rows:
        print("  ".join(cell.rjust(widths[j]) for j, cell in enumerate(row)))


def _fmt(x: float, nd: int = 6) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}g}"


def report_largest_increments(table: np.ndarray, top: int) -> None:
    print(f"\n(a) {top} largest `increment` rows (amount added to k[arm])")
    print("=" * 72)
    if table.size == 0:
        print("  (no rows)")
        return
    order = np.argsort(table["increment"])[::-1][:top]
    headers = [
        "trial", "t", "arm", "increment", "prob_raw", "p_norm",
        "loss", "k_before", "k_after", "scale_b", "scale_a",
    ]
    rows: list[list[str]] = []
    for i in order:
        r = table[i]
        rows.append([
            str(int(r["trial"])),
            str(int(r["t"])),
            str(int(r["arm"])),
            _fmt(float(r["increment"])),
            _fmt(float(r["prob_raw"])),
            _fmt(float(r["p_norm"])),
            _fmt(float(r["loss"]), 4),
            _fmt(float(r["k_arm_before"])),
            _fmt(float(r["k_arm_after"])),
            _fmt(float(r["scale_before"])),
            _fmt(float(r["scale_after"])),
        ])
    _print_table(headers, rows)


def report_slope_check(
    table: np.ndarray, k_hist: np.ndarray | None, trial: int
) -> None:
    print("\n(b) per-arm slope check: k_i(T)/T  vs  (mean observed loss of arm i - 1)")
    print("=" * 72)
    tt = table[table["trial"] == trial] if table.size else table
    if tt.size == 0:
        print(f"  (no rows for trial {trial})")
        return
    T = int(tt["t"].max())  # 1-indexed clock, so max t == number of rounds
    arms = np.unique(tt["arm"]).astype(int)
    # Final cumulative k per arm: prefer k_hist (authoritative), else the last
    # recorded k_arm_after seen for that arm.
    k_final: dict[int, float] = {}
    if k_hist is not None and k_hist.ndim == 3 and trial < k_hist.shape[0] and k_hist.shape[1] > 0:
        last = k_hist[trial, -1, :]
        for a in range(k_hist.shape[2]):
            k_final[a] = float(last[a])
    headers = ["arm", "pulls", "k_i(T)", "k_i(T)/T", "mean_loss", "mean_loss-1", "abs_diff"]
    rows: list[list[str]] = []
    for a in arms:
        mask = tt["arm"] == a
        n = int(np.count_nonzero(mask))
        mean_loss = float(np.mean(tt["loss"][mask])) if n else float("nan")
        if a in k_final:
            k_T = k_final[a]
        else:
            # fall back to the last k_arm_after logged for this arm
            after = tt["k_arm_after"][mask]
            k_T = float(after[-1]) if after.size else float("nan")
        slope = k_T / T if T > 0 else float("nan")
        target = mean_loss - 1.0
        rows.append([
            str(int(a)),
            str(n),
            _fmt(k_T),
            _fmt(slope),
            _fmt(mean_loss),
            _fmt(target),
            _fmt(abs(slope - target)),
        ])
    _print_table(headers, rows)
    print(f"  (T = {T} rounds; trial = {trial})")


def report_scale_jumps(table: np.ndarray, trial: int, threshold: float) -> None:
    pct = threshold * 100.0
    print(f"\n(c) rounds where scale changed by > {pct:.3g}% (scale_after vs scale_before)")
    print("=" * 72)
    tt = table[table["trial"] == trial] if table.size else table
    if tt.size == 0:
        print(f"  (no rows for trial {trial})")
        return
    sb = tt["scale_before"].astype(float)
    sa = tt["scale_after"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(sb != 0.0, (sa - sb) / sb, np.where(sa != sb, np.inf, 0.0))
    hits = np.nonzero(np.abs(rel) > threshold)[0]
    if hits.size == 0:
        print("  (none)")
        return
    headers = ["t", "round", "scale_before", "scale_after", "pct_change"]
    rows: list[list[str]] = []
    for i in hits:
        r = tt[i]
        rows.append([
            str(int(r["t"])),
            str(int(r["t"]) - 1),
            _fmt(float(r["scale_before"])),
            _fmt(float(r["scale_after"])),
            (f"{rel[i] * 100.0:+.2f}%" if np.isfinite(rel[i]) else "inf"),
        ])
    _print_table(headers, rows)
    print(f"  ({hits.size} rounds; compare `round` against the heatmap's x-axis columns)")


def report_arm_probabilities(
    table: np.ndarray, p_hist: np.ndarray | None, trial: int, arm: int
) -> None:
    print(f"\n(d) arm {arm}: pull times and p at pull / +1 step / midway to next pull")
    print("=" * 72)
    if p_hist is None or p_hist.ndim != 3 or trial >= p_hist.shape[0]:
        print("  (p_hist unavailable; need an --npz with 'tinf_p_hist')")
        return
    if arm < 0 or arm >= p_hist.shape[2]:
        print(f"  (arm {arm} out of range 0..{p_hist.shape[2] - 1})")
        return
    tt = table[table["trial"] == trial] if table.size else table
    pull_mask = tt["arm"] == arm
    pull_ts = np.sort(tt["t"][pull_mask].astype(int))
    if pull_ts.size == 0:
        print(f"  (arm {arm} was never pulled in trial {trial})")
        return
    p_arm = p_hist[trial, :, arm]
    T = p_arm.shape[0]
    pull_rounds = pull_ts - 1  # 0-indexed rounds into p_hist
    headers = ["t", "round", "p_at_pull", "p_next", "next_pull_t", "mid_round", "p_mid"]
    rows: list[list[str]] = []
    for j, r0 in enumerate(pull_rounds):
        p_at = float(p_arm[r0]) if 0 <= r0 < T else float("nan")
        p_next = float(p_arm[r0 + 1]) if 0 <= r0 + 1 < T else float("nan")
        if j + 1 < pull_rounds.size:
            r_next = int(pull_rounds[j + 1])
            mid = (int(r0) + r_next) // 2
            p_mid = float(p_arm[mid]) if 0 <= mid < T else float("nan")
            next_t = str(int(pull_ts[j + 1]))
            mid_s = str(mid)
        else:
            p_mid = float("nan")
            next_t = "-"
            mid_s = "-"
        rows.append([
            str(int(pull_ts[j])),
            str(int(r0)),
            _fmt(p_at),
            _fmt(p_next),
            next_t,
            mid_s,
            _fmt(p_mid),
        ])
    _print_table(headers, rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=str, default=None, help="Run .npz (holds tinf_diag / tinf_k_hist / tinf_p_hist).")
    p.add_argument("--csv", type=str, default=None, help="Optional diag CSV (defaults to the npz's .csv sibling).")
    p.add_argument("--trial", type=int, default=0, help="Trial index for parts (b)-(d).")
    p.add_argument("--arm", type=int, default=None, help="Arm index for part (d); omit to skip it.")
    p.add_argument("--top", type=int, default=20, help="How many largest-increment rows to show in (a).")
    p.add_argument("--scale-threshold", type=float, default=0.05, help="Relative scale-change threshold for (c).")
    args = p.parse_args()

    if args.npz is None and args.csv is None:
        raise SystemExit("pass at least one of --npz or --csv")

    npz_path = Path(args.npz) if args.npz else None
    csv_path = Path(args.csv) if args.csv else None

    table = _load_scalar_table(npz_path, csv_path)

    k_hist: np.ndarray | None = None
    p_hist: np.ndarray | None = None
    if npz_path is not None and npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            if "tinf_k_hist" in data.files:
                k_hist = data["tinf_k_hist"]
            if "tinf_p_hist" in data.files:
                p_hist = data["tinf_p_hist"]

    print(f"loaded {table.shape[0]} diag rows"
          + (f" from {csv_path}" if (csv_path and csv_path.exists()) else (f" from {npz_path}" if npz_path else "")))

    report_largest_increments(table, args.top)
    report_slope_check(table, k_hist, args.trial)
    report_scale_jumps(table, args.trial, args.scale_threshold)
    if args.arm is not None:
        report_arm_probabilities(table, p_hist, args.trial, args.arm)
    else:
        print("\n(d) skipped: pass --arm <index> to inspect a specific arm's p trajectory")


if __name__ == "__main__":
    main()
