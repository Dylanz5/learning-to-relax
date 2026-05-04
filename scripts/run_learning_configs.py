#!/usr/bin/env python3
"""Run ``learning.py`` once per JSON config — optionally in parallel subprocesses.

Put launcher flags first, then config paths, then any flags for ``learning.py``::

  python scripts/run_learning_configs.py --parallel 2 \\
      configs/bench/learning_chain_default.json configs/bench/learning_ring_default.json \\
      --trials 5 --benchmark-solver

Unknown arguments after the known ones are forwarded to ``learning.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum concurrent subprocesses (default 1 = serial).",
    )
    p.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="JSON files (LearningExperimentConfig).",
    )
    args, forwarded = p.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    learning_py = repo_root / "scripts" / "learning.py"

    def run_one(cfg_path: Path) -> tuple[Path, int]:
        cmd = [
            sys.executable,
            str(learning_py),
            "--config",
            str(cfg_path.resolve()),
            *forwarded,
        ]
        print(f"[run_learning_configs] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=repo_root)
        return cfg_path, proc.returncode

    max_p = max(1, args.parallel)
    results: list[tuple[Path, int]] = []
    if max_p == 1:
        for c in args.configs:
            results.append(run_one(c))
    else:
        with ThreadPoolExecutor(max_workers=max_p) as ex:
            futs = {ex.submit(run_one, c): c for c in args.configs}
            for fut in as_completed(futs):
                results.append(fut.result())

    bad = [r for r in results if r[1] != 0]
    if bad:
        for path, code in bad:
            print(f"[run_learning_configs] FAILED {path} exit {code}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
