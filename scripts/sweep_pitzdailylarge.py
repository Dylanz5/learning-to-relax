#!/usr/bin/env python
"""Hyperparameter sweep for pitzDailyLarge (409 arms).

Thin wrapper around ``scripts/sweep_pitzdaily.py`` with defaults aimed at the
large sweep.pkl conversion:

  data:  data/fullinfo/pitzDailyLarge_wallclock_even.npy   # (2440, 409, 1)
  graph: graphs/pitzDailyLarge_laplacian_hand_crafted.npy  # (409, 409)

Override with the same flags as the base sweep (``--data``, ``--path-similarity``,
``--etas``, ``--jobs``, ...). Useful aliases::

    # hand-crafted graph (default)
    python scripts/sweep_pitzdailylarge.py --jobs 8

    # per-t full-info-style Laplacian instead
    python scripts/sweep_pitzdailylarge.py \\
        --path-similarity graphs/pitzDailyLarge_laplacian_per_t_full.npy --jobs 8

    # built-in double_chain on the large arm set
    python scripts/sweep_pitzdailylarge.py --similarity-kind double_chain --jobs 8

    # dry-run / tiny probe
    python scripts/sweep_pitzdailylarge.py --dry-run
    python scripts/sweep_pitzdailylarge.py --limit 1 --jobs 1

Note: Exp3-Spectral cost scales roughly with K^2 per round. At K=409 this is
~150x the small pitzDaily (K=33) cost per trial; with trials=1 that still means
tens of minutes per config. Start with ``--limit 1`` to calibrate wall time.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = "data/fullinfo/pitzDailyLarge_wallclock_even.npy"
DEFAULT_GRAPH = "graphs/pitzDailyLarge_laplacian_hand_crafted.npy"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Inject defaults only when the user didn't already pass the flag.
    flags = set()
    for a in argv:
        if a.startswith("--"):
            flags.add(a.split("=", 1)[0])

    if "--data" not in flags:
        argv = ["--data", DEFAULT_DATA, *argv]
    if "--path-similarity" not in flags and "--similarity-kind" not in flags:
        argv = ["--path-similarity", DEFAULT_GRAPH, *argv]

    # Re-enter the shared sweep entrypoint with the injected defaults.
    sys.path.insert(0, str(REPO / "scripts"))
    import sweep_pitzdaily as base  # noqa: WPS433 - intentional local import

    # Replace argv so argparse inside the base sweep sees our defaults.
    old = sys.argv
    try:
        sys.argv = [old[0], *argv]
        base.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
