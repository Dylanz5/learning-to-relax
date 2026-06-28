"""Summarize and optionally plot arm losses from a precomputed ``(T, K, trials)`` tensor.

Examples::

    python scripts/inspect_precomputed_losses.py data/fullinfo/boxTurb32.npy
    python scripts/inspect_precomputed_losses.py data/bundle.npz --key losses_low
    python scripts/inspect_precomputed_losses.py data/fullinfo/boxTurb32.npy --plot plots/arm_loss_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_tensor(path: Path, key: str | None) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            names = list(z.files)
            if key is not None:
                if key not in z:
                    raise SystemExit(f"{path}: no array {key!r}; keys: {names}")
                arr = np.asarray(z[key], dtype=float)
                label = key
            else:
                for cand in (
                    "losses_low",
                    "losses_high",
                    "losses_a",
                    "losses_b",
                    "losses",
                ):
                    if cand in z:
                        arr = np.asarray(z[cand], dtype=float)
                        label = cand
                        break
                else:
                    raise SystemExit(
                        f"{path}: pass --key NAME; available: {names}",
                    )
    else:
        arr = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        label = path.stem
    return arr, label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=str, help=".npy tensor or .npz archive")
    p.add_argument(
        "--key",
        type=str,
        default=None,
        help="Array name inside .npz (default: first losses_* / losses)",
    )
    p.add_argument(
        "--plot",
        type=str,
        default=None,
        help="If set, write a PNG with mean loss per arm and a T×K heatmap (mean over trials)",
    )
    args = p.parse_args()

    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")

    losses, label = _load_tensor(path, args.key)
    if losses.ndim != 3:
        raise SystemExit(f"expected shape (T, K, trials), got {losses.shape} for {label!r}")

    T, K, trials = losses.shape
    print(f"file: {path}")
    print(f"array: {label!r}")
    print(f"shape: (T, K, trials) = ({T}, {K}, {trials})  [time, arm, trial]")
    print(f"dtype: {losses.dtype}  finite: {np.isfinite(losses).all()}")
    print(f"global min / max / mean: {np.nanmin(losses):.6g} / {np.nanmax(losses):.6g} / {np.nanmean(losses):.6g}")

    mean_over_t_trials = losses.mean(axis=(0, 2))
    std_over_trials_at_mean_t = losses.mean(axis=0).std(axis=1)
    print("\nper-arm mean loss (avg over time steps and trials):")
    for i in range(K):
        print(f"  arm {i:3d}: mean={mean_over_t_trials[i]:.6g}  std_across_trials@mean_t={std_over_trials_at_mean_t[i]:.6g}")

    best_per_step = losses.mean(axis=2).argmin(axis=1)
    print(f"\nbest arm (lowest mean loss over trials) at each t: unique arms used = {len(np.unique(best_per_step))}")

    if args.plot:
        import matplotlib.pyplot as plt

        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        heat = losses.mean(axis=2)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        axes[0].bar(np.arange(K), mean_over_t_trials)
        axes[0].set_xlabel("arm index")
        axes[0].set_ylabel("mean loss")
        axes[0].set_title(f"{label}: mean loss per arm (avg over t, trials)")

        im = axes[1].imshow(heat.T, aspect="auto", origin="lower", interpolation="nearest")
        axes[1].set_xlabel("time t")
        axes[1].set_ylabel("arm")
        axes[1].set_title("mean loss (avg over trials): rows = arms, cols = time")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.suptitle(str(path))
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
