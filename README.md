# Spectral bandits for SOR ω

Learners pick a relaxation parameter ω each round. Feedback is SOR iteration count
on a Poisson (5-point Laplacian) system, or a precomputed full-info loss tensor.

## Setup

```bash
conda activate banditos-clean   # or: pip install -e .
pip install -e .
python -m pytest
```

Configs are Python files. A config builds already-parameterized learners (the
graph lives on the spectral algo) and a data source, then calls
`run(algos, data)`. Sweeps are ordinary loops in the config.

```bash
python scripts/run.py configs/live_sor.py
python scripts/run.py configs/replay_pitzdaily.py
python scripts/run.py configs/sweep_alpha.py
python scripts/run.py configs/sweep_k.py
```

Optional Weights & Biases: install `wandb` yourself and set `WANDB_PROJECT`, or
pass `wandb_project=...` into `run()`. It is not a package dependency.

## Paper figures

**Regret vs timestep** (live SOR):

```bash
python scripts/run.py configs/live_sor.py
```

**Same plot from archived losses** (`data/fullinfo/pitzDaily.npy`):

```bash
python scripts/run.py configs/replay_pitzdaily.py
```

Each `run()` writes `plots/runs/<stamp>_<name>/`:

- `metrics.csv` — per-timestep mean cost / cumulative loss
- `summary.csv` — final totals
- `run.txt` — names, HPs, graph, data, seed, T, K, trials
- `arrays.npz` — cost and action-probability arrays

CSV losses work the same way (`load_losses("losses.csv", similarity_kind="chain")`,
T×K, one trial). Then pass the bundle into `run()` like the replay config.

**Arm probabilities vs time** are written next to the regret plot.
Replot from a saved `.npz`:

```bash
python scripts/plot_regret_t.py plots/runs/<run>/arrays.npz
python scripts/plot_arm_probs.py plots/runs/<run>/arrays.npz
```

**Regret vs number of configs (K)**:

```bash
python scripts/run.py configs/sweep_k.py
python scripts/plot_regret_k.py plots/sweeps/<stamp>_k/results.csv
```

## Basic hyperparameter sweep (offline α)

```bash
python scripts/run.py configs/sweep_alpha.py
```

## Layout

```
src/ltr/learners/     Tsallis-INF, Exp3-Spectral, Tsallis-Spectral
src/ltr/solvers/      SOR
src/ltr/data/         load .npy / .npz / .csv losses
src/ltr/graphs/       chain / ring / double_chain + load K×K
src/ltr/plot/         the three figures
src/ltr/experiment.py run(algos, data)
configs/              Python configs (live, replay, sweeps)
scripts/run.py        python scripts/run.py configs/foo.py
```

Old experiment dumps (pitzDailyLarge, sweeps, etc.) are in
[`../learning-to-relax-archive/`](ARCHIVE.md), not in this tree.
