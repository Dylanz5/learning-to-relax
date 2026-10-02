"""Shared bandit loop: ``run(algos, data)`` for live SOR and offline tensors."""

from __future__ import annotations

import csv
import datetime
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import scipy.sparse as sp

from ltr.data import LossBundle, load_losses
from ltr.domains import delsq_numgrid
from ltr.graphs import similarity_spectrum
from ltr.learners import Exp3Spectral, TsallisINF, TsallisSpectral
from ltr.plot import extract_action_probs, plot_arm_probabilities, plot_regret_over_time
from ltr.solvers import precalc_lower_diag, sor
from ltr.utils import truncated_normal

DEFAULT_LABELS: dict[str, str] = {
    "tinf": "Tsallis-INF",
    "exp3": "Exp3-Spectral",
    "tspec": "Tsallis-Spectral",
    "best_arm": "best arm",
    "best_fixed": "best fixed ω",
    "best_sampled": "best sampled ω",
}

LIVE_HPS: dict[str, Any] = dict(
    tinf_eta_scale=1.0,
    exp3_eta=0.05,
    exp3_eta_scale=1.0,
    exp3_eta_decay=0.0,
    exp3_gamma=0.0,
    exp3_mu=0.01,
    exp3_smoothness=0.01,
    tspec_eta=0.05,
    tspec_eta_scale=1.0,
    tspec_eta_decay=0.0,
    tspec_gamma=0.0,
    tspec_mu=0.01,
    tspec_smoothness=0.01,
    tspec_alpha=0.5,
)

REPLAY_HPS: dict[str, Any] = dict(
    tinf_eta_scale=1.0,
    exp3_eta=0.3,
    exp3_eta_scale=1.0,
    exp3_eta_decay=0.0,
    exp3_gamma=0.01,
    exp3_mu=0.01,
    exp3_smoothness=1.0,
    tspec_eta=0.1,
    tspec_eta_scale=1.0,
    tspec_eta_decay=0.0,
    tspec_gamma=0.0,
    tspec_mu=0.001,
    tspec_smoothness=0.0,
    tspec_alpha=0.9,
)

_DESCRIBE_SKIP = {
    "grid",
    "eigenvectors",
    "eigenvalues",
    "exploration",
    "L",
    "U",
    "lam",
    "arms",
    "S",
    "q",
    "pi",
    "diag",
    "k_hist",
    "p_hist",
    "_last_p",
    "_last_i",
    "k",
}


class RoundEnv(Protocol):
    def loss(self, action: float, index: int) -> float: ...

    def baseline_vector(self) -> np.ndarray | None: ...


class BanditData(Protocol):
    T: int
    trials: int
    grid: np.ndarray

    def make_env(self, t: int, trial: int, rng: np.random.Generator) -> RoundEnv: ...

    def describe(self) -> str: ...

    def trial_seeds(self, seed: int, n_trials: int) -> list[int]: ...


@dataclass
class Algo:
    """A named learner factory. Spectral HPs and the graph live on the learner."""

    name: str
    factory: Callable[[], Any]
    label: str | None = None
    describe: str = ""

    def legend(self) -> str:
        if self.label:
            return self.label
        return DEFAULT_LABELS.get(self.name, self.name)


@dataclass
class RunResult:
    run_dir: Path
    costs: dict[str, np.ndarray]
    probs: dict[str, np.ndarray]
    omega_costs: np.ndarray | None
    extras: dict[str, np.ndarray]
    totals: dict[str, float]
    curves: dict[str, np.ndarray]


@dataclass
class _OfflineEnv:
    row: np.ndarray
    base_idx: np.ndarray

    def loss(self, action: float, index: int) -> float:
        return float(self.row[int(index)])

    def baseline_vector(self) -> np.ndarray | None:
        if self.base_idx.size == 0:
            return None
        return np.asarray(self.row[self.base_idx], dtype=float)


@dataclass
class OfflineData:
    """Full-info tensor: ``losses[t, arm, trial]``."""

    bundle: LossBundle
    label: str = ""
    path: str = ""
    baseline_count: int = 5
    graph_desc: str = ""

    def __post_init__(self) -> None:
        K = int(self.bundle.K)
        n = int(self.baseline_count)
        if n < 0 or n >= K:
            self.base_idx = np.arange(K, dtype=int)
        elif n == 0:
            self.base_idx = np.array([], dtype=int)
        else:
            self.base_idx = np.unique(np.linspace(0, K - 1, n, dtype=int))

    @property
    def T(self) -> int:
        return int(self.bundle.T)

    @property
    def trials(self) -> int:
        return int(self.bundle.trials)

    @property
    def grid(self) -> np.ndarray:
        return self.bundle.grid

    def make_env(self, t: int, trial: int, rng: np.random.Generator) -> RoundEnv:
        del rng
        return _OfflineEnv(self.bundle.losses[t, :, trial], self.base_idx)

    def extra_arrays(self) -> dict[str, np.ndarray]:
        return {"best_arm": np.asarray(self.bundle.losses.min(axis=1), dtype=float)}

    def trial_seeds(self, seed: int, n_trials: int) -> list[int]:
        return [int(seed) + k for k in range(n_trials)]

    def describe(self) -> str:
        b = self.bundle
        bits = [
            f"offline tensor  T={b.T}  K={b.K}  trials={b.trials}",
            f"losses shape={tuple(b.losses.shape)}",
        ]
        if self.path:
            bits.append(f"path={self.path}")
        if self.label:
            bits.append(f"label={self.label}")
        if self.graph_desc:
            bits.append(f"graph={self.graph_desc}")
        bits.append(f"sampled baseline arms={list(map(int, self.base_idx))}")
        return "\n  ".join(bits)


@dataclass
class _LiveEnv:
    iters: Callable[[float], int]
    omegas: np.ndarray

    def loss(self, action: float, index: int) -> float:
        del index
        return float(self.iters(float(action)))

    def baseline_vector(self) -> np.ndarray | None:
        if self.omegas.size == 0:
            return None
        return np.array([self.iters(float(om)) for om in self.omegas], dtype=float)


@dataclass
class LiveSOR:
    """Poisson / 5-point Laplacian SOR: loss = iteration count at the chosen ω."""

    A: sp.csr_matrix
    grid: np.ndarray
    T: int
    trials: int
    epsilon: float = 1e-8
    rhs_a: float = 2.0
    rhs_b: float = 6.0
    baseline_omegas: np.ndarray | None = None
    domain_desc: str = ""

    def make_env(self, t: int, trial: int, rng: np.random.Generator) -> RoundEnv:
        del t, trial
        n = int(self.A.shape[0])
        At = self.A.copy().tocsr()
        base_diag = self.A.diagonal().copy()
        c = -0.15 + 0.6 * float(rng.beta(self.rhs_a, self.rhs_b))
        At.setdiag(base_diag + float(c))
        bt = truncated_normal(n, rng=rng)
        L_at, D_at = precalc_lower_diag(At)
        eps = float(self.epsilon)

        def iters(omega: float) -> int:
            return int(sor(At, bt, None, omega, eps, lower_triangle=L_at, diagonal=D_at).iterations)

        omegas = np.asarray(self.baseline_omegas, dtype=float).reshape(-1) if self.baseline_omegas is not None else np.array([])
        return _LiveEnv(iters, omegas)

    def extra_arrays(self) -> dict[str, np.ndarray]:
        return {}

    def trial_seeds(self, seed: int, n_trials: int) -> list[int]:
        rng = np.random.default_rng(int(seed))
        return [int(rng.integers(0, 2**32 - 1)) for _ in range(n_trials)]

    def describe(self) -> str:
        n = int(self.A.shape[0])
        grid = np.asarray(self.grid, dtype=float).reshape(-1)
        om = np.asarray(self.baseline_omegas, dtype=float).reshape(-1) if self.baseline_omegas is not None else np.array([])
        bits = [
            f"live SOR  n={n}  epsilon={self.epsilon:g}  rhs_a={self.rhs_a:g}  rhs_b={self.rhs_b:g}",
            f"T={self.T}  trials={self.trials}",
            f"action grid: [{grid[0]:g}, {grid[-1]:g}]  K={grid.size}",
        ]
        if self.domain_desc:
            bits.append(self.domain_desc)
        if om.size:
            bits.append(f"baseline ω ({om.size}): {np.array2string(om, precision=4)}")
        return "\n  ".join(bits)


def live_sor(
    *,
    T: int,
    trials: int,
    grid: np.ndarray,
    domain_s: int = 12,
    epsilon: float = 1e-8,
    rhs_a: float = 2.0,
    rhs_b: float = 6.0,
    omega_start: float = 1.0,
    omega_end: float = 1.8,
    omega_count: int = 5,
) -> LiveSOR:
    dom = delsq_numgrid("S", int(domain_s))
    omegas = np.linspace(float(omega_start), float(omega_end), int(omega_count))
    return LiveSOR(
        A=dom.A,
        grid=np.asarray(grid, dtype=float).reshape(-1),
        T=int(T),
        trials=int(trials),
        epsilon=float(epsilon),
        rhs_a=float(rhs_a),
        rhs_b=float(rhs_b),
        baseline_omegas=omegas,
        domain_desc=f"Poisson delsq(S, s={int(domain_s)})  unknowns={dom.A.shape[0]}",
    )


def offline(
    bundle: LossBundle,
    *,
    label: str = "",
    path: str = "",
    baseline_count: int = 5,
    graph_desc: str = "",
) -> OfflineData:
    return OfflineData(
        bundle=bundle,
        label=label,
        path=path,
        baseline_count=int(baseline_count),
        graph_desc=graph_desc,
    )


def offline_from_path(
    data: str | Path,
    *,
    path_similarity: str | Path | None = None,
    similarity_kind: str | None = None,
    similarity_is_laplacian: bool = False,
    baseline_count: int = 5,
) -> tuple[str, OfflineData]:
    path = Path(data)
    graph_desc = ""
    if path_similarity is not None:
        graph_desc = f"file {Path(path_similarity)}" + (" (laplacian)" if similarity_is_laplacian else "")
    elif similarity_kind:
        graph_desc = str(similarity_kind)
    bundles = load_losses(
        path,
        path_similarity=path_similarity,
        similarity_kind=similarity_kind,
        similarity_is_laplacian=similarity_is_laplacian,
    )
    name, bundle = next(iter(bundles.items()))
    return name, offline(
        bundle,
        label=name,
        path=str(path),
        baseline_count=baseline_count,
        graph_desc=graph_desc,
    )


def paper_algos(
    grid: np.ndarray,
    U: np.ndarray,
    lam: np.ndarray,
    L: Any,
    *,
    T: int,
    hps: Mapping[str, Any] | None = None,
    which: Sequence[str] = ("tinf", "exp3", "tspec"),
) -> list[Algo]:
    """Tsallis-INF / Exp3-Spectral / Tsallis-Spectral with graph baked into factories."""
    hp = dict(LIVE_HPS)
    if hps:
        hp.update(dict(hps))
    grid = np.asarray(grid, dtype=float).reshape(-1)
    want = tuple(which)
    out: list[Algo] = []
    same_mu = abs(float(hp["tspec_mu"]) - float(hp["exp3_mu"])) < 1e-15

    q_exp3 = None
    q_tspec = None
    if "exp3" in want or ("tspec" in want and same_mu):
        q_exp3 = Exp3Spectral(
            grid=grid,
            eigenvectors=U,
            eigenvalues=lam,
            eta=float(hp["exp3_eta"]),
            eta_scale=float(hp["exp3_eta_scale"]),
            eta_decay=float(hp["exp3_eta_decay"]),
            gamma=float(hp["exp3_gamma"]),
            mu=float(hp["exp3_mu"]),
            smoothness=float(hp["exp3_smoothness"]),
            L=L,
        ).q
        if same_mu:
            q_tspec = q_exp3
    if "tspec" in want and q_tspec is None:
        q_tspec = TsallisSpectral(
            grid=grid,
            eigenvectors=U,
            eigenvalues=lam,
            eta=float(hp["tspec_eta"]),
            eta_scale=float(hp["tspec_eta_scale"]),
            eta_decay=float(hp["tspec_eta_decay"]),
            gamma=float(hp["tspec_gamma"]),
            mu=float(hp["tspec_mu"]),
            smoothness=float(hp["tspec_smoothness"]),
            alpha=float(hp["tspec_alpha"]),
            L=L,
        ).pi

    if "tinf" in want:
        tinf_eta = float(hp["tinf_eta_scale"])

        def make_tinf(eta=tinf_eta, g=grid, horizon=int(T)) -> TsallisINF:
            return TsallisINF(g, T=horizon, eta_scale=eta)

        out.append(Algo("tinf", make_tinf, describe=f"TsallisINF eta_scale={tinf_eta} T={int(T)}"))

    if "exp3" in want:
        assert q_exp3 is not None
        kw = dict(
            grid=grid,
            eigenvectors=U,
            eigenvalues=lam,
            eta=float(hp["exp3_eta"]),
            eta_scale=float(hp["exp3_eta_scale"]),
            eta_decay=float(hp["exp3_eta_decay"]),
            gamma=float(hp["exp3_gamma"]),
            mu=float(hp["exp3_mu"]),
            smoothness=float(hp["exp3_smoothness"]),
            L=L,
            exploration=q_exp3,
        )

        def make_exp3(kwargs=kw) -> Exp3Spectral:
            return Exp3Spectral(**kwargs)

        out.append(Algo("exp3", make_exp3, describe=_describe_hps("Exp3Spectral", kw)))

    if "tspec" in want:
        assert q_tspec is not None
        kw = dict(
            grid=grid,
            eigenvectors=U,
            eigenvalues=lam,
            eta=float(hp["tspec_eta"]),
            eta_scale=float(hp["tspec_eta_scale"]),
            eta_decay=float(hp["tspec_eta_decay"]),
            gamma=float(hp["tspec_gamma"]),
            mu=float(hp["tspec_mu"]),
            smoothness=float(hp["tspec_smoothness"]),
            alpha=float(hp["tspec_alpha"]),
            L=L,
            exploration=q_tspec,
        )

        def make_tspec(kwargs=kw) -> TsallisSpectral:
            return TsallisSpectral(**kwargs)

        out.append(Algo("tspec", make_tspec, describe=_describe_hps("TsallisSpectral", kw)))

    return out


def _describe_hps(cls_name: str, kw: Mapping[str, Any]) -> str:
    skip = {"grid", "eigenvectors", "eigenvalues", "exploration", "L"}
    parts = [cls_name]
    for k, v in kw.items():
        if k in skip:
            continue
        parts.append(f"{k}={v}")
    L = kw.get("L")
    if L is not None:
        shape = getattr(L, "shape", None)
        parts.append(f"L.shape={shape}")
    return " ".join(parts)


def _normalize_algos(algos: Sequence[Any]) -> list[Algo]:
    out: list[Algo] = []
    for a in algos:
        if isinstance(a, Algo):
            out.append(a)
            continue
        if isinstance(a, Mapping):
            out.append(
                Algo(
                    name=str(a["name"]),
                    factory=a["factory"],
                    label=a.get("label"),
                    describe=str(a.get("describe") or ""),
                )
            )
            continue
        obj = a
        name = type(obj).__name__
        out.append(Algo(name=name, factory=lambda o=obj: o, describe=_describe_learner(obj)))
    if not out:
        raise ValueError("algos is empty")
    return out


def _describe_learner(obj: Any) -> str:
    parts = [type(obj).__name__]
    if is_dataclass(obj):
        for f in fields(obj):
            if f.name in _DESCRIBE_SKIP:
                continue
            val = getattr(obj, f.name, None)
            if isinstance(val, (int, float, str, bool)):
                parts.append(f"{f.name}={val}")
    L = getattr(obj, "L", None)
    if L is not None:
        parts.append(f"L.shape={getattr(L, 'shape', None)}")
    return " ".join(parts)


def _learner_index(learner: Any) -> int:
    idx = getattr(learner, "index", None)
    if idx is not None:
        return int(idx)
    last = getattr(learner, "_last_i", None)
    if last is not None:
        return int(last)
    raise RuntimeError(f"{type(learner).__name__} did not record an arm index after predict()")


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return s or "run"


def _one_trial(
    algos: list[Algo],
    data: BanditData,
    *,
    T: int,
    K: int,
    n_base: int,
    trial_idx: int,
    trial_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(trial_seed))
    learners = {a.name: a.factory() for a in algos}
    costs = {a.name: np.zeros(T, dtype=float) for a in algos}
    probs = {a.name: np.zeros((T, K), dtype=float) for a in algos}
    omega = np.zeros((T, n_base), dtype=float) if n_base else None
    for t in range(T):
        env = data.make_env(t, trial_idx, rng)
        for a in algos:
            learner = learners[a.name]
            action = float(learner.predict(rng=rng))
            p = extract_action_probs(learner, K)
            if p is not None:
                probs[a.name][t, :] = p
            idx = _learner_index(learner)
            loss = float(env.loss(action, idx))
            costs[a.name][t] = loss
            learner.update(loss)
        if omega is not None:
            vec = env.baseline_vector()
            if vec is None or int(np.asarray(vec).size) != n_base:
                raise RuntimeError("baseline vector missing or wrong length")
            omega[t, :] = np.asarray(vec, dtype=float)
    return {"costs": costs, "probs": probs, "omega": omega}


def _baseline_count(data: BanditData) -> int:
    om = getattr(data, "baseline_omegas", None)
    if om is not None:
        return int(np.asarray(om).reshape(-1).size)
    idx = getattr(data, "base_idx", None)
    if idx is not None:
        return int(np.asarray(idx).reshape(-1).size)
    return 0


def _curves(
    costs: Mapping[str, np.ndarray],
    algos: Sequence[Algo],
    omega_costs: np.ndarray | None,
    extras: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for a in algos:
        curves[a.legend()] = np.mean(np.cumsum(costs[a.name], axis=0), axis=1)
    if omega_costs is not None and omega_costs.size:
        best_fixed = omega_costs.min(axis=2)
        key = "best_sampled" if "best_arm" in extras else "best_fixed"
        curves[DEFAULT_LABELS[key]] = np.mean(np.cumsum(best_fixed, axis=0), axis=1)
    if "best_arm" in extras:
        curves[DEFAULT_LABELS["best_arm"]] = np.mean(np.cumsum(extras["best_arm"], axis=0), axis=1)
    return curves


def _write_metrics_csv(
    path: Path,
    *,
    T: int,
    algos: Sequence[Algo],
    costs: Mapping[str, np.ndarray],
    omega_costs: np.ndarray | None,
    extras: Mapping[str, np.ndarray],
) -> None:
    series: list[tuple[str, np.ndarray]] = [(a.name, costs[a.name]) for a in algos]
    if omega_costs is not None and omega_costs.size:
        key = "best_sampled" if "best_arm" in extras else "best_fixed"
        series.append((key, omega_costs.min(axis=2)))
    for name, arr in extras.items():
        series.append((name, arr))
    fieldnames = ["t"]
    cols: dict[str, np.ndarray] = {}
    for name, arr in series:
        mean_cost = np.mean(arr, axis=1)
        mean_cum = np.mean(np.cumsum(arr, axis=0), axis=1)
        fieldnames.extend([f"{name}_mean_cost", f"{name}_mean_cum"])
        cols[f"{name}_mean_cost"] = mean_cost
        cols[f"{name}_mean_cum"] = mean_cum
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for t in range(T):
            row: dict[str, Any] = {"t": t}
            for k, v in cols.items():
                row[k] = float(v[t])
            w.writerow(row)


def _write_summary_csv(path: Path, totals: Mapping[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "mean_total"])
        w.writeheader()
        for k, v in totals.items():
            w.writerow({"name": k, "mean_total": v})


def _write_run_txt(
    path: Path,
    *,
    stem: str,
    seed: int,
    T: int,
    K: int,
    trials: int,
    jobs: int,
    algos: Sequence[Algo],
    data: BanditData,
    notes: Mapping[str, Any],
    wandb_project: str | None,
) -> str:
    lines = [
        f"run: {stem}",
        f"seed: {seed}",
        f"T: {T}",
        f"K: {K}",
        f"trials: {trials}",
        f"jobs: {jobs}",
        "",
        "data:",
        f"  {data.describe()}",
        "",
        "algos:",
    ]
    for a in algos:
        desc = a.describe or "(factory)"
        lines.append(f"  {a.name} ({a.legend()}): {desc}")
    if notes:
        lines.append("")
        lines.append("notes:")
        for k, v in notes.items():
            lines.append(f"  {k}: {v}")
    if wandb_project:
        lines.append("")
        lines.append(f"wandb_project: {wandb_project}")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _run_txt_as_config(text: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if k and k not in {"data", "algos", "notes"}:
                cfg[k] = v
    return cfg


def _resolve_wandb_project(explicit: str | None) -> str | None:
    """``None`` → ``WANDB_PROJECT`` env; empty string disables logging."""
    if explicit is not None:
        return str(explicit).strip() or None
    return (os.environ.get("WANDB_PROJECT") or "").strip() or None


def _maybe_wandb(
    *,
    project: str | None,
    stem: str,
    run_dir: Path,
    totals: Mapping[str, float],
    run_txt: str,
) -> None:
    if not project:
        return
    try:
        import wandb
    except ImportError:
        print("[run] wandb not installed; skipping", flush=True)
        return
    wb = wandb.init(project=project, name=stem, config=_run_txt_as_config(run_txt))
    for k, v in totals.items():
        wb.summary[k] = v
    art = wandb.Artifact(_slug(stem), type="run")
    for fname in ("metrics.csv", "summary.csv", "run.txt", "arrays.npz"):
        fpath = run_dir / fname
        if fpath.exists():
            art.add_file(str(fpath))
    wb.log_artifact(art)
    wb.finish()


def run(
    algos: Sequence[Any],
    data: BanditData,
    *,
    seed: int = 0,
    trials: int | None = None,
    out_dir: str | Path = "plots/runs",
    name: str = "run",
    jobs: int = 1,
    notes: Mapping[str, Any] | None = None,
    wandb_project: str | None = None,
    plot: bool = True,
) -> RunResult:
    """``predict`` → loss from ``data`` → ``update``. Writes CSV + run.txt + npz."""
    algos_n = _normalize_algos(algos)
    for a in algos_n:
        if not a.describe:
            a.describe = _describe_learner(a.factory())
    T = int(data.T)
    n_trials = int(trials) if trials is not None else int(data.trials)
    grid = np.asarray(data.grid, dtype=float).reshape(-1)
    K = int(grid.size)
    n_base = _baseline_count(data)
    seeds = data.trial_seeds(int(seed), n_trials)

    def job(item: tuple[int, int]) -> dict[str, Any]:
        trial_idx, trial_seed = item
        return _one_trial(algos_n, data, T=T, K=K, n_base=n_base, trial_idx=trial_idx, trial_seed=trial_seed)

    items = list(enumerate(seeds))
    workers = max(1, int(jobs))
    if workers == 1 or n_trials == 1:
        outs = [job(it) for it in items]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, n_trials)) as ex:
            outs = list(ex.map(job, items))

    costs = {a.name: np.stack([o["costs"][a.name] for o in outs], axis=1) for a in algos_n}
    probs = {a.name: np.stack([o["probs"][a.name] for o in outs], axis=1) for a in algos_n}
    omega_costs = None
    if n_base:
        omega_costs = np.stack([o["omega"] for o in outs], axis=1)
    extras_fn = getattr(data, "extra_arrays", None)
    extras = dict(extras_fn()) if callable(extras_fn) else {}

    totals: dict[str, float] = {}
    for a in algos_n:
        totals[f"{a.name}_total"] = float(np.mean(costs[a.name].sum(axis=0)))
    if omega_costs is not None:
        key = "best_sampled_total" if "best_arm" in extras else "best_fixed_total"
        totals[key] = float(np.mean(omega_costs.min(axis=2).sum(axis=0)))
    for name_e, arr in extras.items():
        totals[f"{name_e}_total"] = float(np.mean(np.asarray(arr).sum(axis=0)))

    curves = _curves(costs, algos_n, omega_costs, extras)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    stem = f"{stamp}_{_slug(name)}"
    run_dir = Path(out_dir) / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    save: dict[str, Any] = dict(
        T=np.int32(T),
        trials=np.int32(n_trials),
        seed=np.int32(seed),
        grid=grid,
        algo_names=np.array([a.name for a in algos_n]),
    )
    for a in algos_n:
        save[f"{a.name}_costs"] = costs[a.name]
        save[f"{a.name}_action_probs"] = probs[a.name].astype(np.float32)
    if omega_costs is not None:
        save["omega_costs"] = omega_costs
        if getattr(data, "baseline_omegas", None) is not None:
            save["omegas"] = np.asarray(data.baseline_omegas, dtype=float)
    for name_e, arr in extras.items():
        save[f"{name_e}_costs"] = np.asarray(arr)
    np.savez_compressed(run_dir / "arrays.npz", **save)

    _write_metrics_csv(
        run_dir / "metrics.csv",
        T=T,
        algos=algos_n,
        costs=costs,
        omega_costs=omega_costs,
        extras=extras,
    )
    _write_summary_csv(run_dir / "summary.csv", totals)
    project = _resolve_wandb_project(wandb_project)
    run_txt = _write_run_txt(
        run_dir / "run.txt",
        stem=stem,
        seed=int(seed),
        T=T,
        K=K,
        trials=n_trials,
        jobs=workers,
        algos=algos_n,
        data=data,
        notes=dict(notes or {}),
        wandb_project=project,
    )

    if plot:
        plot_regret_over_time(curves, run_dir / "regret_t.png", title=stem)
        for a in algos_n:
            plot_arm_probabilities(
                np.mean(probs[a.name], axis=1),
                run_dir / f"{a.name}_action_probs.png",
                title=f"{a.legend()}: arm probability vs time",
            )

    _maybe_wandb(
        project=project,
        stem=stem,
        run_dir=run_dir,
        totals=totals,
        run_txt=run_txt,
    )
    print(f"[run] wrote {run_dir}", flush=True)
    return RunResult(
        run_dir=run_dir,
        costs=costs,
        probs=probs,
        omega_costs=omega_costs,
        extras=extras,
        totals=totals,
        curves=curves,
    )


def spectrum_for(grid: np.ndarray, *, kind: str | None = None, path: str | Path | None = None, is_laplacian: bool | None = None):
    """``(U, lam, L)`` for a named graph or a loaded matrix; K comes from ``grid``."""
    return similarity_spectrum(int(np.asarray(grid).reshape(-1).size), kind, path=path, is_laplacian=is_laplacian)
