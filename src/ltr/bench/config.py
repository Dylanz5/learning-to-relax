"""Experiment config for ``scripts/learning.py`` (JSON + dataclass)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class LearningExperimentConfig:
    """All knobs that ``run()`` needs; CLI can override after loading JSON."""

    T: int = 5000
    trials: int = 40
    seed: int = 0
    jobs: int = 0
    domain_s: int = 12
    epsilon: float = 1e-8
    grid_start: float = 1.0
    grid_end: float = 1.95
    grid_points: int = 20
    omega_start: float = 1.0
    omega_end: float = 1.8
    omega_count: int = 5
    similarity_kind: str = "chain"
    #: Prefix for plot filenames (defaults to similarity_kind if empty).
    run_name: str = ""
    high_var_dist_a: float = 0.5
    high_var_dist_b: float = 1.5
    low_var_dist_a: float = 2.0
    low_var_dist_b: float = 6.0
    exp3_eta: float = 0.05
    exp3_gamma: float = 0.1
    exp3_mu: float = 1e-2
    exp3_smoothness: float = 1.0
    #: Spectral Tsallis (Algorithm 2) hyperparameters. ``alpha`` is the Tsallis
    #: entropy parameter in (0, 1); at 1/2 the q_t solver matches Tsallis-INF.
    tsallis_spectral_eta: float = 0.05
    tsallis_spectral_gamma: float = 0.1
    tsallis_spectral_mu: float = 1e-2
    tsallis_spectral_smoothness: float = 1.0
    tsallis_spectral_alpha: float = 0.5
    #: Bandit feedback: ``"sor"`` (stationary iteration) or ``"ssor_pcg"``
    #: (SSOR-preconditioned CG; MATLAB ``ssor_pcg`` / ``pcg``).
    solver: str = "sor"

    def __post_init__(self) -> None:
        if self.solver not in ("sor", "ssor_pcg"):
            raise ValueError("solver must be 'sor' or 'ssor_pcg', " f"got {self.solver!r}")

    def plot_prefix(self) -> str:
        name = (self.run_name or self.similarity_kind).strip()
        return f"{name}_" if name else ""

    def param_tag(self) -> str:
        """Compact, filename-safe tag encoding the Exp3 hyperparameters.

        Uses ``ep``/``et``/``g``/``mu``/``sm`` for epsilon, eta, gamma, mu and
        smoothness. Decimal points are rendered as ``p`` so the tag stays a
        single dot-free path component (e.g. ``ep1e-08_et0p05_g0_mu0p01_sm0p01``).
        """

        def fmt(x: float) -> str:
            return f"{x:g}".replace(".", "p")

        return (
            f"ep{fmt(self.epsilon)}"
            f"_et{fmt(self.exp3_eta)}"
            f"_g{fmt(self.exp3_gamma)}"
            f"_mu{fmt(self.exp3_mu)}"
            f"_sm{fmt(self.exp3_smoothness)}"
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> LearningExperimentConfig:
        path = Path(path)
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> LearningExperimentConfig:
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**{k: raw[k] for k in valid if k in raw})

    def to_json_file(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
