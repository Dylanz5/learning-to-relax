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

    def plot_prefix(self) -> str:
        name = (self.run_name or self.similarity_kind).strip()
        return f"{name}_" if name else ""

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
