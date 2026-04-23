from __future__ import annotations

import numpy as np

from ltr.learners.chebcb import ChebCB
from ltr.learners.tsallis_inf import TsallisINF
from ltr.learners.tsallis_inf_cb import TsallisINFCB


def test_tsallis_inf_runs() -> None:
    rng = np.random.default_rng(0)
    alg = TsallisINF(np.linspace(1.0, 1.95, 7), T=10)
    for _ in range(5):
        _ = alg.predict(rng=rng)
        alg.update(float(rng.integers(1, 100)))
 
 
def test_tsallis_inf_cb_runs() -> None:
    rng = np.random.default_rng(0)
    alg = TsallisINFCB(np.linspace(1.0, 1.95, 7), m=3, a=-0.15, b=0.65)
    for _ in range(5):
        _ = alg.predict(float(rng.uniform(-0.15, 0.65)), rng=rng)
        alg.update(float(rng.integers(1, 100)))
 
 
def test_chebcb_runs() -> None:
    rng = np.random.default_rng(0)
    alg = ChebCB(np.linspace(1.0, 1.95, 7), T=20, m=6, a=-0.15, b=0.65)
    for _ in range(10):
        _ = alg.predict(float(rng.uniform(-0.15, 0.65)), rng=rng)
        alg.update(float(rng.integers(1, 100)))
 
