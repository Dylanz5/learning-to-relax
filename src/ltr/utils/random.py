from __future__ import annotations

import numpy as np


def truncated_normal(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
     """Radially-truncated standard Gaussian (port of `utils/truncated_normal.m`).
 
     Rejection samples x ~ N(0, I_n) until ||x|| <= sqrt(n).
     """
 
     rng = rng or np.random.default_rng()
     bound = np.sqrt(float(n))
     while True:
         sample = rng.normal(0.0, 1.0, size=(n,))
         if np.linalg.norm(sample) <= bound:
             return sample
 
