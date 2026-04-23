from __future__ import annotations

import numpy as np


def bump(x: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
     """Smooth bump function (port of `utils/bump.m`).
 
     Parameters
     ----------
     x:
         Array of points of shape (n, d).
     center:
         Center point of shape (d,) or broadcastable.
     radius:
         Radius scalar.
     """
 
     x = np.asarray(x, dtype=float)
     center = np.asarray(center, dtype=float)
     r2 = float(radius) ** 2
     # MATLAB: exp(1 - 1 / (1 - min(1, sum((x-center).^2 / radius^2, 2))))
     t = np.sum((x - center) ** 2, axis=1) / r2
     t = np.minimum(1.0, t)
     return np.exp(1.0 - 1.0 / (1.0 - t))
 
