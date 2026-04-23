from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

Region = Literal["S", "L"]


@dataclass(frozen=True)
class Domain:
     """Discrete domain and its Laplacian matrix.
 
     The MATLAB code uses `A = delsq(numgrid(region, s))`, which produces a
     5-point finite-difference Laplacian on a unit square grid with a shaped
     domain. For this Python port (idiomatic; minor numeric differences OK),
     we implement the same *intent*:
 
     - region='S': full interior of an s-by-s grid (so (s-2)^2 unknowns)
     - region='L': L-shaped domain (unit square minus upper-right quadrant)
 
     The returned matrix is SPD and sparse.
     """
 
     region: Region
     s: int
     A: sp.csr_matrix
     coords: np.ndarray  # (n, 2) interior point coordinates in [0,1]^2
     mask: np.ndarray  # (n_int, n_int) bool mask for interior points


def _laplacian_from_mask(mask: np.ndarray, h: float) -> tuple[sp.csr_matrix, np.ndarray]:
     """Build 5-point Laplacian on the masked interior points.
 
     Uses Dirichlet boundary conditions on the boundary of the (masked) domain.
     """
 
     if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
         raise ValueError("mask must be a square 2D boolean array")
     if mask.dtype != bool:
         mask = mask.astype(bool)
 
     n = mask.shape[0]
     # index map: (i,j) -> row id for unknowns, -1 for excluded
     idx = -np.ones((n, n), dtype=np.int64)
     pts = np.argwhere(mask)
     idx[mask] = np.arange(pts.shape[0], dtype=np.int64)
 
     rows: list[int] = []
     cols: list[int] = []
     data: list[float] = []
 
     # 5-point stencil scaled by 1/h^2
     inv_h2 = 1.0 / (h * h)
 
     for (i, j) in pts:
         r = int(idx[i, j])
         # center
         rows.append(r)
         cols.append(r)
         data.append(4.0 * inv_h2)
 
         # neighbors inside the domain contribute -1/h^2
         for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
             ni, nj = i + di, j + dj
             if 0 <= ni < n and 0 <= nj < n and mask[ni, nj]:
                 c = int(idx[ni, nj])
                 rows.append(r)
                 cols.append(c)
                 data.append(-1.0 * inv_h2)
 
     A = sp.coo_matrix((data, (rows, cols)), shape=(pts.shape[0], pts.shape[0])).tocsr()
 
     # coords for unknowns in [0,1]^2 (interior points)
     # MATLAB's grid is on (0,1) with s points including boundary.
     # Interior points correspond to i=1..s-2 (0-indexed in Python: 0..n-1).
     coords = (pts + 1) * h
     return A, coords.astype(np.float64, copy=False)


def delsq_numgrid(region: Region, s: int) -> Domain:
     """Replacement for MATLAB `delsq(numgrid(region, s))`.
 
     Parameters
     ----------
     region:
         'S' (square) or 'L' (L-shaped).
     s:
         Grid size in the MATLAB sense. Unknown count for 'S' is (s-2)^2.
 
     Returns
     -------
     Domain
         Contains sparse Laplacian `A` and coordinates for each unknown.
     """
 
     if s < 4:
         raise ValueError("s must be >= 4")
 
     # number of interior points per side
     n_int = s - 2
     h = 1.0 / s
 
     if region == "S":
         mask = np.ones((n_int, n_int), dtype=bool)
     elif region == "L":
         # L-shape: remove upper-right quadrant of the interior.
         # For even n_int, this removes a (n_int/2)x(n_int/2) block.
         mask = np.ones((n_int, n_int), dtype=bool)
         mid = n_int // 2
         mask[mid:, mid:] = False
     else:
         raise ValueError("region must be 'S' or 'L'")
 
     A, coords = _laplacian_from_mask(mask, h=h)
     print("A shape:", A.shape)

     return Domain(region=region, s=s, A=A, coords=coords, mask=mask)
 
