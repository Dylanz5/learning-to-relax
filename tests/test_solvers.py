from __future__ import annotations

import numpy as np

from ltr.domains import delsq_numgrid
from ltr.solvers.sor import sor
from ltr.solvers.ssor_pcg import ssor_pcg


def test_sor_converges_on_laplacian() -> None:
     dom = delsq_numgrid("S", 12)
     A = dom.A
     n = A.shape[0]
     b = np.ones(n)
     res = sor(A, b, np.zeros(n), omega=1.5, tol=1e-8, maxiter=10_000)
     assert res.iterations > 0
     r = b - A @ res.x
     assert np.linalg.norm(r) / np.linalg.norm(b) < 1e-6


def test_ssor_pcg_converges_on_laplacian() -> None:
     dom = delsq_numgrid("S", 12)
     A = dom.A
     n = A.shape[0]
     b = np.ones(n)
     res = ssor_pcg(A, b, np.zeros(n), omega=1.5, tol=1e-8, maxiter=5_000)
     assert res.iterations > 0
     r = b - A @ res.x
     assert np.linalg.norm(r) / np.linalg.norm(b) < 1e-6
 
