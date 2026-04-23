from __future__ import annotations

import numpy as np

from ltr.domains import delsq_numgrid


def test_square_domain_size() -> None:
     dom = delsq_numgrid("S", 12)
     assert dom.A.shape[0] == (12 - 2) ** 2
     assert dom.coords.shape == (dom.A.shape[0], 2)


def test_l_domain_smaller_than_square() -> None:
     s = 12
     sq = delsq_numgrid("S", s)
     l = delsq_numgrid("L", s)
     assert l.A.shape[0] < sq.A.shape[0]
     # matrix should be SPD-ish: diagonal positive
     assert np.all(l.A.diagonal() > 0)
 
