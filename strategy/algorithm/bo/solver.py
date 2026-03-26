"""Inner-solver implementations.

An ``InnerSolver`` finds candidate decision vectors by optimising an
acquisition function over [0, 1]^d.

Available
---------
LBFGSBSolver
    Multi-start L-BFGS-B (scipy).  Replaces the pymoo CMA-ES used in the
    original HVI-distribution code (``mobo/solver/parego/parego.py``),
    which relies on ``pymoo.algorithms.so_cmaes`` removed in pymoo ≥ 0.6.

References
----------
[1] R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A limited memory
    algorithm for bound constrained optimization," SIAM J. Sci. Comput.,
    vol. 16, no. 5, pp. 1190–1208, 1995.
    https://doi.org/10.1137/0916069
    (L-BFGS-B algorithm implemented in scipy.optimize.minimize.)

[2] M. Lukovic, Y. Tian, and W. Ma, "Diversity-guided multi-objective
    Bayesian optimization with batch evaluations," in Proc. NeurIPS, 2020.
    Source code: https://github.com/yunshengtian/DGEMO
    (Parallel-process CMA-ES inner solver adapted from
    ``mobo/solver/parego/parego.py::ParEGOSolver.solve``.)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as _sp_minimize

from .base import InnerSolver, SurrogateModel


class LBFGSBSolver(InnerSolver):
    """Multi-start L-BFGS-B inner optimiser.

    Maximises a scalar objective ``acq_fn(x_norm, surrogate, **acq_kwargs)``
    over the unit hypercube [0, 1]^n_var.

    Parameters
    ----------
    n_restarts : int
        Number of random starting points per candidate (default 3).
    max_iter   : int
        Maximum L-BFGS-B iterations per restart (default 200).
    """

    def __init__(self, n_restarts: int = 3, max_iter: int = 200) -> None:
        self.n_restarts = n_restarts
        self.max_iter   = max_iter

    def solve(
        self,
        surrogate: SurrogateModel,
        y_min_norm: np.ndarray,
        n_candidates: int,
        n_var: int,
        rng: np.random.RandomState,
        acq_fn,
        **acq_kwargs,
    ) -> np.ndarray:
        """Return (n_candidates, n_var) candidates, each maximising ``acq_fn``.

        Parameters
        ----------
        acq_fn : callable
            Signature: ``acq_fn(x_norm_1d, surrogate, y_min_norm, **acq_kwargs)
            → float``.  Should return a *negative* value (convention: we
            minimise the negative acquisition).
        acq_kwargs :
            Extra keyword arguments forwarded to ``acq_fn`` (e.g. weights).
        """
        bounds = [(0.0, 1.0)] * n_var
        results = np.empty((n_candidates, n_var))

        for c in range(n_candidates):
            best_x, best_val = rng.uniform(0.0, 1.0, n_var), np.inf
            # Draw all restart starting points at once for reproducibility
            starts = rng.uniform(0.0, 1.0, (self.n_restarts, n_var))
            # Per-candidate kwargs (e.g. weight vectors indexed by candidate)
            kw_c = {k: (v[c] if isinstance(v, np.ndarray) and v.ndim > 1 else v)
                    for k, v in acq_kwargs.items()}
            # Build a plain negative-acquisition function for scipy
            def _neg_acq(x, _s=surrogate, _y=y_min_norm, _kw=kw_c):
                return acq_fn(x, _s, _y, **_kw)
            for x0 in starts:
                try:
                    res = _sp_minimize(
                        _neg_acq,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': self.max_iter, 'ftol': 1e-9},
                    )
                    if res.fun < best_val:
                        best_val = float(res.fun)
                        best_x   = res.x
                except Exception:
                    pass
            results[c] = np.clip(best_x, 0.0, 1.0)

        return results
