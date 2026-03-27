"""Inner-solver implementations.

An ``InnerSolver`` finds candidate decision vectors by optimising an
acquisition function over [0, 1]^d.

Available
---------
LBFGSBSolver
    Multi-start L-BFGS-B (scipy).  Replaces the pymoo CMA-ES used in the
    original HVI-distribution code (``mobo/solver/parego/parego.py``),
    which relies on ``pymoo.algorithms.so_cmaes`` removed in pymoo ≥ 0.6.
EASolver
    Differential-Evolution inner solver (scipy).  Initialises its
    population from a mix of existing archive points and random samples,
    matching the EVOLALG step in the ParEGO pseudocode.

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

[3] R. Storn and K. Price, "Differential evolution — a simple and
    efficient heuristic for global optimization over continuous spaces,"
    J. Glob. Optim., vol. 11, pp. 341–359, 1997.
    (DE/rand/1/bin variant used in scipy.optimize.differential_evolution.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize as _sp_minimize
from scipy.optimize import differential_evolution as _de

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
        X_existing: Optional[np.ndarray] = None,  # ignored by L-BFGS-B
        **acq_kwargs,
    ) -> np.ndarray:
        """Return (n_candidates, n_var) candidates, each maximising ``acq_fn``.

        Parameters
        ----------
        acq_fn : callable
            Signature: ``acq_fn(x_norm_1d, surrogate, y_min_norm, **acq_kwargs)
            → float``.  Should return a *negative* value (convention: we
            minimise the negative acquisition).
        X_existing : ignored (accepted for interface compatibility).
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


class EASolver(InnerSolver):
    """Differential Evolution inner solver.

    Matches the EVOLALG step in the ParEGO pseudocode: initialises its
    population from a mix of existing archive points (mutants) and purely
    random samples, then evolves to maximise the acquisition function.

    Parameters
    ----------
    pop_size : int
        DE population size.  Must be ≥ 4.
    max_iter : int
        Maximum DE generations per candidate.
    mutation : float
        DE mutation scale factor F (default 0.8).
    recombination : float
        DE crossover probability CR (default 0.9).
    archive_frac : float
        Fraction of the initial population seeded from the archive
        (remainder is random).  Default 0.5.
    """

    def __init__(
        self,
        pop_size: int = 20,
        max_iter: int = 100,
        mutation: float = 0.8,
        recombination: float = 0.9,
        archive_frac: float = 0.5,
    ) -> None:
        if pop_size < 4:
            raise ValueError("EASolver requires pop_size >= 4")
        self.pop_size      = pop_size
        self.max_iter      = max_iter
        self.mutation      = mutation
        self.recombination = recombination
        self.archive_frac  = archive_frac

    def solve(
        self,
        surrogate: SurrogateModel,
        y_min_norm: np.ndarray,
        n_candidates: int,
        n_var: int,
        rng: np.random.RandomState,
        acq_fn,
        X_existing: Optional[np.ndarray] = None,
        **acq_kwargs,
    ) -> np.ndarray:
        """Return (n_candidates, n_var) candidates, each maximising ``acq_fn``.

        Parameters
        ----------
        acq_fn : callable
            Same convention as ``LBFGSBSolver``: returns a *negative* scalar.
        X_existing : (N, n_var) current archive in normalised space.
            Half of the initial DE population is seeded from random rows of
            this array (mirrors "some as mutants of xpop[]" in EVOLALG).
        acq_kwargs :
            Extra keyword arguments forwarded to ``acq_fn`` (e.g. weights).
        """
        bounds  = [(0.0, 1.0)] * n_var
        results = np.empty((n_candidates, n_var))

        for c in range(n_candidates):
            kw_c = {k: (v[c] if isinstance(v, np.ndarray) and v.ndim > 1 else v)
                    for k, v in acq_kwargs.items()}

            def _neg_acq(x, _s=surrogate, _y=y_min_norm, _kw=kw_c):
                return acq_fn(x, _s, _y, **_kw)

            init_pop = self._build_init_pop(X_existing, n_var, rng)
            res = _de(
                _neg_acq,
                bounds,
                init=init_pop,
                maxiter=self.max_iter,
                mutation=self.mutation,
                recombination=self.recombination,
                seed=int(rng.randint(0, 2 ** 31)),
                tol=0,
                polish=False,
            )
            results[c] = np.clip(res.x, 0.0, 1.0)

        return results

    def _build_init_pop(
        self,
        X_existing: Optional[np.ndarray],
        n_var: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Build initial DE population: archive_frac from archive, rest random."""
        n_from_archive = int(self.pop_size * self.archive_frac)
        n_random       = self.pop_size - n_from_archive
        random_part    = rng.uniform(0.0, 1.0, (n_random, n_var))

        if X_existing is not None and len(X_existing) >= 1:
            n_take        = min(n_from_archive, len(X_existing))
            idx           = rng.choice(len(X_existing), n_take, replace=False)
            archive_part  = X_existing[idx]
            if n_take < n_from_archive:
                extra        = rng.uniform(0.0, 1.0, (n_from_archive - n_take, n_var))
                archive_part = np.vstack([archive_part, extra])
        else:
            archive_part = rng.uniform(0.0, 1.0, (n_from_archive, n_var))

        return np.vstack([archive_part, random_part])
