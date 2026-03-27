"""ParEGO — Pareto Efficient Global Optimisation.

Single module exposing ``run_parego`` as the benchmark runner entry-point.

Algorithm
---------
ParEGO [1] decomposes a multi-objective problem into a sequence of scalarised
single-objective problems via augmented Tchebycheff:

    s(w, f) = max_j(w_j * f_j) + ρ · Σ_j(w_j * f_j)

A fresh weight vector w is sampled each iteration from the unit simplex [1,
Section III-A].  The scalarised *objective* is replaced here by the
scalarised *Expected Improvement* [2], so the surrogate is never evaluated
on the true objective directly.

Implementation details
----------------------
- Backbone  : ``strategy.algorithm.bo.BayesianOptimizer``
- Surrogate : ``GaussianProcessSurrogate`` (Matern-5/2, one GP per objective)
- Acquisition: per-objective Expected Improvement (``expected_improvement``) [2]
- Scalarisation: augmented Tchebycheff on EI values, ρ = 0.05 [1]
- Inner solver: ``LBFGSBSolver`` (multi-start L-BFGS-B, default) or
                ``EASolver`` (Differential Evolution)
- DoE        : Latin Hypercube Sampling (default) or uniform random
- GP subset  : fitness-based — best ``max_train_n`` points by scalarised
               augmented Tchebycheff using a fresh weight vector [1, line 24]
- Selection  : random (all batch candidates accepted) [3]
- Weight sampling: uniform Dirichlet (negative-log-uniform) [1]

References
----------
[1] J. Knowles, "ParEGO: a hybrid algorithm with on-line landscape
    approximation for expensive multiobjective optimization problems,"
    IEEE Trans. Evol. Comput., vol. 10, no. 1, pp. 50–66, Feb. 2006.
    https://doi.org/10.1109/TEVC.2005.851274

[2] D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient global
    optimization of expensive black-box functions," J. Glob. Optim.,
    vol. 13, no. 4, pp. 455–492, 1998.
    https://doi.org/10.1023/A:1008306431147

[3] M. Lukovic, Y. Tian, and W. Ma, "Diversity-guided multi-objective
    Bayesian optimization with batch evaluations," in Proc. NeurIPS,
    2020.  Source code: https://github.com/yunshengtian/DGEMO
    (``mobo/algorithms.py``, ``mobo/solver/parego/``,
    ``mobo/selection.py::Random``)

Acknowledgments
---------------
This implementation was developed with the assistance of GitHub Copilot.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .bo.base import BayesianOptimizer, SurrogateModel, InnerSolver
from .bo.surrogate import GaussianProcessSurrogate
from .bo.acquisition import expected_improvement
from .bo.solver import LBFGSBSolver, EASolver

# Augmented Tchebycheff regularisation weight (standard ParEGO literature)
_RHO: float = 0.05


# ---------------------------------------------------------------------------
# Acquisition function: scalarised EI
# ---------------------------------------------------------------------------

def _neg_scalarised_ei(
    x_norm: np.ndarray,
    surrogate: SurrogateModel,
    y_min_norm: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Negative augmented-Tchebycheff scalarisation of EI (for minimisation).

    The per-objective EI values are combined as:
        s = max_j(w_j · ei_j)  +  ρ · Σ_j(w_j · ei_j)

    Returns the *negative* so that ``scipy.optimize.minimize`` maximises EI.
    """
    x_2d = x_norm.reshape(1, -1)
    mu, std = surrogate.predict(x_2d, return_std=True)      # (1, m)
    ei      = expected_improvement(mu, std, y_min_norm)     # (1, m)
    wei     = weights * ei[0]                               # (m,)
    return float(-(np.max(wei) + _RHO * np.sum(wei)))


# ---------------------------------------------------------------------------
# Weight sampling
# ---------------------------------------------------------------------------

def _sample_weight(n_obj: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample uniformly from the (n_obj-1)-simplex via negative-log transform."""
    w = -np.log(rng.uniform(1e-12, 1.0, size=n_obj))
    return w / w.sum()


# ---------------------------------------------------------------------------
# ParEGO algorithm
# ---------------------------------------------------------------------------

class ParEGO(BayesianOptimizer):
    """ParEGO multi-objective Bayesian Optimisation.

    Inherits all infrastructure from ``BayesianOptimizer``; only the hooks
    below need to be implemented here.

    Parameters
    ----------
    n_restarts_inner : int
        Number of random restarts per candidate (L-BFGS-B solver only).
    inner_solver : {'lbfgsb', 'ea'}
        Inner solver for acquisition maximisation.
        ``'lbfgsb'`` — multi-start L-BFGS-B (default).
        ``'ea'``     — Differential Evolution; matches EVOLALG in the
                       ParEGO pseudocode (initialises from archive + random).
    ea_pop_size : int
        DE population size (EA solver only, default 20).
    ea_max_iter : int
        Maximum DE generations per candidate (EA solver only, default 100).
    doe : {'lhs', 'random'}
        Initial design of experiments strategy (forwarded to base).
        ``'lhs'`` — Latin Hypercube Sampling (default).
        ``'random'`` — uniform random.
    All other parameters are forwarded to ``BayesianOptimizer``.
    """

    def __init__(
        self,
        *args,
        n_restarts_inner: int = 3,
        inner_solver: str = 'lbfgsb',
        ea_pop_size: int = 20,
        ea_max_iter: int = 100,
        **kwargs,
    ) -> None:
        self.n_restarts_inner  = n_restarts_inner
        self.inner_solver_type = inner_solver
        self.ea_pop_size       = ea_pop_size
        self.ea_max_iter       = ea_max_iter
        super().__init__(*args, **kwargs)

    def _build_surrogate(self) -> GaussianProcessSurrogate:
        return GaussianProcessSurrogate(n_var=self.n_var)

    def _build_solver(self) -> InnerSolver:
        if self.inner_solver_type == 'ea':
            return EASolver(pop_size=self.ea_pop_size, max_iter=self.ea_max_iter)
        return LBFGSBSolver(n_restarts=self.n_restarts_inner)

    def _get_training_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Fitness-based subset selection (pseudocode DACE, line 24).

        When the archive exceeds ``max_train_n``, a fresh weight vector is
        sampled and the best ``max_train_n`` points by augmented Tchebycheff
        scalar fitness are kept for GP training.
        """
        N      = len(self.all_X)
        X_norm = self._norm_X(self.all_X)
        F_min  = self.all_F.min(axis=0)
        F_rng  = np.maximum(self.all_F.max(axis=0) - F_min, 1e-10)
        Y_norm = (self.all_F - F_min) / F_rng

        if N > self.max_train_n:
            w      = _sample_weight(self.n_obj, self.rng)         # (m,)
            wei    = w * Y_norm                                    # (N, m)
            scalar = np.max(wei, axis=1) + _RHO * np.sum(wei, axis=1)  # (N,)
            idx    = np.argsort(scalar)[:self.max_train_n]
            return X_norm[idx], Y_norm[idx]
        return X_norm, Y_norm

    def _ask(
        self,
        surrogate: SurrogateModel,
        y_min_norm: np.ndarray,
        X_norm: np.ndarray,
        Y_norm: np.ndarray,
    ) -> np.ndarray:
        """Generate one candidate per slot, each with a fresh weight vector."""
        weights = np.vstack([
            _sample_weight(self.n_obj, self.rng) for _ in range(self.pop_size)
        ])

        return self.solver.solve(
            surrogate=surrogate,
            y_min_norm=y_min_norm,
            n_candidates=self.pop_size,
            n_var=self.n_var,
            rng=self.rng,
            acq_fn=_neg_scalarised_ei,
            X_existing=X_norm,          # used by EASolver for archive-seeded init
            weights=weights,            # per-candidate weights forwarded via kw_c
        )


# ---------------------------------------------------------------------------
# Public benchmark entry-point
# ---------------------------------------------------------------------------

def run_parego(
    problem_name: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    n_obj: int = 2,
    n_var: Optional[int] = None,
    max_train_n: int = 500,
    doe: str = 'lhs',
    inner_solver: str = 'lbfgsb',
    n_restarts_inner: int = 3,
    ea_pop_size: int = 20,
    ea_max_iter: int = 100,
) -> dict:
    """Run ParEGO on a continuous pymoo benchmark.

    Drop-in replacement for other benchmark runners; returns a dict with keys
    ``indicators``, ``obj_pop``, ``var_pop``.

    Parameters
    ----------
    problem_name, seed, pop_size, n_gen, n_obj, n_var :
        Standard benchmark runner arguments.
    max_train_n : int
        Cap on GP training set size; excess points are pruned by
        fitness-based selection rather than random subsampling.
    doe : {'lhs', 'random'}
        Initial design strategy.  ``'lhs'`` (default) uses Latin Hypercube
        Sampling for better space coverage; ``'random'`` uses uniform random.
    inner_solver : {'lbfgsb', 'ea'}
        Acquisition maximiser.  ``'lbfgsb'`` (default) uses multi-start
        L-BFGS-B; ``'ea'`` uses Differential Evolution (EVOLALG-style).
    n_restarts_inner : int
        L-BFGS-B restarts per candidate (only when ``inner_solver='lbfgsb'``).
    ea_pop_size : int
        DE population size (only when ``inner_solver='ea'``).
    ea_max_iter : int
        Maximum DE generations per candidate (only when ``inner_solver='ea'``).
    """
    algo = ParEGO(
        problem_name=problem_name,
        n_obj=n_obj,
        n_var=n_var,
        seed=seed,
        pop_size=pop_size,
        n_gen=n_gen,
        max_train_n=max_train_n,
        doe=doe,
        inner_solver=inner_solver,
        n_restarts_inner=n_restarts_inner,
        ea_pop_size=ea_pop_size,
        ea_max_iter=ea_max_iter,
    )
    return algo.run()

