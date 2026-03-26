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
- Inner solver: ``LBFGSBSolver`` (multi-start L-BFGS-B)
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
from .bo.solver import LBFGSBSolver

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

    Inherits all infrastructure from ``BayesianOptimizer``; only three hooks
    need to be implemented here.

    Parameters
    ----------
    n_restarts_inner : int
        Number of random restarts for the inner L-BFGS-B per candidate.
    All other parameters are forwarded to ``BayesianOptimizer``.
    """

    def __init__(self, *args, n_restarts_inner: int = 3, **kwargs) -> None:
        self.n_restarts_inner = n_restarts_inner
        super().__init__(*args, **kwargs)

    def _build_surrogate(self) -> GaussianProcessSurrogate:
        return GaussianProcessSurrogate(n_var=self.n_var)

    def _build_solver(self) -> LBFGSBSolver:
        return LBFGSBSolver(n_restarts=self.n_restarts_inner)

    def _ask(
        self,
        surrogate: SurrogateModel,
        y_min_norm: np.ndarray,
        X_norm: np.ndarray,
        Y_norm: np.ndarray,
    ) -> np.ndarray:
        """Generate one candidate per slot, each with a fresh weight vector."""
        # Build one weight vector per candidate (shape: pop_size × n_obj)
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
    n_restarts_inner: int = 3,
) -> dict:
    """Run ParEGO on a continuous pymoo benchmark.

    Drop-in replacement for other benchmark runners; returns a dict with keys
    ``indicators``, ``obj_pop``, ``var_pop``.

    Parameters
    ----------
    problem_name, seed, pop_size, n_gen, n_obj, n_var :
        Standard benchmark runner arguments.
    max_train_n : int
        Cap on GP training set size (random subsample when exceeded).
    n_restarts_inner : int
        L-BFGS-B restarts per candidate (inner solver).
    """
    algo = ParEGO(
        problem_name=problem_name,
        n_obj=n_obj,
        n_var=n_var,
        seed=seed,
        pop_size=pop_size,
        n_gen=n_gen,
        max_train_n=max_train_n,
        n_restarts_inner=n_restarts_inner,
    )
    return algo.run()

