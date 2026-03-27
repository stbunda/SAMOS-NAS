"""Base abstractions and the generic Bayesian Optimisation run-loop.

Concrete algorithms (ParEGO, etc.) subclass ``BayesianOptimizer`` and only
override hook methods; the shared infrastructure handles:
  - initial DoE (uniform random)
  - iterative "fit → ask → evaluate → tell" loop
  - per-generation indicator bookkeeping (HV, IGD+)

Design notes
------------
- All decision variables are handled in the *normalised* space [0, 1]^d
  internally; the ``BayesianOptimizer`` maps back to the problem bounds
  before evaluation.
- Objectives are also normalised to [0, 1]^m before surrogate fitting so
  that acquisition functions are scale-invariant.

References
----------
[1] D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient global
    optimization of expensive black-box functions," J. Glob. Optim.,
    vol. 13, no. 4, pp. 455–492, 1998.
    https://doi.org/10.1023/A:1008306431147
    (Foundational BO loop: surrogate fit → acquisition maximise → evaluate.)

[2] M. Lukovic, Y. Tian, and W. Ma, "Diversity-guided multi-objective
    Bayesian optimization with batch evaluations," in Proc. NeurIPS, 2020.
    Source code: https://github.com/yunshengtian/DGEMO
    (``mobo/mobo.py`` — the MOBO base class that this module mirrors.)

Acknowledgments
---------------
This implementation was developed with the assistance of GitHub Copilot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np
from scipy.stats.qmc import LatinHypercube as _LHS
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from problem.pymoo.benchmark_utils import build_problem, get_pareto_front, default_ref_point


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class SurrogateModel(ABC):
    """Fits independent regression models, one per objective."""

    @abstractmethod
    def fit(self, X_norm: np.ndarray, Y_norm: np.ndarray) -> None:
        """Train on normalised inputs X_norm ∈ [0,1]^d and outputs Y_norm."""

    @abstractmethod
    def predict(
        self, X_norm: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return predicted means (N, m); optionally also stds (N, m)."""


class InnerSolver(ABC):
    """Finds the next candidate(s) by optimising an acquisition function."""

    @abstractmethod
    def solve(
        self,
        surrogate: SurrogateModel,
        y_min_norm: np.ndarray,
        n_candidates: int,
        n_var: int,
        rng: np.random.RandomState,
        X_existing: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Return candidate matrix of shape (n_candidates, n_var) in [0,1]^d.

        Parameters
        ----------
        X_existing : (N, n_var) array of current archive in normalised space.
            Solvers may use this to initialise their inner population (e.g. EA).
        """


# ---------------------------------------------------------------------------
# Generic Bayesian Optimisation run-loop
# ---------------------------------------------------------------------------


class BayesianOptimizer:
    """Backbone for sequential / batch Bayesian Optimisation on pymoo  benchmarks.

    Subclasses must implement ``_build_surrogate``, ``_build_solver``, and
    ``_ask``.  Everything else (DoE, evaluate, bookkeeping) is handled here.

    Parameters
    ----------
    problem_name, n_obj, n_var :
        Problem specification forwarded to ``benchmark_utils.build_problem``.
    seed :
        Master PRNG seed.
    pop_size :
        Number of candidates to evaluate per generation (batch size).
    n_gen :
        Total number of generations (including the initial DoE generation).
    max_train_n :
        Cap on the GP training set size (random subsample when exceeded).
    """

    def __init__(
        self,
        problem_name: str,
        n_obj: int = 2,
        n_var: Optional[int] = None,
        seed: int = 0,
        pop_size: int = 20,
        n_gen: int = 50,
        max_train_n: int = 500,
        doe: str = 'lhs',
    ) -> None:
        self.problem_name = problem_name
        self.seed         = seed
        self.pop_size     = pop_size
        self.n_gen        = n_gen
        self.max_train_n  = max_train_n
        self.doe          = doe

        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)

        self.prob      = build_problem(problem_name, n_obj, n_var)
        self.pf        = get_pareto_front(self.prob, self.prob.n_obj)
        self.ref_point = default_ref_point(problem_name, self.prob.n_obj)
        self.hv_ind    = HV(ref_point=self.ref_point)
        self.igd_ind   = IGDPlus(self.pf)

        self.n_var = self.prob.n_var
        self.n_obj = self.prob.n_obj
        self.xl    = self.prob.xl.copy()
        self.xu    = self.prob.xu.copy()

        self.surrogate: SurrogateModel = self._build_surrogate()
        self.solver:    InnerSolver    = self._build_solver()

        # Running archive (decision variables + objectives)
        self.all_X: Optional[np.ndarray] = None
        self.all_F: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Hooks – subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_surrogate(self) -> SurrogateModel:
        """Instantiate and return the surrogate model."""

    @abstractmethod
    def _build_solver(self) -> InnerSolver:
        """Instantiate and return the inner solver."""

    @abstractmethod
    def _ask(
        self,
        surrogate: SurrogateModel,
        y_min_norm: np.ndarray,
        X_norm: np.ndarray,
        Y_norm: np.ndarray,
    ) -> np.ndarray:
        """Return (pop_size, n_var) candidate matrix in *normalised* [0,1]^d.

        Called after the surrogate has already been fitted for the current
        generation.  ``y_min_norm`` is the per-objective minimum of Y_norm.
        """

    # ------------------------------------------------------------------
    # Shared infrastructure
    # ------------------------------------------------------------------

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the real problem; returns (N, n_obj) array."""
        out: dict = {}
        self.prob._evaluate(X, out)
        return out['F'].copy()

    def _norm_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.xl) / (self.xu - self.xl)

    def _denorm_X(self, X_norm: np.ndarray) -> np.ndarray:
        return np.clip(self.xl + X_norm * (self.xu - self.xl), self.xl, self.xu)

    def _get_training_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (possibly subsampled) normalised training arrays."""
        N       = len(self.all_X)
        X_norm  = self._norm_X(self.all_X)
        F_min   = self.all_F.min(axis=0)
        F_rng   = np.maximum(self.all_F.max(axis=0) - F_min, 1e-10)
        Y_norm  = (self.all_F - F_min) / F_rng

        if N > self.max_train_n:
            idx    = self.rng.choice(N, self.max_train_n, replace=False)
            return X_norm[idx], Y_norm[idx]
        return X_norm, Y_norm

    def _record_gen(
        self,
        X_batch: np.ndarray,
        F_batch: np.ndarray,
        indicators: list,
        obj_pop: list,
        var_pop: list,
    ) -> None:
        nd_idx = NonDominatedSorting().do(self.all_F, only_non_dominated_front=True)
        nd_F   = self.all_F[nd_idx]
        indicators.append({
            'hv':       float(self.hv_ind(nd_F)),
            'igd_plus': float(self.igd_ind(nd_F)),
        })
        obj_pop.append(F_batch.copy())
        var_pop.append(X_batch.copy())

    def run(self) -> dict:
        """Execute the full optimisation and return the results dict.

        Returns
        -------
        dict with keys:
          ``indicators`` — list of dicts with ``hv`` and ``igd_plus`` per gen
          ``obj_pop``    — list of (pop_size, n_obj) arrays per gen
          ``var_pop``    — list of (pop_size, n_var) arrays per gen
        """
        indicators: list = []
        obj_pop:    list = []
        var_pop:    list = []

        # ── generation 0: initial DoE ────────────────────────────────────────
        if self.doe == 'lhs':
            lhs_seed = int(self.rng.randint(0, 2 ** 31))
            X_unit = _LHS(d=self.n_var, seed=lhs_seed).random(n=self.pop_size)
        else:
            X_unit = self.rng.uniform(size=(self.pop_size, self.n_var))
        X_init = self.xl + X_unit * (self.xu - self.xl)
        F_init = self._evaluate(X_init)
        self.all_X = X_init.copy()
        self.all_F = F_init.copy()
        self._record_gen(X_init, F_init, indicators, obj_pop, var_pop)

        # ── generations 1 … n_gen-1 ─────────────────────────────────────────
        for gen in range(1, self.n_gen):
            X_norm, Y_norm = self._get_training_set()
            self.surrogate.fit(X_norm, Y_norm)
            y_min_norm = Y_norm.min(axis=0)

            X_cand_norm = self._ask(self.surrogate, y_min_norm, X_norm, Y_norm)
            X_cand      = self._denorm_X(X_cand_norm)
            F_cand      = self._evaluate(X_cand)

            self.all_X = np.vstack([self.all_X, X_cand])
            self.all_F = np.vstack([self.all_F, F_cand])
            self._record_gen(X_cand, F_cand, indicators, obj_pop, var_pop)

            print(
                f'[{self.__class__.__name__}] gen {gen:4d}/{self.n_gen - 1}'
                f'  hv={indicators[-1]["hv"]:.4f}'
                f'  igd+={indicators[-1]["igd_plus"]:.4f}'
            )

        return {'indicators': indicators, 'obj_pop': obj_pop, 'var_pop': var_pop}
