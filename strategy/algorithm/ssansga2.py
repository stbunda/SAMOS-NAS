"""SSA-NSGA-II algorithm variants for the pymoo benchmark.

Exports
-------
SSANSGA2                  — original pysamoo class (re-exported for convenience)
SklearnSSANSGA2           — SSA-NSGA-II with sklearn surrogates for all objectives
SklearnSSANSGA2CheapReal  — SSA-NSGA-II with sklearn surrogates for expensive objectives
                            and direct benchmark lookups for cheap objectives (params, FLOPs)
"""

import numpy as np
from pysamoo.algorithms.ssansga2 import SSANSGA2  # noqa: F401  (re-export)
from strategy.algorithm.gpsaf import _sklearn_surrogate, _SklearnTarget

__all__ = ["SSANSGA2", "SklearnSSANSGA2", "SklearnSSANSGA2CheapReal"]


class SklearnSSANSGA2(SSANSGA2):
    """SSA-NSGA-II variant that injects sklearn surrogates after _setup resolves the problem."""

    def __init__(self, sklearn_models, **kw):
        super().__init__(**kw)
        self._sklearn_models = sklearn_models

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.surrogate = _sklearn_surrogate(problem, self._sklearn_models)


# ─── cheap-real target & surrogate ────────────────────────────────────────────

class _BenchmarkTarget:
    """Pysamoo Target that evaluates one objective via a benchmark lookup table.

    No fitting is required — the benchmark is a fast lookup (EvoXBench), so
    calling it directly during the inner SSA surrogate loop is effectively free.
    """

    def __init__(self, label, benchmark, no_norm: bool = False):
        self.label      = label      # ('F', orig_obj_idx)
        self._benchmark = benchmark
        self._no_norm   = no_norm
        self.best       = 'benchmark'
        self.obj        = None       # kept for interface compatibility

    def validate(self, trn=None, tst=None, find_best=True, **kwargs):
        pass

    def fit(self, sols):
        pass  # lookup table — nothing to train

    def predict(self, X, out):
        X_int  = np.round(X).astype(int)
        F_real = self._benchmark.evaluate(X_int, true_eval=False)
        if not self._no_norm and not self._benchmark.normalized_objectives:
            F_real = self._benchmark.normalize(F_real)
        F_real = np.where(np.isfinite(F_real), F_real, 1.0)
        key, idx = self.label
        out.get(key)[:, [idx]] = F_real[:, [idx]]

    def performance(self, indicator, model=None, func=None):
        return 0.0


def _sklearn_surrogate_cheap(
    problem,
    sklearn_models,
    predict_obj_indices,
    real_obj_indices,
    benchmark,
    no_norm: bool = False,
):
    """Build a pysamoo Surrogate with mixed sklearn + benchmark targets.

    Parameters
    ----------
    problem             : pymoo Problem (used to initialise the Surrogate).
    sklearn_models      : list of models, one per entry in *predict_obj_indices*.
    predict_obj_indices : objective columns to approximate with sklearn models.
    real_obj_indices    : objective columns to evaluate directly via *benchmark*.
    benchmark           : EvoXBench benchmark instance (fast lookup table).
    no_norm             : pass-through for benchmark normalisation flag.
    """
    from pysamoo.core.surrogate import Surrogate

    real_set   = set(real_obj_indices)
    model_iter = iter(sklearn_models)
    targets    = []
    for orig_idx in sorted(predict_obj_indices + real_obj_indices):
        if orig_idx in real_set:
            targets.append(_BenchmarkTarget(('F', orig_idx), benchmark, no_norm))
        else:
            targets.append(_SklearnTarget(('F', orig_idx), next(model_iter)))
    return Surrogate(problem, targets)


class SklearnSSANSGA2CheapReal(SSANSGA2):
    """SSA-NSGA-II with a mixed surrogate: sklearn models for expensive objectives,
    direct benchmark lookups for cheap objectives (params, FLOPs, latency, etc.)."""

    def __init__(
        self,
        sklearn_models,
        predict_obj_indices,
        real_obj_indices,
        benchmark,
        no_norm: bool = False,
        **kw,
    ):
        super().__init__(**kw)
        self._sklearn_models      = sklearn_models
        self._predict_obj_indices = predict_obj_indices
        self._real_obj_indices    = real_obj_indices
        self._benchmark           = benchmark
        self._no_norm             = no_norm

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.surrogate = _sklearn_surrogate_cheap(
            problem,
            self._sklearn_models,
            self._predict_obj_indices,
            self._real_obj_indices,
            self._benchmark,
            self._no_norm,
        )
