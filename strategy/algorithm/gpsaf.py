"""GPSAF-NSGA-II algorithm variants for the pymoo benchmark.

Exports
-------
GPSAF               — original pysamoo class (re-exported for convenience)
_SklearnTarget      — minimal pysamoo Target wrapper around any sklearn estimator
_sklearn_surrogate  — build a pysamoo Surrogate from a list of sklearn models
SklearnGPSAF        — GPSAF variant that replaces the default surrogate with sklearn models
"""

from copy import deepcopy

import numpy as np
from pysamoo.algorithms.gpsaf import GPSAF  # noqa: F401  (re-export)

__all__ = ["GPSAF", "_SklearnTarget", "_sklearn_surrogate", "SklearnGPSAF"]


class _SklearnTarget:
    """Minimal pysamoo Target wrapping one sklearn estimator.

    Bypasses ezmodel's cross-validation: the estimator is always used as-is.
    """

    def __init__(self, label, model):
        self.label = label
        self._tmpl = deepcopy(model)
        self.best  = 'sklearn'
        self.obj   = None

    def validate(self, trn=None, tst=None, find_best=True, **kwargs):
        pass

    def fit(self, sols):
        key, idx = self.label
        X = sols.get('X')
        y = sols.get(key)[:, idx]
        m = deepcopy(self._tmpl)
        m.fit(X, y)
        self.obj = m

    def predict(self, X, out):
        v = self.obj.predict(X).reshape(-1, 1)
        key, idx = self.label
        out.get(key)[:, [idx]] = v

    def performance(self, indicator, model=None, func=None):
        return 0.0


def _sklearn_surrogate(problem, sklearn_models):
    """Build a pysamoo Surrogate from a list of sklearn models (one per objective)."""
    from pysamoo.core.surrogate import Surrogate
    targets = [_SklearnTarget(('F', i), m) for i, m in enumerate(sklearn_models)]
    return Surrogate(problem, targets)


class SklearnGPSAF(GPSAF):
    """GPSAF variant that injects sklearn surrogates after _setup resolves the problem."""

    def __init__(self, base_algorithm, sklearn_models, **kw):
        super().__init__(base_algorithm, **kw)
        self._sklearn_models = sklearn_models

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.surrogate = _sklearn_surrogate(problem, self._sklearn_models)
