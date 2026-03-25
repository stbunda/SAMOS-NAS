"""SSA-NSGA-II algorithm variants for the pymoo benchmark.

Exports
-------
SSANSGA2         — original pysamoo class (re-exported for convenience)
SklearnSSANSGA2  — SSA-NSGA-II variant that replaces the default surrogate with sklearn models
"""

from pysamoo.algorithms.ssansga2 import SSANSGA2  # noqa: F401  (re-export)
from strategy.algorithm.gpsaf import _sklearn_surrogate

__all__ = ["SSANSGA2", "SklearnSSANSGA2"]


class SklearnSSANSGA2(SSANSGA2):
    """SSA-NSGA-II variant that injects sklearn surrogates after _setup resolves the problem."""

    def __init__(self, sklearn_models, **kw):
        super().__init__(**kw)
        self._sklearn_models = sklearn_models

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.surrogate = _sklearn_surrogate(problem, self._sklearn_models)
