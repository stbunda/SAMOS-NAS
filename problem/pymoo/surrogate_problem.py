"""
Inner pymoo Problem for the SAMOS surrogate loop on continuous MOO benchmarks.

All objectives are approximated by the fitted surrogates (no cheap real-eval
objective, unlike the NASBench surrogate problems).  The problem shares
n_var / xl / xu with the outer (real) problem so that the inner NSGA-II
explores the same decision space.
"""

import numpy as np
from pymoo.core.problem import Problem


class SurrogateProblemMOO(Problem):
    """
    Continuous inner problem wrapping one fitted surrogate per objective.

    Parameters
    ----------
    surrogates : list
        Fitted surrogate models (one per objective).
        Each must implement ``predict(X) -> np.ndarray``.
    n_var : int
        Number of decision variables (must match the real problem).
    xl : np.ndarray
        Lower bounds for decision variables.
    xu : np.ndarray
        Upper bounds for decision variables.
    """

    def __init__(
        self,
        surrogates: list,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
    ):
        super().__init__(n_var=n_var, n_obj=len(surrogates), xl=xl, xu=xu)
        self.surrogates = surrogates

    def _evaluate(self, X, out, *args, **kwargs):
        out['F'] = np.column_stack([s.predict(X) for s in self.surrogates])
