"""
Inner pymoo Problem for the SAMOS surrogate loop on NASBench-101.

Objectives (in order): predicted ones first, real ones second.
  - Predicted objectives: approximated by the fitted surrogates.
  - Real objectives (e.g. n_params): evaluated exactly via bench_db lookup
    at zero additional training cost.
"""

import numpy as np
from pymoo.core.problem import Problem

from problem.nasbench101_utils import (
    N_VAR, XL, XU, MIN_PARAMS, MAX_PARAMS,
    _vec_to_arch_str,
)


class SurrogateProblem101(Problem):
    """
    Inner pymoo problem for the SAMOS surrogate loop.

    Objectives (in order): predicted ones first, real ones second.
    - Predicted objectives: approximated by the fitted surrogates.
    - Real objectives (e.g. n_params): evaluated exactly via bench_db lookup
      at zero additional training cost.
    """

    # Mapping from real-objective name -> how to compute it from a bench_db entry
    _REAL_OBJ_FNS = {
        'n_params': lambda e: (e['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS),
    }

    def __init__(self, surrogates, real_objectives: list, bench_db: dict, **kwargs):
        n_obj = len(surrogates) + len(real_objectives)
        super().__init__(
            n_var=N_VAR, n_obj=n_obj,
            xl=XL.astype(float), xu=XU.astype(float),
        )
        self.surrogates      = surrogates
        self.real_objectives = real_objectives
        self.bench_db        = bench_db

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        n_pred = len(self.surrogates)
        n_real = len(self.real_objectives)
        F = np.zeros((n, n_pred + n_real))

        # Predicted objectives via surrogates
        X_float = np.round(X).astype(float)
        for i, surrogate in enumerate(self.surrogates):
            preds = surrogate.predict(X_float)
            F[:, i] = np.clip(preds.squeeze(), 0.0, 1.0)

        # Real objectives via bench_db lookup
        for j, obj_name in enumerate(self.real_objectives):
            fn = self._REAL_OBJ_FNS.get(obj_name)
            if fn is None:
                raise ValueError(f'Unknown real objective: {obj_name}')
            for k, x in enumerate(X):
                arch_str = _vec_to_arch_str(np.round(x).astype(int), self.bench_db)
                if arch_str is not None and arch_str in self.bench_db:
                    F[k, n_pred + j] = fn(self.bench_db[arch_str])
                else:
                    F[k, n_pred + j] = 1.0   # penalise invalid

        out['F'] = F
