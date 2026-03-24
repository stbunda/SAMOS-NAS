"""
Inner pymoo Problem for the SAMOS surrogate loop on NASBench-201.

Objectives (in order): predicted ones first, real ones second.
  - Predicted objectives: approximated by the fitted surrogates.
  - Real objectives (e.g. flops): evaluated exactly via bench_db lookup
    at zero additional training cost.
"""

import numpy as np
from pymoo.core.problem import Problem

from problem.nasbench201_utils import (
    N_VAR, XL, XU, DATASET_INFO, _vec_to_arch_str,
)


class SurrogateProblem201(Problem):
    """
    Inner pymoo problem for the SAMOS surrogate loop on NASBench-201.

    Objectives (in order): predicted ones first, real ones second.
    - Predicted objectives: approximated by the fitted surrogates.
    - Real objectives (e.g. flops): evaluated exactly via bench_db lookup
      at zero additional training cost.
    """

    _SUPPORTED_REAL_OBJ = frozenset({'flops'})

    def __init__(
        self,
        surrogates,
        real_objectives: list,
        bench_db: dict,
        dataset: str = 'cifar10-valid',
        min_flops: float = None,
        max_flops: float = None,
        **kwargs,
    ):
        n_obj = len(surrogates) + len(real_objectives)
        super().__init__(
            n_var=N_VAR, n_obj=n_obj,
            xl=XL.astype(float), xu=XU.astype(float),
        )
        unsupported = set(real_objectives) - self._SUPPORTED_REAL_OBJ
        if unsupported:
            raise ValueError(
                f'Unsupported real objective(s) for NASBench-201: {unsupported}. '
                f'Supported: {self._SUPPORTED_REAL_OBJ}'
            )
        info = DATASET_INFO[dataset]
        self.surrogates      = surrogates
        self.real_objectives = real_objectives
        self.bench_db        = bench_db
        self.val_key         = info['val_key']
        self.min_flops       = min_flops
        self.max_flops       = max_flops

    def _evaluate(self, X, out, *args, **kwargs):
        n      = len(X)
        n_pred = len(self.surrogates)
        n_real = len(self.real_objectives)
        F      = np.zeros((n, n_pred + n_real))

        # Predicted objectives via surrogates
        X_float = np.round(X).astype(float)
        for i, surrogate in enumerate(self.surrogates):
            preds   = surrogate.predict(X_float)
            F[:, i] = np.clip(preds.squeeze(), 0.0, 1.0)

        # Real objectives via bench_db lookup (no training cost)
        for j, obj_name in enumerate(self.real_objectives):
            # obj_name == 'flops' (validated in __init__)
            for k, x in enumerate(X):
                arch_str = _vec_to_arch_str(np.round(x).astype(int))
                if arch_str in self.bench_db and self.val_key in self.bench_db[arch_str]:
                    flops = self.bench_db[arch_str][self.val_key]['flops']
                    F[k, n_pred + j] = (
                        (flops - self.min_flops) / (self.max_flops - self.min_flops)
                    )
                else:
                    F[k, n_pred + j] = 1.0   # penalise invalid / missing

        out['F'] = F
