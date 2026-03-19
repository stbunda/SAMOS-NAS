"""
Pymoo Problem wrapper for the integer-vector NASBench-101 search space.

Objectives (both minimised):
  - val_acc_12 error  (1 − val_acc@epoch12)
  - n_params normalised to [0, 1]
"""

import time

import numpy as np
from pymoo.core.problem import Problem

from problem.nasbench101_utils import (
    N_VAR, XL, XU, MIN_PARAMS, MAX_PARAMS,
    _vec_to_arch_str,
)


class NASBench101Problem(Problem):
    """
    Pymoo problem over the 26-gene integer NASBench-101 search space.

    Time tracking mirrors PDNS: self.time is a list of
    wall_clock + cumulative_estimated_training_time values, one per
    *unique* architecture evaluation.  self.log_archs stores each
    evaluated vector (to avoid double-counting duplicates).
    """

    TRAIN_TIME_ESTIMATE = 0.30215823150349097   # seconds — PDNS n_params estimate

    def __init__(self, bench_db: dict, **kwargs):
        super().__init__(
            n_var=N_VAR, n_obj=2,
            xl=XL.astype(float), xu=XU.astype(float),
        )
        self.bench_db  = bench_db
        self.cum_time  = 0.0
        self.time      = [time.time()]   # start timestamp
        self.log_archs: list = []

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        F = np.zeros((n, 2))

        for i, vec in enumerate(X):
            vec_int = np.round(vec).astype(int)
            vec_int = np.clip(vec_int, XL, XU)

            arch_str = _vec_to_arch_str(vec_int, self.bench_db)
            if arch_str is None or arch_str not in self.bench_db:
                # invalid architecture — penalise
                F[i, 0] = 1.0
                F[i, 1] = 1.0
                continue

            entry         = self.bench_db[arch_str]
            val_err       = 1.0 - entry['val_acc_12']
            n_params      = entry['n_params']
            n_params_norm = (n_params - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)

            F[i, 0] = val_err
            F[i, 1] = n_params_norm

            # time tracking (PDNS-compatible)
            if vec_int.tolist() not in self.log_archs:
                train_time = entry.get('train_time_12', self.TRAIN_TIME_ESTIMATE)
                self.cum_time += float(train_time) + self.TRAIN_TIME_ESTIMATE
                self.log_archs.append(vec_int.tolist())

            self.time.append(time.time() + self.cum_time)

        out['F'] = F
