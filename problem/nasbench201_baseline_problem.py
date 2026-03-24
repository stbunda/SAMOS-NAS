"""
Pymoo Problem wrapper for the categorical NASBench-201 search space.

Objectives (both minimised):
  - val_acc error  (1 − val_acc@epoch200 / 100)
  - FLOPs normalised to [0, 1]

Dataset is selected at construction time; the same problem instance is used
throughout a single run.
"""

import time

import numpy as np
from pymoo.core.problem import Problem

from problem.nasbench201_utils import (
    N_VAR, XL, XU, DATASET_INFO, _vec_to_arch_str,
)


class NASBench201Problem(Problem):
    """
    Pymoo problem over the 6-gene categorical NASBench-201 search space.

    Time tracking mirrors PDNS: self.time is a list of
    wall_clock + cumulative_estimated_training_time values, one per
    *unique* architecture evaluation.  self.log_archs stores each
    evaluated vector (to avoid double-counting duplicates).
    """

    TRAIN_TIME_ESTIMATE = 1444.0   # seconds — typical 200-epoch training fallback

    def __init__(
        self,
        bench_db: dict,
        dataset: str = 'cifar10-valid',
        min_flops: float = None,
        max_flops: float = None,
        **kwargs,
    ):
        super().__init__(
            n_var=N_VAR, n_obj=2,
            xl=XL.astype(float), xu=XU.astype(float),
        )
        info = DATASET_INFO[dataset]
        self.bench_db  = bench_db
        self.val_key   = info['val_key']
        self.min_flops = min_flops
        self.max_flops = max_flops
        self.cum_time  = 0.0
        self.time      = [time.time()]   # start timestamp (PDNS-compatible)
        self.log_archs: list = []

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        F = np.zeros((n, 2))

        for i, vec in enumerate(X):
            vec_int  = np.round(vec).astype(int)
            vec_int  = np.clip(vec_int, XL, XU)
            arch_str = _vec_to_arch_str(vec_int)

            if arch_str not in self.bench_db or self.val_key not in self.bench_db[arch_str]:
                F[i, 0] = 1.0
                F[i, 1] = 1.0
                continue

            entry      = self.bench_db[arch_str][self.val_key]
            val_err    = 1.0 - entry['epoch-200'] / 100.0
            flops_norm = (entry['flops'] - self.min_flops) / (self.max_flops - self.min_flops)

            F[i, 0] = val_err
            F[i, 1] = flops_norm

            # time tracking (PDNS-compatible)
            if vec_int.tolist() not in self.log_archs:
                train_time = entry.get('time-200', self.TRAIN_TIME_ESTIMATE)
                self.cum_time += float(train_time)
                self.log_archs.append(vec_int.tolist())

            self.time.append(time.time() + self.cum_time)

        out['F'] = F
