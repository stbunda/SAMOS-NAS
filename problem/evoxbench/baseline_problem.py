"""
Outer pymoo Problem wrapping an evoxbench benchmark.
"""

import time

import numpy as np
from pymoo.core.problem import Problem


class EvoXBenchProblem(Problem):
    """
    Pymoo problem wrapper for any evoxbench benchmark.

    Bounds, n_var, and n_obj are read directly from the benchmark object.
    ``_evaluate`` delegates entirely to ``benchmark.evaluate()``.

    Parameters
    ----------
    benchmark :
        An evoxbench Benchmark instance (e.g. from ``c10mop(pid)``).
    """

    def __init__(self, benchmark, no_norm: bool = False, **kwargs):
        ss = benchmark.search_space
        super().__init__(
            n_var=ss.n_var,
            n_obj=benchmark.evaluator.n_objs,
            xl=np.asarray(ss.lb, dtype=float),
            xu=np.asarray(ss.ub, dtype=float),
            **kwargs,
        )
        self.benchmark = benchmark
        self.no_norm   = no_norm
        self.time: list = [time.time()]   # wall-clock timestamps (PDNS-compatible)
        self.n_eval_calls: int = 0

    def _evaluate(self, X, out, *args, **kwargs):
        X_int = np.round(X).astype(int)
        F = self.benchmark.evaluate(X_int, true_eval=False)
        if not self.no_norm and not self.benchmark.normalized_objectives:
            F = self.benchmark.normalize(F)
        # Replace non-finite values (inf/NaN from invalid NB-101 archs) with
        # a large but finite penalty so surrogates can train on the archive.
        F = np.where(np.isfinite(F), F, 1.0)
        out['F'] = F

        self.n_eval_calls += len(X_int)
        self.time.append(time.time())
