"""
Inner pymoo Problem for the SAMOS surrogate loop on evoxbench benchmarks.

Objectives are split into:
  - Predicted objectives: approximated by fitted surrogates (cheaper).
  - Real objectives: evaluated directly via benchmark.evaluate() at the
    surrogate-optimisation stage (still cheap — evoxbench is a lookup).

Columns in out['F'] are assembled in the *original* objective index order
(0, 1, … n_obj-1), not predicted-first, so that the inner problem's
objective semantics match the outer EvoXBenchProblem exactly.
"""

import numpy as np
from pymoo.core.problem import Problem


class SurrogateProblemEvox(Problem):
    """
    Inner pymoo problem for the SAMOS surrogate loop on evoxbench.

    Parameters
    ----------
    surrogates : list
        Fitted surrogate models, one per predicted objective.
        Each must implement ``.predict(X) -> np.ndarray``.
    predict_obj_indices : list[int]
        Column indices (in the outer problem's F) for the predicted objectives.
    real_obj_indices : list[int]
        Column indices (in the outer problem's F) for the real objectives.
    benchmark :
        The evoxbench benchmark instance; used for direct real-obj evaluation.
    """

    def __init__(
        self,
        surrogates: list,
        predict_obj_indices: list,
        real_obj_indices: list,
        benchmark,
    ):
        ss = benchmark.search_space
        n_obj = len(predict_obj_indices) + len(real_obj_indices)
        super().__init__(
            n_var=ss.n_var,
            n_obj=n_obj,
            xl=np.asarray(ss.lb, dtype=float),
            xu=np.asarray(ss.ub, dtype=float),
        )
        self.surrogates         = surrogates
        self.predict_obj_indices = predict_obj_indices
        self.real_obj_indices   = real_obj_indices
        self.benchmark          = benchmark

    def _evaluate(self, X, out, *args, **kwargs):
        n     = len(X)
        n_obj = len(self.predict_obj_indices) + len(self.real_obj_indices)
        F     = np.zeros((n, n_obj))

        # ── real objectives via benchmark (single batch call) ─────────────────
        if self.real_obj_indices:
            X_int    = np.round(X).astype(int)
            F_real   = self.benchmark.evaluate(X_int, true_eval=False)
            if not self.benchmark.normalized_objectives:
                F_real = self.benchmark.normalize(F_real)
            F_real   = np.where(np.isfinite(F_real), F_real, 1.0)
            for col, orig_idx in enumerate(self.real_obj_indices):
                F[:, orig_idx] = F_real[:, orig_idx]

        # ── predicted objectives via surrogates ───────────────────────────────
        X_float = X.astype(float)
        for surrogate, orig_idx in zip(self.surrogates, self.predict_obj_indices):
            preds = surrogate.predict(X_float)
            F[:, orig_idx] = preds.squeeze()

        out['F'] = F
