"""
Constrained pymoo Problems for the S1-S4 constraint-scenario campaign.

Two classes, mirroring the unconstrained pair
(``baseline_problem.EvoXBenchProblem`` / ``surrogate_problem.SurrogateProblemEvox``)
but adding a single inequality constraint ``G`` derived from one benchmark
metric column:

    G = (metric - threshold) / threshold              (feasible <=> G <= 0)

The metric and threshold live in the *same* space that ``benchmark.evaluate``
returns after the baseline convention (``normalize()`` only when
``not benchmark.normalized_objectives``). For c10mop the objectives are
already normalised, so no ``normalize()`` is applied and G is taken straight
from the evaluated matrix -- calling ``normalize()`` on top would
double-normalise. The constraint
column and the objective columns therefore come from one shared, once-
converted, non-finite->1.0-guarded matrix.

Because the outer problem declares ``n_ieq_constr=1``, every evaluated
individual carries ``G`` / ``CV`` / ``feasible``, so pymoo's
``RankAndCrowding`` survival (outer archive selection *and* the inner
NSGA-II) automatically becomes feasibility-first (Deb's CDP) with no extra
wiring -- verified against pymoo 0.6.1.1 ``core/survival.py``:
``Survival.do`` splits by feasibility whenever ``filter_infeasible`` (True
for RankAndCrowding) and ``problem.has_constraints()`` hold.

Column semantics
----------------
``obj_indices`` selects *which* benchmark metric columns become the search
objectives, and in *what order* they appear in ``out['F']``. Both classes
emit ``len(obj_indices)`` objective columns in exactly that order, so the
inner surrogate problem's F matches the outer problem's F column-for-column.
The constraint metric (``constr_index``) is a benchmark column that need not
be among ``obj_indices``.
"""

import numpy as np
from pymoo.core.problem import Problem


def _violation(metric, threshold):
    """G = (metric - T) / T, with non-finite mapped to a positive (infeasible)
    but finite violation so CV maths and any constraint surrogate stay finite.

    ponytail: fallback is a flat 1.0 (= 100% over threshold) rather than the
    true magnitude; callers already pass a finite (non-finite->1.0 guarded)
    metric, so this only triggers for a threshold-driven division edge case.
    """
    g = (metric - threshold) / threshold
    return np.where(np.isfinite(g), g, 1.0)


class ConstrainedEvoXBenchProblem(Problem):
    """Outer pymoo problem: benchmark evaluated once per call.

    Parameters
    ----------
    benchmark :
        An evoxbench Benchmark instance (e.g. from ``c10mop(pid)``).
    obj_indices : list[int]
        Benchmark metric columns used as objectives, in output-column order.
        ``out['F']`` has ``len(obj_indices)`` columns in this order.
    constr_index : int
        Benchmark metric column used as the constrained metric.
    threshold : float
        Constraint threshold in the evaluated-metric space (post baseline
        convention). ``G <= 0`` <=> feasible.
    no_norm : bool
        Skip objective normalisation (matches EvoXBenchProblem.no_norm).
    """

    def __init__(self, benchmark, obj_indices, constr_index, threshold,
                 no_norm: bool = False, **kwargs):
        ss = benchmark.search_space
        self.obj_indices  = list(obj_indices)
        self.constr_index = int(constr_index)
        self.threshold    = float(threshold)
        super().__init__(
            n_var=ss.n_var,
            n_obj=len(self.obj_indices),
            n_ieq_constr=1,
            xl=np.asarray(ss.lb, dtype=float),
            xu=np.asarray(ss.ub, dtype=float),
            **kwargs,
        )
        self.benchmark   = benchmark
        self.no_norm     = no_norm
        self.n_eval_calls = 0

    def _evaluate(self, X, out, *args, **kwargs):
        X_int = np.round(X).astype(int)
        # One benchmark call; apply the baseline convention once, then split
        # the same matrix into objective and constraint columns.
        F = self.benchmark.evaluate(X_int, true_eval=False)
        if not self.no_norm and not self.benchmark.normalized_objectives:
            F = self.benchmark.normalize(F)
        F = np.where(np.isfinite(F), F, 1.0)

        out['F'] = F[:, self.obj_indices]
        out['G'] = _violation(F[:, self.constr_index], self.threshold)[:, None]

        self.n_eval_calls += len(X_int)


class ConstrainedSurrogateProblemEvox(Problem):
    """Inner surrogate problem: predicted objectives + real (cheap) objectives,
    plus one constraint that is either predicted by a surrogate or computed
    exactly via the benchmark.

    Objective handling mirrors ``SurrogateProblemEvox`` but works in the
    *reduced* objective space defined by ``obj_indices`` (so its F columns
    match ``ConstrainedEvoXBenchProblem`` exactly).

    Parameters
    ----------
    surrogates : list
        Fitted surrogates, one per predicted objective, aligned with
        ``predict_obj_indices``.
    obj_indices : list[int]
        Benchmark metric columns used as objectives, in output-column order
        (same list given to the outer problem).
    predict_obj_indices : list[int]
        **Output-column positions** (indices into ``obj_indices`` / into the
        outer problem's F) filled by ``surrogates``. These are the same
        positions SAMOS2 fits its surrogates against, so surrogate ``i``
        predicts output column ``predict_obj_indices[i]``.
    real_obj_indices : list[int]
        Output-column positions filled from a real benchmark lookup. The
        benchmark column read for position ``p`` is ``obj_indices[p]``.
    benchmark :
        The evoxbench benchmark instance.
    constr_index : int
        Benchmark metric column for the constraint (evaluated-metric space).
    threshold : float
        Constraint threshold in the evaluated-metric space.
    constr_surrogate : object or None
        If given (constr mode 'surrogate'), ``G = constr_surrogate.predict(X)``.
        If None, G is computed exactly via the benchmark (constr mode 'exact').
    no_norm : bool
        Skip objective normalisation.
    """

    def __init__(
        self,
        surrogates: list,
        obj_indices: list,
        predict_obj_indices: list,
        real_obj_indices: list,
        benchmark,
        constr_index: int,
        threshold: float,
        constr_surrogate=None,
        no_norm: bool = False,
    ):
        ss = benchmark.search_space
        self.obj_indices         = list(obj_indices)
        self.predict_obj_indices = list(predict_obj_indices)
        self.real_obj_indices    = list(real_obj_indices)
        self.constr_index        = int(constr_index)
        self.threshold           = float(threshold)
        self.constr_surrogate    = constr_surrogate
        super().__init__(
            n_var=ss.n_var,
            n_obj=len(self.obj_indices),
            n_ieq_constr=1,
            xl=np.asarray(ss.lb, dtype=float),
            xu=np.asarray(ss.ub, dtype=float),
        )
        self.surrogates = surrogates
        self.benchmark  = benchmark
        self.no_norm    = no_norm

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        F = np.zeros((n, self.n_obj))

        # A single benchmark call (baseline convention applied once) covers
        # both the real objectives and an exact constraint -- avoid a double
        # lookup and keep the same metric space as the outer problem.
        need_bench = bool(self.real_obj_indices) or (self.constr_surrogate is None)
        F_bench = None
        if need_bench:
            X_int = np.round(X).astype(int)
            F_bench = self.benchmark.evaluate(X_int, true_eval=False)
            if not self.no_norm and not self.benchmark.normalized_objectives:
                F_bench = self.benchmark.normalize(F_bench)
            F_bench = np.where(np.isfinite(F_bench), F_bench, 1.0)

        # ── real (cheap) objectives via benchmark ─────────────────────────────
        for pos in self.real_obj_indices:
            F[:, pos] = F_bench[:, self.obj_indices[pos]]

        # ── predicted objectives via surrogates ───────────────────────────────
        X_float = X.astype(float)
        for surrogate, pos in zip(self.surrogates, self.predict_obj_indices):
            F[:, pos] = np.asarray(surrogate.predict(X_float)).squeeze()

        out['F'] = F

        # ── constraint: predicted or exact ────────────────────────────────────
        if self.constr_surrogate is not None:
            G = np.asarray(self.constr_surrogate.predict(X_float)).reshape(n, 1)
        else:
            G = _violation(F_bench[:, self.constr_index], self.threshold)[:, None]
        out['G'] = G
