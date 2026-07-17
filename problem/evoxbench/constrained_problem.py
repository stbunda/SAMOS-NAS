"""
Constrained pymoo Problems for the constraint-scenario campaigns.

Two classes, mirroring the unconstrained pair
(``baseline_problem.EvoXBenchProblem`` / ``surrogate_problem.SurrogateProblemEvox``)
but adding one inequality constraint ``G`` column per constrained benchmark
metric:

    G_j = (metric_j - threshold_j) / threshold_j      (feasible <=> all G_j <= 0)

``constr_index`` / ``threshold`` accept either a scalar (single constraint,
the S1-S4 campaign -- behaviour and shapes bit-identical to the original
single-constraint classes) or parallel sequences (one G column per entry,
``n_ieq_constr = len(constr_index)``).

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


def _as_constraint_lists(constr_index, threshold):
    """Normalize scalar-or-sequence constraint specs to parallel lists."""
    indices    = [int(i) for i in np.atleast_1d(constr_index)]
    thresholds = [float(t) for t in np.atleast_1d(threshold)]
    if len(indices) != len(thresholds):
        raise ValueError(
            f'constr_index and threshold must have equal length, got '
            f'{len(indices)} vs {len(thresholds)}')
    return indices, thresholds


def _violation_columns(F, constr_indices, thresholds):
    """(n, n_constr) violation matrix, one column per constrained metric."""
    return np.column_stack([_violation(F[:, i], t)
                            for i, t in zip(constr_indices, thresholds)])


class ConstrainedEvoXBenchProblem(Problem):
    """Outer pymoo problem: benchmark evaluated once per call.

    Parameters
    ----------
    benchmark :
        An evoxbench Benchmark instance (e.g. from ``c10mop(pid)``).
    obj_indices : list[int]
        Benchmark metric columns used as objectives, in output-column order.
        ``out['F']`` has ``len(obj_indices)`` columns in this order.
    constr_index : int or sequence[int]
        Benchmark metric column(s) used as the constrained metric(s). A
        sequence declares one G column per entry (``n_ieq_constr = len``).
    threshold : float or sequence[float]
        Constraint threshold(s) in the evaluated-metric space (post baseline
        convention), parallel to ``constr_index``. ``G <= 0`` <=> feasible;
        an individual is feasible iff EVERY column satisfies its threshold.
    no_norm : bool
        Skip objective normalisation (matches EvoXBenchProblem.no_norm).
    gate : bool
        Hard-evaluability gate. When True, the F rows of infeasible
        individuals (any G > 0) are masked to ``np.inf`` AFTER the non-finite
        guard: under the gated-hard semantics those objective values are
        physically unobservable (the model cannot run), and inf is the
        fail-safe sentinel -- if a gated row ever leaks past the algorithm's
        archive gate, every comparison ranks it strictly worst (death-penalty
        semantics) instead of silently corrupting dominance sorts the way NaN
        would. G stays real either way (the gate needs it, and for a cheap /
        simulated metric the violation IS observable). The invalid-arch
        fallback (non-finite -> 1.0) feeds the same gate: a guarded row's
        constraint metric of 1.0 exceeds every campaign threshold, so
        can't-even-build architectures are gated out too, not laundered into
        a plausible fitness. Default False = ungated behaviour, bit-exact.
    """

    def __init__(self, benchmark, obj_indices, constr_index, threshold,
                 no_norm: bool = False, gate: bool = False, **kwargs):
        ss = benchmark.search_space
        self.obj_indices  = list(obj_indices)
        self.constr_indices, self.thresholds = _as_constraint_lists(constr_index, threshold)
        # Scalar aliases for the single-constraint campaign path (existing
        # callers read problem.constr_index / .threshold).
        self.constr_index = self.constr_indices[0]
        self.threshold    = self.thresholds[0]
        self.gate         = bool(gate)
        super().__init__(
            n_var=ss.n_var,
            n_obj=len(self.obj_indices),
            n_ieq_constr=len(self.constr_indices),
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

        G = _violation_columns(F, self.constr_indices, self.thresholds)
        F_out = F[:, self.obj_indices]
        if self.gate:
            # Hard gate: infeasible F is unobservable -- inf sentinel,
            # see the class docstring. Fancy indexing above returned a copy,
            # so the in-place mask never touches the shared benchmark matrix.
            F_out[G.max(axis=1) > 0] = np.inf
        out['F'] = F_out
        out['G'] = G

        self.n_eval_calls += len(X_int)


class ConstrainedSurrogateProblemEvox(Problem):
    """Inner surrogate problem: predicted objectives + real (cheap) objectives,
    plus one constraint COLUMN PER constrained metric, each independently
    predicted by a surrogate or computed exactly via the benchmark (so a
    mixed exact/predicted violation vector -- e.g. exact #Params G next to
    predicted latency G -- is a first-class configuration).

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
    constr_index : int or sequence[int]
        Benchmark metric column(s) for the constraint(s) (evaluated-metric
        space). A sequence declares one G column per entry.
    threshold : float or sequence[float]
        Constraint threshold(s) in the evaluated-metric space, parallel to
        ``constr_index``.
    constr_surrogate : object, sequence, or None
        Per-constraint G source. A sequence parallel to ``constr_index``:
        slot ``j`` predicts column ``G[:, j]`` via ``.predict(X)`` when it is
        a model, or None for an exact benchmark computation of that column.
        A bare object (single constraint) and a bare None (ALL columns
        exact) keep the original scalar call signature working.
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
        self.constr_indices, self.thresholds = _as_constraint_lists(constr_index, threshold)
        n_constr = len(self.constr_indices)
        if constr_surrogate is None:
            self.constr_surrogates = [None] * n_constr
        elif isinstance(constr_surrogate, (list, tuple)):
            if len(constr_surrogate) != n_constr:
                raise ValueError(
                    f'constr_surrogate sequence length {len(constr_surrogate)} '
                    f'!= number of constraints {n_constr}')
            self.constr_surrogates = list(constr_surrogate)
        else:
            if n_constr != 1:
                raise ValueError(
                    'a bare constr_surrogate object is only valid for a '
                    'single constraint; pass a per-slot sequence instead')
            self.constr_surrogates = [constr_surrogate]
        # Scalar aliases for the single-constraint campaign path.
        self.constr_index     = self.constr_indices[0]
        self.threshold        = self.thresholds[0]
        self.constr_surrogate = constr_surrogate
        super().__init__(
            n_var=ss.n_var,
            n_obj=len(self.obj_indices),
            n_ieq_constr=n_constr,
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
        # both the real objectives and every exact constraint column -- avoid
        # a double lookup and keep the same metric space as the outer problem.
        need_bench = bool(self.real_obj_indices) or any(
            s is None for s in self.constr_surrogates)
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

        # ── constraints: per-column predicted or exact ────────────────────────
        G_cols = []
        for surrogate, idx, thr in zip(self.constr_surrogates,
                                       self.constr_indices, self.thresholds):
            if surrogate is not None:
                G_cols.append(np.asarray(surrogate.predict(X_float)).reshape(n))
            else:
                G_cols.append(_violation(F_bench[:, idx], thr))
        out['G'] = np.column_stack(G_cols)
