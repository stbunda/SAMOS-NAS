"""experiments2/scenario_run/_problems.py -- pymoo Problem classes and
penalty wrapper for the scenario_run campaign's samos/random methods.

Ported from experiments/constraint2/run_constraint.py (COPIED, not
modified in place) so experiments2/ has no import-time dependency on
experiments/ -- see algorithms.py's module docstring for the builders that
consume these classes.

B0ObjectiveProblem / B0SurrogateProblemEvox carry ONE deliberate fix over
the run_constraint.py originals: a floor constraint (sense=-1, feasible <=>
metric >= tau) must enter the b0-as-obj EXTRA objective NEGATED. Every
EvoXBench objective is minimized, so appending the raw metric under a floor
constraint optimizes directly AWAY from feasibility. ``constr_pos``/``sense``
select which output column (if any) gets negated; sense=1 (the default,
every ceiling scenario) reproduces the original un-negated behaviour
exactly.
"""

import numpy as np
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.core.individual import calc_cv
from pymoo.core.problem import Problem
from pymoo.util.misc import from_dict


class _ConstraintsAsPenaltyMO(ConstraintsAsPenalty):
    """Multi-objective static penalty: F + penalty * CV broadcast across every
    objective column. pymoo 0.6.1.1's own ``ConstraintsAsPenalty.do`` reshapes
    the per-individual aggregated CV to F's shape, which only works for
    single-objective F; this override broadcasts with ``CV.reshape(-1, 1)``
    and is otherwise identical."""

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)
        F, G, H = from_dict(out, 'F', 'G', 'H')
        out['__F__'], out['__G__'], out['__H__'] = F, G, H
        CV = calc_cv(G=G, H=H)
        out['F'] = F + self.penalty * CV.reshape(-1, 1)
        out.pop('G', None)
        out.pop('H', None)
        return out


def _obj_split(obj_indices, cheap_cols):
    """(predict_pos, real_pos): output-column POSITIONS into obj_indices,
    split by whether the underlying benchmark column is cheap. Unused by
    method 'samos' (everything predicted); kept for parity with
    run_constraint.py's helper of the same name and possible future
    samos-cheap support."""
    real_pos    = [pos for pos, col in enumerate(obj_indices) if col in cheap_cols]
    predict_pos = [pos for pos in range(len(obj_indices)) if pos not in real_pos]
    return predict_pos, real_pos


class B0ObjectiveProblem(Problem):
    """Outer problem for the b0-as-obj row: UNCONSTRAINED search over
    ``obj_indices`` benchmark columns (the two scoring objectives + the
    constrained metric, in that order) -- no ``G`` anywhere. Same
    evaluate-once / normalize-once / non-finite-guard convention as
    ``ConstrainedEvoXBenchProblem``, minus the constraint machinery. Hard-mode
    gating is algorithm-side only (SAMOS2's ``gate_g_fn`` / the nsga2 path's
    ``_HardGateNSGA2B0`` read the constrained-metric column of evaluated F);
    gated rows are discarded whole, so their F is never consumed by anything.

    ``constr_pos``/``sense`` : output-column position of the appended
    constrained metric and its direction (see module docstring) -- when
    sense < 0 that column is negated so minimizing it maximizes the metric.
    """

    def __init__(self, benchmark, obj_indices, constr_pos=None, sense=1,
                 no_norm: bool = False):
        ss = benchmark.search_space
        self.obj_indices = list(obj_indices)
        self.constr_pos  = constr_pos
        self.sense        = sense
        super().__init__(
            n_var=ss.n_var,
            n_obj=len(self.obj_indices),
            xl=np.asarray(ss.lb, dtype=float),
            xu=np.asarray(ss.ub, dtype=float),
        )
        self.benchmark    = benchmark
        self.no_norm      = no_norm
        self.n_eval_calls = 0

    def _evaluate(self, X, out, *args, **kwargs):
        X_int = np.round(X).astype(int)
        F = self.benchmark.evaluate(X_int, true_eval=False)
        if not self.no_norm and not self.benchmark.normalized_objectives:
            F = self.benchmark.normalize(F)
        F = np.where(np.isfinite(F), F, 1.0)
        F_out = F[:, self.obj_indices]
        if self.constr_pos is not None and self.sense < 0:
            F_out[:, self.constr_pos] = -F_out[:, self.constr_pos]
        out['F'] = F_out
        self.n_eval_calls += len(X_int)


class B0SurrogateProblemEvox(Problem):
    """Inner problem for the b0-as-obj row: the unconstrained 3-objective
    counterpart of ``ConstrainedSurrogateProblemEvox`` (same predict/real
    split over output-column POSITIONS into ``obj_indices``, same
    evaluate-once / normalize-once / non-finite-guard convention) -- minus
    the constraint. No constraint-surrogate seam: the constrained metric is
    an ordinary predicted-or-real objective column here.

    ``constr_pos``/``sense`` : see ``B0ObjectiveProblem`` -- applied
    identically after both the real and predicted columns are assembled, so
    it does not matter whether that column came from the benchmark or a
    surrogate.
    """

    def __init__(self, surrogates, obj_indices, predict_obj_indices,
                 real_obj_indices, benchmark, constr_pos=None, sense=1,
                 no_norm: bool = False):
        ss = benchmark.search_space
        self.obj_indices         = list(obj_indices)
        self.predict_obj_indices = list(predict_obj_indices)
        self.real_obj_indices    = list(real_obj_indices)
        self.constr_pos          = constr_pos
        self.sense                = sense
        super().__init__(
            n_var=ss.n_var,
            n_obj=len(self.obj_indices),
            xl=np.asarray(ss.lb, dtype=float),
            xu=np.asarray(ss.ub, dtype=float),
        )
        self.surrogates = surrogates
        self.benchmark  = benchmark
        self.no_norm    = no_norm

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        F = np.zeros((n, self.n_obj))

        if self.real_obj_indices:
            X_int   = np.round(X).astype(int)
            F_bench = self.benchmark.evaluate(X_int, true_eval=False)
            if not self.no_norm and not self.benchmark.normalized_objectives:
                F_bench = self.benchmark.normalize(F_bench)
            F_bench = np.where(np.isfinite(F_bench), F_bench, 1.0)
            for pos in self.real_obj_indices:
                F[:, pos] = F_bench[:, self.obj_indices[pos]]

        X_float = X.astype(float)
        for surrogate, pos in zip(self.surrogates, self.predict_obj_indices):
            F[:, pos] = np.asarray(surrogate.predict(X_float)).squeeze()

        if self.constr_pos is not None and self.sense < 0:
            F[:, self.constr_pos] = -F[:, self.constr_pos]

        out['F'] = F
