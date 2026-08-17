"""experiments2/scenario_run/_ioccobra.py -- IOC-SAMO-COBRA for the
scenario_run campaign.

Wires the vendored ``strategy.algorithm.ioc_samo_cobra`` (de Winter et al.'s
self-adaptive RBF-surrogate constrained multi-objective optimizer) into the
S1-S6 grid as the method ``ioc-cobra``. Two classes and one driver function:

``_CobraProblem``
    The duck-typed problem COBRA's two entry points expect (``lower``,
    ``upper``, ``nObj``, ``nConstraints``, ``ref``, ``evaluate``), implemented
    as a thin wrapper around the campaign's OWN outer pymoo problem
    (algorithms.build_problem -> ConstrainedEvoXBenchProblem). Every method in
    this campaign therefore shares one evaluation path: one benchmark call per
    point, the baseline normalize-once convention, the non-finite -> 1.0 guard,
    ``G = sense * (metric - tau) / tau``, and the hard gate's inf-masking.

``IOCCobraRun``
    The recording shim. COBRA drives its own optimization loop, so it never
    goes through pymoo's ``minimize`` / ``Evaluator`` / ``Callback`` and
    nothing would otherwise record a trajectory. This object holds the
    campaign callback and the HardGateMixin-style counters, and exposes exactly
    the attributes ``run_scenario.run_single`` reads off a finished pymoo
    algorithm, so the pkl it produces is structurally identical to every other
    method's.

Four things need saying about how COBRA sits in a NAS campaign built around
pymoo GAs:

1. IT SEARCHES A CONTINUOUS BOX. COBRA optimizes its infill criterion with
   COBYLA over ``[-1, 1]^d`` and rescales to the problem's bounds; the NAS
   genotype is a lattice point. ``_CobraProblem.evaluate`` therefore rounds
   before evaluating, exactly as every pymoo problem in this repo does. The
   RBF surrogates are fitted on the CONTINUOUS points (``cobra['A']``), so
   two proposals that round to the same architecture are distinct centres and
   never make the interpolation matrix singular -- they are simply two real
   evaluations of the same architecture, i.e. wasted budget, which is counted
   like any other evaluation and visible in ``n_eval_realised``.

2. THE HARD GATE HAS NO ARCHIVE TO DROP FROM. SAMOS2 and SSA-NSGA-II implement
   the gate by dropping infeasible points from the archive their surrogates
   train on. COBRA cannot: ``A`` / ``Fres`` / ``Gres`` are one aligned archive
   that also drives termination (``n = len(cobra['A'])``) and the
   distance-to-evaluated term in ``gCOBRA``, and the whole method rests on
   modelling the objectives over the WHOLE box, feasible or not. So under the
   hard gate the gated rows stay, and their unobservable objective vector is
   imputed with the scenario's HV reference point -- the death penalty every
   other method applies via ``inf``, in the finite form COBRA's standardization
   and RBF fits need. No real objective value of a gated architecture ever
   enters the run: ``ref`` carries no information beyond "infeasible", it is
   dominated by every scoring point, and it contributes zero hypervolume. The
   violation stays REAL either way (the gate needs it, and G is observable by
   construction -- see ConstrainedEvoXBenchProblem's docstring), so the
   constraint surrogate is trained on true violations under both modes.

3. ONE HANDLER SLOT, 'h4-cdp'. Like ctaea, COBRA's constraint handling is
   intrinsic and has no seam for the 7-handler row: one RBF per constraint, a
   self-adapting epsilon margin on the predicted violation, and a
   feasibility-first Pareto filter (``paretofrontFeasible``) that is Deb-style
   constraint domination by another name. scenarios.fixed_handlers pins it to
   'h4-cdp' in both modes, which also lands it next to the nsga2 / samos /
   ctaea / ssansga2 h4-cdp cells in the analysis grid.

4. BATCH SIZE IS A BUDGET-PARITY KNOB, and defaults to ``pop_size``. Upstream's
   ``batch=1`` buys one real evaluation per iteration, and an iteration is
   expensive in a way a GA generation is not: it refits 6 RBF kernels x
   {2 objectives, 1 constraint} x {plain, plog} surrogates AND runs a 10-fold
   kernel-selection sweep over the same grid, i.e. ~400 SVD-solved RBF fits on
   the full archive. At ``batch = pop_size`` the run spends its budget in the
   same 20-evaluations-per-generation rhythm as every other method and
   completes the same ~60 generations, which is both the fair comparison and
   the only tractable one.
"""

import numpy as np
from pymoo.core.population import Population

from strategy.algorithm.ioc_samo_cobra import (cheap_SAMO_COBRA_Init,
                                               cheap_SAMO_COBRA_PhaseII)

# Upstream default: how many COBYLA restarts the infill search uses per
# iteration (cheap_SAMO_COBRA_Init's own default, restated here because this
# campaign passes it explicitly rather than inheriting it).
DEFAULT_COMPUTE_STARTING_POINTS = 16

# LHS, not upstream's HALTON default: the Halton sequence is deterministic and
# ignores cobraSeed entirely, so all 20 seeds of a cell would share one initial
# design. lhs() draws from the numpy global RNG that Init seeds with cobraSeed,
# so the DOE varies per seed the way every other method's does.
DEFAULT_INIT_DESIGN = 'LHS'


class _BudgetExhausted(Exception):
    """Raised by the problem adapter once ``n_evals`` real evaluations have
    been spent, to unwind COBRA's own loop from inside its evaluation call.

    PhaseII terminates on ``len(cobra['A']) >= feval`` checked BETWEEN
    iterations, so an iteration that starts one point short of the budget still
    evaluates a whole batch and overshoots it. Every other method in this
    campaign gets exactly ``n_evals`` (run_scenario.py terminates on
    ('n_evals', N)); unwinding here keeps that identical rather than handing
    ioc-cobra up to batch-1 free evaluations. With the default
    ``batch = pop_size`` and an n_evals that is a multiple of it, the budget
    lands exactly and this never fires.
    """


class _Evaluator:
    """Stand-in for pymoo's Evaluator: run_scenario.run_single reads
    ``algorithm.evaluator.n_eval`` when it assembles the meta block."""

    def __init__(self):
        self.n_eval = 0


class IOCCobraRun:
    """Recording shim standing in for the finished pymoo algorithm.

    Maintains what nothing else would (COBRA calls no callback and keeps no
    pymoo counters):

    * the campaign callback's per-generation series -- one ``notify`` per
      ``pop_size`` real evaluations, so ioc-cobra's trajectory is sampled on the
      same grid as every GA's per-generation one;
    * ``n_hf_evaluated`` / ``n_hf_feasible``, the cumulative counters
      FeasibilityAwareEvoxBenchCallback prefers over archive counts (under the
      gate the archive holds survivors only, so counting it reports zero waste);
    * ``_rejected_X`` / ``_rejected_G``, hard mode's rejection log.

    Under the hard gate the callback is shown only the FEASIBLE rows of each
    batch, matching SAMOS2 / SSA-NSGA-II: a gated architecture's objectives are
    unobservable, so it must not enter the scored archive. The waste it
    represents is carried by the counters and the rejection log instead. In soft
    mode every row is passed through with its real F and G.
    """

    def __init__(self, problem, callback, pop_size, n_evals, hard_gate):
        self.problem   = problem
        self.callback  = callback
        self.pop_size  = int(pop_size)
        self.n_evals   = int(n_evals)
        self.hard_gate = bool(hard_gate)

        self.evaluator      = _Evaluator()
        self.n_gen          = 0
        self.pop            = Population.empty()
        self.n_hf_evaluated = 0
        self.n_hf_feasible  = 0

        self._pending     = []    # rows awaiting the next notify
        self._rejected_X  = []
        self._rejected_G  = []

    def record(self, x_int, F, G):
        """Book one real evaluation and flush a generation whenever
        ``pop_size`` of them have accumulated."""
        self.evaluator.n_eval += 1
        self.n_hf_evaluated   += 1
        feasible = bool(np.max(G) <= 0.0)
        self.n_hf_feasible += int(feasible)

        if self.hard_gate and not feasible:
            self._rejected_X.append(x_int)
            self._rejected_G.append(float(np.max(G)))

        self._pending.append((x_int, F, G, feasible))
        while len(self._pending) >= self.pop_size:
            self._notify(self.pop_size)

    def _notify(self, n):
        batch, self._pending = self._pending[:n], self._pending[n:]
        rows = [r for r in batch if r[3] or not self.hard_gate]
        if rows:
            self.pop = Population.new(
                X=np.array([r[0] for r in rows], dtype=float),
                F=np.array([r[1] for r in rows], dtype=float),
                G=np.array([r[2] for r in rows], dtype=float))
        else:
            self.pop = Population.empty()
        self.n_gen += 1
        self.callback.notify(self)

    def finalize(self):
        """Flush the trailing partial batch and freeze the rejection log into
        the array shapes run_scenario.run_single expects."""
        if self._pending:
            self._notify(len(self._pending))
        self._rejected_X = (np.asarray(self._rejected_X, dtype=float)
                            if self._rejected_X
                            else np.empty((0, self.problem.n_var)))
        self._rejected_G = np.asarray(self._rejected_G, dtype=float)


class _CobraProblem:
    """The duck-typed problem cheap_SAMO_COBRA_Init / _PhaseII expect (see the
    package __init__ for the contract), backed by the campaign's outer pymoo
    problem so the evaluation path is shared with every other method.

    ``cheap_evaluate`` is deliberately NOT defined: nothing in this campaign is
    a cheap metric, so Init leaves ``cheapObj`` / ``cheapCon`` all-False and
    COBRA models both objectives AND the constraint -- the same all-predicted
    wiring method 'samos' uses.
    """

    def __init__(self, problem, ref_point, run):
        self.problem      = problem
        self.run          = run
        self.lower        = np.asarray(problem.xl, dtype=float)
        self.upper        = np.asarray(problem.xu, dtype=float)
        self.ref          = np.asarray(ref_point, dtype=float)
        self.nObj         = int(problem.n_obj)
        self.nConstraints = int(problem.n_ieq_constr)

    def evaluate(self, x):
        """One real evaluation: round the continuous proposal onto the genotype
        lattice, evaluate through the outer problem, impute any gated (inf)
        objective row with the reference point, and record it.

        Returns upstream's ``[F, G]`` pair of 1-D rows."""
        if self.run.evaluator.n_eval >= self.run.n_evals:
            raise _BudgetExhausted

        x_int = np.clip(np.round(np.asarray(x, dtype=float)), self.lower, self.upper)
        F, G = self.problem.evaluate(x_int.reshape(1, -1), return_values_of=['F', 'G'])
        F, G = np.asarray(F[0], dtype=float), np.atleast_1d(np.asarray(G[0], dtype=float))

        # Hard gate only: the outer problem masks an infeasible row's F to inf
        # (unobservable objectives). The non-finite -> 1.0 guard runs BEFORE
        # that mask, so a non-finite F here can only be the gate -- see 2 in the
        # module docstring for why the imputation is the reference point.
        F = np.where(np.isfinite(F), F, self.ref)

        self.run.record(x_int, F, G)
        return [F, G]


def run_ioc_cobra(problem, callback, *, seed, n_evals, pop_size, ref_point,
                  hard_gate, batch=None, init_des_points=None,
                  compute_starting_points=DEFAULT_COMPUTE_STARTING_POINTS,
                  init_design=DEFAULT_INIT_DESIGN, infill_criteria='PHV',
                  seq_feval=None):
    """Run one ioc-cobra cell and return the finished ``IOCCobraRun``.

    Parameters
    ----------
    problem : the outer pymoo problem from algorithms.build_problem (always a
        ConstrainedEvoXBenchProblem here -- ioc-cobra runs only 'h4-cdp').
    callback : the campaign's FeasibilityAwareEvoxBenchCallback, notified once
        per ``pop_size`` real evaluations.
    n_evals : total real-evaluation budget, enforced exactly (see
        _BudgetExhausted).
    pop_size : generation granularity for the callback, and the default batch
        and initial-design size -- so ioc-cobra's DOE, per-generation spend and
        realised generation count all match method 'samos'.
    ref_point : the scenario's HV reference point; COBRA's own PHV infill
        criterion is computed against it, and ``nObj`` is derived from its
        length.
    batch, init_des_points, compute_starting_points, init_design,
    infill_criteria : upstream knobs (see the module docstring for why batch
        defaults to pop_size rather than upstream's 1).
    seq_feval : COBYLA iteration cap for the infill search. None keeps
        upstream's ``dimension * batch^2 * 50``; it is a compute knob, not an
        algorithm parameter (COBYLA normally converges on its own trust-region
        tolerance well before the cap).
    """
    batch    = int(batch) if batch else int(pop_size)
    init_des = int(init_des_points) if init_des_points else int(pop_size)

    run  = IOCCobraRun(problem, callback, pop_size, n_evals, hard_gate)
    prob = _CobraProblem(problem, ref_point, run)

    try:
        cobra = cheap_SAMO_COBRA_Init(
            prob, nCores=1, feval=n_evals, batch=batch, initDesPoints=init_des,
            computeStartingPoints=compute_starting_points,
            initDesign=init_design, infillCriteria=infill_criteria,
            cobraSeed=seed, saveResults=False)
        if seq_feval is not None:
            cobra['seqFeval'] = int(seq_feval)
        cheap_SAMO_COBRA_PhaseII(cobra)
    except _BudgetExhausted:
        pass

    run.finalize()
    return run
