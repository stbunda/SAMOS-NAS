"""experiments2/scenario_run/algorithms.py -- algorithm builder for the
scenario_run constrained-NAS campaign (S1-S6, scenarios.py).

Four methods, fully native to experiments2/ (no experiments/ import --
see _problems.py for the ported problem classes):
  random : RandomGA -- pure random search, exact evaluation, no selection
           pressure. Runs only the scenario default handler
           (scenarios.DEFAULT_HANDLER[mode]) -- random has no selection
           pressure or sampling strategy for a handler to act on.
  samos  : SAMOS2 with XGBoost surrogates on both objectives and, outside
           b0-as-obj, an XGBoost constraint surrogate. Ported from
           experiments/constraint2/run_constraint.py::build_algorithm's
           'samos' branch, restricted to what scenarios.HANDLERS actually
           uses (no samos-cheap, no h1-cdp-reject/h4-cdp-sms/b0-nsga2). The
           b0-as-obj row always uses an NSGA-II inner GA (never SMS-EMOA --
           this campaign's b0 row was always run_constraint's 'b0-nsga2'
           code path under the old HANDLER_CODE translation, now dropped
           since the ids are native).
  nsga2  : plain pymoo NSGA2 over the real ConstrainedEvoXBenchProblem.
           Each of the 7 handlers acts on SURVIVAL only, never by wrapping
           the real problem: SAMOS2 can wrap its inner surrogate problem
           because the outer archive still records unpenalized F and raw G,
           but plain NSGA-II has no inner/outer split, so wrapping the real
           problem would write penalized/relaxed F straight into the
           population the callback records. See the per-handler classes
           below for how each one stays survival-only.
  ctaea  : pymoo CTAEA (Li et al.) over the same real problem. Like random it
           runs ONE handler slot (scenarios.fixed_handlers -> 'h4-cdp' in both
           modes): its constraint handling is intrinsic -- the CA/DA archive
           pair and the CV-first restricted-mating tournament -- with no seam
           to swap another handler into. It is a constrained-MOEA reference
           point for the nsga2/samos x h4-cdp cells, not a fourth row of the
           handler grid.
  ssansga2 : pysamoo SSA-NSGA-II (_ssansga2.ScenarioSSANSGA2), a
           surrogate-assisted-MOEA reference point for the samos cells. Its
           seam is what the surrogate MODELS, so it runs TWO fixed slots
           (scenarios.fixed_handlers): 'h4-cdp', where the constraint gets its
           own XGBoost surrogate and the inner NSGA-II ranks by constraint
           domination, and 'b1-unconstrained', where the constraint is neither
           modelled nor declared (build_problem returns
           _problems.UnconstrainedGatedProblem) so selection is blind to it.
           Everything else is shared between the two, so the pair isolates the
           value of handing a surrogate-assisted baseline the constraint at
           all.

'b1-unconstrained' is not part of the 7-handler row (scenarios.HANDLERS), but
nsga2 and samos DO build it on request -- an explicit --handler
b1-unconstrained, never --all_handlers -- so the same
constrained-vs-unconstrained contrast ssansga2 supplies exists for the two
full-row methods. Neither needs handler machinery for it: the outer problem
carries the whole difference (n_ieq_constr=0), so nsga2 is the ordinary
_HardGateNSGA2 and samos is SAMOS2 with objective surrogates only and no
constraint surrogate. Both keep their waste counters by reading the hidden
violation column (HIDDEN_G_KEY) instead of 'G'.

build(method, handler, sid, benchmark, seed, pop_size, ...) ->
    (algorithm, copy_algorithm, handler_state)
``copy_algorithm`` is False for h1/h3/h5: their handler_state dict (and, for
h1, the live evaluator/counters) is mutated by the running instance during
the run, and pymoo's default copy_algorithm=True deep-copies the algorithm
(including its survival object's own handler_state dict, and any mutable
state it closes over) before running the COPY -- the ORIGINAL handler_state
dict returned by build() would then never see the trajectory appended to
the copy's private one.

build_problem(sid, benchmark, mode, handler=None) builds the matching outer
pymoo problem: ConstrainedEvoXBenchProblem for every handler except
b0-as-obj (B0ObjectiveProblem, unconstrained) and b1-unconstrained
(UnconstrainedGatedProblem: same evaluation and hard gate, n_ieq_constr
dropped to 0). Bounds always come from
bounds_with_override, so MoSegNAS gets x0>=1
(scenarios.SEARCH_SPACE_LB_OVERRIDE) on both the problem's xl/xu and the
nsga2/samos operators.

b0-as-obj on a floor constraint (S6, scenarios.constr_sense == -1): every
EvoXBench objective is minimized, so appending the raw constrained metric as
an extra objective optimizes AWAY from feasibility when higher is better.
B0ObjectiveProblem/B0SurrogateProblemEvox (_problems.py) negate that column
when sense < 0; the hard-gate counters (_HardGateNSGA2B0._count here, and
this module's SAMOS2 gate_g_fn) read the same (possibly negated) column, so
their violation formula is ``(F - sense*tau) / tau`` -- algebraically
``sense*(metric - tau)/tau`` with ``metric = sense*F`` substituted in,
reducing to the original ``(F - tau)/tau`` when sense=+1.
"""

import os
import sys
from functools import partial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.operators.survival.rank_and_crowding.classes import get_crowding_function
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.display.output import Output
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.ref_dirs import get_reference_directions

import scenarios as SC
from _problems import (
    HIDDEN_G_KEY,
    B0ObjectiveProblem,
    B0SurrogateProblemEvox,
    UnconstrainedGatedProblem,
    _ConstraintsAsPenaltyMO,
)
from _ssansga2 import ScenarioSSANSGA2
from problem.evoxbench.constrained_problem import (
    ConstrainedEvoXBenchProblem,
    ConstrainedSurrogateProblemEvox,
)
from problem.evoxbench.utils import bounds_with_override
from strategy.algorithm.algorithms import HardGateMixin, RandomGA
from strategy.constraints import (
    AdaptivePenaltyProblem,
    DominanceStochasticRanking,
    EpsilonRelaxation,
    RejectionInfillSelector,
    feasible_fraction,
)
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.sampler import EvoxBenchSampler
from strategy.surrogate.models import XGBoost
from strategy.surrogate.samos2 import SAMOS2

METHODS = SC.METHODS
HANDLERS = SC.HANDLERS


def _cv(pop):
    """Per-individual aggregated inequality violation, CV = sum(max(0, G)).
    The population here always comes straight from the real
    ConstrainedEvoXBenchProblem, so G is always present."""
    G = pop.get('G')
    if G is None or np.asarray(G).size == 0:
        return np.zeros(len(pop))
    G = np.asarray(G, dtype=float).reshape(len(pop), -1)
    return np.maximum(0.0, G).sum(axis=1)


# ══════════════════════════════════════════════════════════════════════════
# outer problem
# ══════════════════════════════════════════════════════════════════════════

def _b0_search_obj_indices(sid):
    """(search_obj_indices, constr_pos): scoring objectives plus the
    constrained metric (unless it is already one of them), and the
    constrained metric's position in that list -- shared by build_problem's
    B0ObjectiveProblem and the b0-as-obj algorithm's gate counters."""
    obj_idx = list(SC.obj_indices(sid))
    constr_idx = SC.constr_index(sid)
    if constr_idx in obj_idx:
        return obj_idx, obj_idx.index(constr_idx)
    return obj_idx + [constr_idx], len(obj_idx)


def build_problem(sid, benchmark, mode, handler=None):
    """Outer pymoo problem for scenario ``sid``: ConstrainedEvoXBenchProblem
    for every handler except two baselines --

      'b0-as-obj'       : B0ObjectiveProblem (unconstrained, scoring
          objectives + the constrained metric as an extra objective, no G --
          mirrors run_constraint.run_single's branch);
      'b1-unconstrained': UnconstrainedGatedProblem (identical evaluation and
          hard gate to ConstrainedEvoXBenchProblem, but n_ieq_constr dropped
          to 0 so no survival or surrogate can select on the constraint).

    ``mode`` ('hard'/'soft') sets the evaluability gate; sense comes from
    scenarios.constr_sense (S6 is a floor). Bounds are always overridden via
    bounds_with_override so a plain-NSGA2 run and its outer problem never
    disagree on MoSegNAS's x0>=1 floor."""
    scenario = SC.SCENARIOS[sid]
    lb_override = SC.SEARCH_SPACE_LB_OVERRIDE.get(scenario['space'])
    xl, xu = bounds_with_override(benchmark, lb_override)

    if handler == 'b0-as-obj':
        search_obj, constr_pos = _b0_search_obj_indices(sid)
        problem = B0ObjectiveProblem(benchmark, search_obj, constr_pos=constr_pos,
                                     sense=SC.constr_sense(sid))
    else:
        cls = (UnconstrainedGatedProblem if handler == 'b1-unconstrained'
               else ConstrainedEvoXBenchProblem)
        obj_idx = list(SC.obj_indices(sid))
        problem = cls(
            benchmark, obj_idx, SC.constr_index(sid), scenario['tau'],
            sense=SC.constr_sense(sid), gate=(mode == 'hard'))
    problem.xl = xl.astype(float)
    problem.xu = xu.astype(float)
    return problem


# ══════════════════════════════════════════════════════════════════════════
# generic hard-gate bookkeeping (h4, h6, h2, h3, h5, b0)
# ══════════════════════════════════════════════════════════════════════════

class _HardGateNSGA2(HardGateMixin, NSGA2):
    """Plain NSGA2 plus HardGateMixin's evaluated/feasible counters
    (n_hf_evaluated, n_hf_feasible, _rejected_pop X-log), maintained every
    generation exactly like RandomGA/SAMOS2's bookkeeping (whether the gate
    is on or off, per the mixin's convention). The counters are a pure
    side-effect of ``_gate_keep`` here -- its filtered/dropped return value
    is discarded, so population size and survival dynamics are untouched:
    in hard mode the outer ConstrainedEvoXBenchProblem already death-
    penalises infeasible F to inf (see its docstring), so natural NSGA2
    selection excludes infeasible individuals without any row ever being
    physically removed.
    """

    def __init__(self, *args, hard_gate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_hard_gate(hard_gate)

    def _gate_violation(self, pop):
        """b1-unconstrained's outer problem declares n_ieq_constr=0 and
        publishes its violation under HIDDEN_G_KEY (UnconstrainedGatedProblem),
        so the inherited 'G' lookup would find no constraint and count every
        draw feasible. Read it from there instead -- the counters and the
        hard-mode rejection log stay correct while SELECTION stays blind
        (same override, same reason, as ScenarioSSANSGA2._gate_violation)."""
        if getattr(self.problem, 'hidden_constraints', False):
            hidden = np.asarray(pop.get(HIDDEN_G_KEY), dtype=float)
            return hidden.reshape(len(pop), -1).max(axis=1)
        return super()._gate_violation(pop)

    def _initialize_advance(self, infills=None, **kwargs):
        # Count the DOE too: advance_after_initial_infill routes the initial
        # population through _initialize_advance, never _advance.
        if infills is not None:
            self._gate_keep(infills)
        super()._initialize_advance(infills=infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self._gate_keep(infills)
        super()._advance(infills=infills, **kwargs)


class _FiniteOptOutput(MultiObjectiveOutput):
    """MultiObjectiveOutput that shows its convergence metric the FINITE rows
    of ``opt`` only. Needed by the b1-unconstrained nsga2 cells and nothing
    else.

    Their outer problem declares n_ieq_constr=0, so pymoo's ``filter_optimum``
    has no feasibility split to make: the hard gate's inf-masked individuals
    stay in ``opt`` and are marked feasible. The running ideal/nadir metric
    then computes inf - inf = nan and trips pymoo's own
    ``Termination.update`` assert. Every handler that declares a G is spared
    this -- ``filter_optimum`` drops its infeasible rows first, and a
    generation with no feasible row reaches the metric as an EMPTY F, which
    the base class skips.

    So do exactly that: pass the finite subset through, and skip the metric
    entirely while there is none. The empty subset cannot simply be forwarded
    -- the base class indexes F by opt's 'feas' array (float when opt is
    empty) BEFORE it checks the length. Display only: ``algorithm.opt`` is
    restored before returning, and the population, its survival and the
    callback's archive are never touched.
    """

    def update(self, algorithm):
        opt = algorithm.opt
        F = (np.asarray(opt.get('F'), dtype=float) if opt is not None and len(opt) > 0
             else np.empty((0, 1)))
        finite = np.isfinite(F).all(axis=1)

        if not finite.any():
            Output.update(self, algorithm)          # n_gen / n_eval / n_nds
            for col in (self.igd, self.gd, self.hv, self.eps, self.indicator):
                col.set(None)
            return

        algorithm.opt = opt[finite]
        try:
            super().update(algorithm)
        finally:
            algorithm.opt = opt


class _HardGateCTAEA(HardGateMixin, CTAEA):
    """pymoo CTAEA plus the same evaluated/feasible counters _HardGateNSGA2
    adds to NSGA2, and for exactly the same reason: ``_gate_keep``'s filtered
    return value is discarded, so the CA/DA survival dynamics are untouched
    and only the counters (plus the hard-mode X-log the run pickles) are
    populated. In hard mode the outer ConstrainedEvoXBenchProblem already
    masks infeasible F to inf, which CTAEA's CADASurvival degrades to
    CV-ordering on (its CA update splits on CV, and an all-inf F column is a
    tie in the non-dominated sort of the CV sub-problem) -- the death-penalty
    semantics every other method sees here.
    """

    def __init__(self, *args, hard_gate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_hard_gate(hard_gate)

    def _initialize_infill(self):
        # pymoo's Initialization drops duplicate DOE draws WITHOUT resampling,
        # so a collision -- likely on a small tabular space, NB201 is 5**6 =
        # 15625 architectures against a 20-draw DOE -- yields a population
        # shorter than len(ref_dirs). CTAEA then passes that short length as
        # n_survive while niche ids still span len(ref_dirs), and its
        # _updateDA scans only range(n_survive): the top niche's member is
        # unreachable, so `while len(S) < n_survive` can never exit and the run
        # hangs in initialization, before generation 1, forever.
        #
        # Topping up only ever draws EXTRA samples when a collision actually
        # happened, so every seed whose DOE was already distinct consumes an
        # unchanged RNG stream and stays bit-reproducible against results
        # computed before this fix.
        pop = super()._initialize_infill()
        for _ in range(100):
            if len(pop) >= self.pop_size:
                break
            extra = self.initialization.do(self.problem, self.pop_size - len(pop),
                                            algorithm=self)
            pop = self.eliminate_duplicates.do(Population.merge(pop, extra))
        return pop[:self.pop_size]

    def _initialize_advance(self, infills=None, **kwargs):
        if infills is not None:
            self._gate_keep(infills)
        super()._initialize_advance(infills=infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self._gate_keep(infills)
        super()._advance(infills=infills, **kwargs)


class _HardGateNSGA2B0(HardGateMixin, NSGA2):
    """b0-as-obj hard-gate counters: the outer problem defines no G (the
    constrained metric is an ordinary extra objective column), so the
    gate/violation reads that column of evaluated F directly -- the same
    convention as this module's SAMOS2 ``gate_g_fn`` (see ``_build_delegated``).
    Search stays fully unconstrained (no row is ever dropped from the
    population); only the counters and the X-only rejection log are
    populated. F's constr_pos column is already sense-negated for a floor
    constraint (B0ObjectiveProblem), so the violation formula un-negates it
    algebraically -- see the module docstring."""

    def __init__(self, *args, hard_gate=False, constr_pos, threshold, sense=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_hard_gate(hard_gate)
        self._constr_pos = int(constr_pos)
        self._threshold = float(threshold)
        self._sense = float(sense)

    def _count(self, infills):
        F = infills.get('F')
        viol = (F[:, self._constr_pos] - self._sense * self._threshold) / self._threshold
        feas_mask = viol <= 0.0
        self.n_hf_evaluated += len(infills)
        self.n_hf_feasible += int(np.sum(feas_mask))
        if self.hard_gate:
            rejected = infills[~feas_mask]
            if len(rejected) > 0:
                self._rejected_pop = Population.merge(
                    self._rejected_pop, Population.new('X', rejected.get('X')))

    def _initialize_advance(self, infills=None, **kwargs):
        if infills is not None:
            self._count(infills)
        super()._initialize_advance(infills=infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self._count(infills)
        super()._advance(infills=infills, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# h2 / h3 -- static / adaptive penalty, survival-level
# ══════════════════════════════════════════════════════════════════════════

class _PenaltyRankingSurvival(Survival):
    """h2/h3 shared machinery: ranks and crowds by a TEMPORARY penalized
    view F' = F + w(pop)*CV (w supplied by ``weight_fn(pop)``, recomputed
    fresh every generation), then returns the ORIGINAL individuals -- their
    real F/G are never written to. filter_infeasible=False: a penalty
    handler is not feasibility-first by construction."""

    def __init__(self, weight_fn, crowding_func='cd'):
        super().__init__(filter_infeasible=False)
        self.weight_fn = weight_fn
        self._nds = NonDominatedSorting()
        self._crowding_func = get_crowding_function(crowding_func)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get('F').astype(float, copy=False)
        cv = _cv(pop)
        w = self.weight_fn(pop)
        F_pen = F + w * cv.reshape(-1, 1)

        survivors = []
        fronts = self._nds.do(F_pen, n_stop_if_ranked=n_survive)
        for k, front in enumerate(fronts):
            I = np.arange(len(front))
            if len(survivors) + len(I) > n_survive:
                n_remove = len(survivors) + len(front) - n_survive
                crowd = self._crowding_func.do(F_pen[front, :], n_remove=n_remove)
                I = randomized_argsort(crowd, order='descending', method='numpy')[:-n_remove]
            else:
                crowd = self._crowding_func.do(F_pen[front, :], n_remove=0)
            for j, i in enumerate(front):
                pop[i].set('rank', k)
                pop[i].set('crowding', crowd[j])
            survivors.extend(front[I])
        return pop[survivors]


def _h3_weight_fn(pen, handler_state):
    """h3: adapt the penalty weight from the CURRENT population's own
    feasible fraction (the population being ranked this generation IS the
    live one -- unlike SAMOS2's rebuilt-every-generation inner problem, no
    algo_ref/live-handle indirection is needed), and record it."""

    def weight_fn(pop):
        pen.adapt(feasible_fraction(pop))
        handler_state['penalty_trajectory'].append(float(pen.weight))
        return pen.weight

    return weight_fn


# ══════════════════════════════════════════════════════════════════════════
# h5 -- epsilon relaxation, survival-level
# ══════════════════════════════════════════════════════════════════════════

class _EpsilonRelaxedSurvival(Survival):
    """h5: feasibility-first split using the RELAXED violation G' = G -
    eps(t) (temporary, for ranking only); survivors keep their raw F/G.
    eps(t) is this object's own per-generation clock -- this survival is the
    persistent live object across the whole run, so (unlike SAMOS2's inner
    problem, rebuilt every generation) it can hold EpsilonRelaxation
    directly with no algo_ref indirection."""

    def __init__(self, eps_relaxation, handler_state):
        super().__init__(filter_infeasible=False)
        self.eps = eps_relaxation
        self.handler_state = handler_state
        self._rc = RankAndCrowding()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        cv = _cv(pop)
        self.eps.maybe_init_eps0(pop)   # eps0 = mean DOE CV, first call only
        eps_t = self.eps.eps
        self.handler_state['eps_trajectory'].append(float(eps_t))

        cv_relaxed = cv - eps_t
        feas_mask = cv_relaxed <= 0.0
        feas, infeas = pop[feas_mask], pop[~feas_mask]
        infeas = infeas[np.argsort(cv_relaxed[~feas_mask])]   # least-violating first

        survivors = (self._rc._do(problem, feas, n_survive=min(len(feas), n_survive))
                     if len(feas) > 0 else Population.empty())
        n_remaining = n_survive - len(survivors)
        if n_remaining > 0:
            survivors = Population.merge(survivors, infeas[:n_remaining])

        self.eps.advance()
        return survivors


# ══════════════════════════════════════════════════════════════════════════
# h1 -- rejection sampling, infill-level
# ══════════════════════════════════════════════════════════════════════════

def _plain_dominance_survival():
    """RankAndCrowding with feasibility-first splitting disabled: h1 isolates
    PURE rejection, so the survival step must not ALSO filter by
    feasibility (mirrors run_constraint's h1-rejection recipe)."""
    surv = RankAndCrowding()
    surv.filter_infeasible = False
    return surv


class H1RejectionNSGA2(HardGateMixin, NSGA2):
    """h1-rejection (nsga2 shape): resample offspring via mating until
    feasible, capped at ``resample_cap`` batch draws per generation (default
    10, i.e. 10*pop_size individual attempts total) -- then proceed with
    whatever feasible offspring were found (no forced padding: population
    size self-corrects via the merge with ``self.pop`` in ``_advance``,
    exactly like RandomGA's "no replacement" hard-gate philosophy).

    BUDGET: every probe (feasible or not) is evaluated through
    ``self.evaluator.eval`` -- never a bypassing benchmark call -- so every
    attempt is charged to the run's n_evals budget; this is the whole point
    of testing h1 (measuring rejection's waste). Because probes are marked
    evaluated on the same Individual objects returned from ``_infill``,
    pymoo's automatic post-infill ``evaluator.eval`` call in ``next()`` is a
    no-op (nothing left unevaluated) -- no double counting.

    Rejection is applied identically to the initial DOE and to every later
    generation's offspring (both go through ``_draw_feasible`` below), so the
    GA population never carries a stray infeasible survivor forward just
    because it predates the rejection loop.

    hard mode : infeasible draws are gated -- F unobservable, X-only logged
                to ``_rejected_pop`` (HardGateMixin), never entering the GA
                population.
    soft mode : infeasible draws are evaluated with real F/G and kept in
                ``self._archive`` (this generation's feasible survivors plus
                every excluded draw) for scoring, but excluded from the GA
                population -- this is what keeps hard and soft distinct for
                h1: both exclude infeasible individuals from selection
                pressure, only soft mode preserves their F/G for scoring.
    """

    def __init__(self, *args, hard_gate=False, resample_cap=10,
                 handler_state=None, **kwargs):
        kwargs.setdefault('survival', _plain_dominance_survival())
        super().__init__(*args, **kwargs)
        self._init_hard_gate(hard_gate)
        self.resample_cap = int(resample_cap)
        self.handler_state = handler_state if handler_state is not None else {}
        self.handler_state['h1_resample_cap'] = self.resample_cap
        self.handler_state.setdefault('h1_rejected_count', 0)
        self._archive = Population.empty()
        self._gen_soft_archive = Population.empty()

    def _draw_feasible(self, draw_batch, n_needed):
        """Shared resampling loop for both the DOE and every generation's
        offspring: draw a batch via ``draw_batch(n)``, evaluate it through
        ``self.evaluator`` (charging the run's budget), keep the feasible
        rows, and retry up to ``resample_cap`` times; then proceed with
        whatever feasible set was found (no padding -- population size
        self-corrects via the merge in ``_advance``/``_initialize_advance``).
        Rejected draws are logged (hard: X-only; soft: into
        ``self._gen_soft_archive`` with real F/G) and counted."""
        collected = Population.empty()
        for _ in range(self.resample_cap):
            batch = draw_batch(n_needed)
            if len(batch) == 0:
                break
            self.evaluator.eval(self.problem, batch, algorithm=self)

            cv = _cv(batch)
            feas_mask = cv <= 0.0
            feas, infeas = batch[feas_mask], batch[~feas_mask]

            self.n_hf_evaluated += len(batch)
            self.n_hf_feasible += int(feas_mask.sum())
            self.handler_state['h1_rejected_count'] += int(len(infeas))

            if self.hard_gate:
                if len(infeas) > 0:
                    self._rejected_pop = Population.merge(
                        self._rejected_pop, Population.new('X', infeas.get('X')))
            else:
                self._gen_soft_archive = Population.merge(self._gen_soft_archive, batch)

            collected = Population.merge(collected, feas)
            if len(collected) >= n_needed:
                break

        return collected[:n_needed] if len(collected) > n_needed else collected

    def _initialize_infill(self):
        self._gen_soft_archive = Population.empty()
        return self._draw_feasible(
            lambda n: self.initialization.do(self.problem, n, algorithm=self),
            self.pop_size)

    def _infill(self):
        self._gen_soft_archive = Population.empty()
        return self._draw_feasible(
            lambda n: self.mating.do(self.problem, self.pop, n, algorithm=self),
            self.n_offsprings)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)
        self._set_archive()

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills=infills, **kwargs)
        self._set_archive()

    def _set_archive(self):
        self._archive = (Population.merge(self.pop, self._gen_soft_archive)
                          if not self.hard_gate else Population.empty())


# ══════════════════════════════════════════════════════════════════════════
# build()
# ══════════════════════════════════════════════════════════════════════════

def _build_delegated(method, handler, sid, benchmark, seed, pop_size, mode,
                      n_gen, n_doe, n_infill, n_gen_inner, inner_pop_size, penalty):
    """'random' and 'samos': native port of
    experiments/constraint2/run_constraint.py::build_algorithm's 'random' and
    'samos' branches (ids kept exactly as scenarios.HANDLERS defines them --
    no HANDLER_CODE translation). Only the ids this campaign runs are
    supported: no samos-cheap, h1-cdp-reject, h4-cdp-sms or b0-nsga2 (the
    b0-as-obj row here always uses the NSGA-II inner GA the old delegated
    path reached via HANDLER_CODE's 'b0-as-obj' -> 'b0-nsga2' translation --
    never SMS-EMOA). Bounds (xl/xu, sampler, mutation) all come from
    bounds_with_override with scenarios.SEARCH_SPACE_LB_OVERRIDE, so every
    method searches the same space as build_problem's outer problem."""
    scenario     = SC.SCENARIOS[sid]
    lb_override  = SC.SEARCH_SPACE_LB_OVERRIDE.get(scenario['space'])
    xl, xu       = bounds_with_override(benchmark, lb_override)
    gated        = (mode == 'hard')
    sense        = SC.constr_sense(sid)
    constr_indices = [SC.constr_index(sid)]
    thresholds     = [scenario['tau']]

    sampler   = EvoxBenchSampler(xl, xu)
    crossover = IntegerUniformCrossover(prob=0.9)
    mutation  = IntegerPointMutation(xl, xu)
    elim      = IntegerVectorDuplicateElimination()

    n_doe_    = n_doe if n_doe is not None else pop_size
    n_infill_ = n_infill if n_infill is not None else pop_size
    inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

    handler_state = {}

    if method == 'random':
        # Pure random search: no surrogate, no selection pressure, no
        # replacement -- runs only the scenario/mode default.
        if handler not in SC.fixed_handlers('random', mode):
            raise ValueError(
                f"random has no selection pressure for a handler to act on -- it "
                f"only runs the scenario default ({SC.fixed_handlers('random', mode)} "
                f"for mode={mode!r}), got handler={handler!r}")
        return RandomGA(pop_size=pop_size, sampling=sampler, eliminate_duplicates=elim,
                        hard_gate=gated), True, handler_state

    if method != 'samos':
        raise ValueError(f'Unknown method for delegated build: {method!r}')

    # b0-as-obj searches the scoring objectives + the constrained metric as
    # an ordinary extra objective (B0ObjectiveProblem); every other handler
    # searches just the two scoring objectives and defines G.
    if handler == 'b0-as-obj':
        obj_idx, constr_pos = _b0_search_obj_indices(sid)
    else:
        obj_idx, constr_pos = list(SC.obj_indices(sid)), None

    rng = np.random.RandomState(seed)
    # method 'samos' always predicts every objective (no cheap/real split --
    # that is samos-cheap, not ported, see module docstring).
    predict_pos = list(range(len(obj_idx)))
    real_pos    = []
    surrogates  = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_pos]
    # No constraint surrogate for the two baselines: b0 makes the constrained
    # metric an ordinary predicted objective column, b1 hides it entirely --
    # neither has a G for a surrogate to fill.
    constr_surrogate = (None if handler in ('b0-as-obj', 'b1-unconstrained')
                        else [XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                              for _ in constr_indices])
    # Single constraint keeps the scalar surrogate object of the original
    # campaign path (bit-identical wiring); sequences are multi-only (unused
    # here -- scenarios.py is single-constraint throughout).
    if isinstance(constr_surrogate, list) and len(constr_surrogate) == 1:
        constr_surrogate = constr_surrogate[0]

    if handler == 'b0-as-obj':
        def factory(surrs):
            return B0SurrogateProblemEvox(surrs, obj_idx, predict_pos, real_pos, benchmark,
                                          constr_pos=constr_pos, sense=sense)

        # No G defined by the outer problem -- gate/waste counters read the
        # constrained metric straight off evaluated F, at its search-obj
        # position. F's constr_pos column is already sense-negated for a
        # floor constraint (B0SurrogateProblemEvox), so the violation
        # formula un-negates it algebraically -- see the module docstring.
        _constr_pos_arr = np.array([obj_idx.index(ci) for ci in constr_indices])
        _thr            = np.asarray(thresholds, dtype=float)

        def b0_gate_g_fn(pop, _t=_thr, _p=_constr_pos_arr, _s=sense):
            return (pop.get('F')[:, _p] - _s * _t) / _t

        algorithm = SAMOS2(
            sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
            predict_obj_indices=predict_pos,
            crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
            n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
            inner_algorithm=NSGA2,
            hard_gate=gated, gate_g_fn=b0_gate_g_fn,
        )
        return algorithm, True, handler_state

    if handler == 'b1-unconstrained':
        # Two scoring objectives and nothing else: the inner problem is the
        # same unconstrained surrogate problem b0 uses (B0SurrogateProblemEvox
        # is generic over which benchmark columns it is handed -- b0 differs
        # only by appending the constrained metric), and no constraint
        # surrogate is fitted, so neither the infill selection nor the inner
        # survival can see feasibility.
        def factory(surrs):
            return B0SurrogateProblemEvox(surrs, obj_idx, predict_pos, real_pos,
                                          benchmark)

        # The outer UnconstrainedGatedProblem declares no G, so the gate reads
        # the violation off HIDDEN_G_KEY -- measurement stays exact while
        # selection stays blind (see _problems.UnconstrainedGatedProblem).
        def b1_gate_g_fn(pop):
            return np.asarray(pop.get(HIDDEN_G_KEY), dtype=float)

        algorithm = SAMOS2(
            sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
            predict_obj_indices=predict_pos,
            crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
            n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
            inner_algorithm=NSGA2,
            hard_gate=gated, gate_g_fn=b1_gate_g_fn,
        )
        return algorithm, True, handler_state

    # ── handler wiring (scenario-independent) ─────────────────────────────
    # wrap_inner: callable(inner) applied inside the factory each outer
    # generation, so only the inner surrogate problem is ever wrapped -- the
    # outer archive / saved pkls always keep unpenalized F and raw,
    # un-relaxed G. algo_ref is a late-bound handle to the live SAMOS2
    # instance for the stateful handlers (h3/h5).
    samos2_kwargs  = {}
    wrap_inner     = None
    algo_ref       = []
    copy_algorithm = True

    if handler == 'h4-cdp':
        pass   # native CDP: out['G'] + RankAndCrowding do everything.

    elif handler == 'h2-static_penalty':
        def wrap_inner(inner):
            # Static penalty on the inner problem only; _ConstraintsAsPenaltyMO,
            # not the plain pymoo class -- see its docstring.
            return _ConstraintsAsPenaltyMO(inner, penalty=penalty)

    elif handler == 'h3-adaptive':
        # w0=1.0 (not scenario['penalty']): matches the old delegated path's
        # run_constraint.build_algorithm exactly -- unlike the nsga2 method's
        # h3 (_build_nsga2 below), which deliberately starts at w0=penalty.
        pen = AdaptivePenaltyProblem(w0=1.0, target=0.5, c=1.2)
        handler_state['penalty_trajectory'] = []   # w(t), one entry per outer gen

        def wrap_inner(inner):
            algo = algo_ref[0]            # live handle (copy_algorithm=False)
            if gated:
                # A gated archive is all-feasible by construction -- the
                # adaptive signal is the cumulative evaluated-feasible ratio.
                if algo.n_hf_evaluated > 0:
                    pen.adapt(algo.n_hf_feasible / algo.n_hf_evaluated)
            else:
                arc = algo._archive
                if len(arc) > 0:
                    pen.adapt(feasible_fraction(arc))
            handler_state['penalty_trajectory'].append(float(pen.weight))
            return pen.wrap(inner)

        copy_algorithm = False            # live-archive handle, see docstring

    elif handler == 'h5-epsilon':
        eps = EpsilonRelaxation(n_gen_total=n_gen)
        handler_state['eps_trajectory'] = []       # eps(t), one entry per outer gen

        def wrap_inner(inner):
            algo = algo_ref[0]
            if gated:
                # eps0 = mean CV over ALL DOE evaluations: gated archive
                # members have CV 0, the rejected draws carry their
                # violation in the rejection log.
                if eps.eps0 is None and algo.n_hf_evaluated > 0:
                    total_cv = (float(np.sum(np.maximum(0.0, algo._rejected_G)))
                                if algo._rejected_G is not None else 0.0)
                    eps.set_eps0(total_cv / algo.n_hf_evaluated)
            else:
                eps.maybe_init_eps0(algo._archive)   # eps0 = mean DOE CV
            handler_state['eps_trajectory'].append(float(eps.eps))
            wrapped = eps.wrap(inner)                # snapshots epsilon(t)
            eps.advance()                            # t+1 for next generation
            return wrapped

        copy_algorithm = False            # live-archive handle, see docstring

    elif handler == 'h6-DSR':
        samos2_kwargs['inner_algorithm'] = partial(
            NSGA2, survival=DominanceStochasticRanking(Pf=0.45, seed=seed))

    elif handler == 'h1-rejection':
        # Rejection on the infill seam (inner candidates already carry the
        # inner problem's G -- predicted, or exact for a cheap constraint),
        # with pure-rejection semantics: the inner survival must NOT be
        # feasibility-first. pymoo 0.6.1.1's RankAndCrowding constructor
        # hardcodes filter_infeasible=True (no kwarg), but it is a plain
        # instance attribute, so it is flipped post-construction; the single
        # instance is safely reused across outer generations (stateless).
        samos2_kwargs['infill_selector'] = RejectionInfillSelector()
        _surv = RankAndCrowding()
        _surv.filter_infeasible = False
        samos2_kwargs['inner_algorithm'] = partial(NSGA2, survival=_surv)

    else:
        raise ValueError(f'Unknown handler for samos: {handler!r}')

    def factory(surrs, fitted_constr_surrogate=None):
        inner = ConstrainedSurrogateProblemEvox(
            surrs, obj_idx, predict_pos, real_pos, benchmark,
            constr_indices, thresholds, sense=sense,
            constr_surrogate=fitted_constr_surrogate,
        )
        return wrap_inner(inner) if wrap_inner is not None else inner

    algorithm = SAMOS2(
        sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
        predict_obj_indices=predict_pos,
        crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
        n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
        constr_surrogate=constr_surrogate,
        hard_gate=gated,
        **samos2_kwargs,
    )
    algo_ref.append(algorithm)   # late-bind the live handle for h3/h5 closures
    return algorithm, copy_algorithm, handler_state


def _build_nsga2(handler, sid, benchmark, seed, pop_size, mode, n_gen,
                  penalty, resample_cap):
    scenario = SC.SCENARIOS[sid]
    lb_override = SC.SEARCH_SPACE_LB_OVERRIDE.get(scenario['space'])
    xl, xu = bounds_with_override(benchmark, lb_override)
    gated = (mode == 'hard')
    handler_state = {}

    # output=MultiObjectiveOutput(): pymoo's own NSGA2 declares
    # output=MultiObjectiveOutput() as a constructor DEFAULT ARGUMENT, so
    # without an explicit instance here every NSGA2 built in this process
    # (across every scenario/handler in one campaign run) shares ONE Output
    # object; pymoo's Callback base class only ever calls .initialize() once
    # per object (is_initialized latch), so its per-run indicator state
    # never resets for the SECOND-and-later builds -- silently comparing
    # ideal/nadir points across runs with different n_obj (e.g. a 2-obj
    # handler followed by 3-obj b0-as-obj) and crashing with a shape
    # mismatch. A fresh instance per build() call sidesteps the shared state
    # entirely.
    common = dict(pop_size=pop_size, sampling=EvoxBenchSampler(xl, xu),
                   crossover=IntegerUniformCrossover(prob=0.9),
                   mutation=IntegerPointMutation(xl, xu),
                   eliminate_duplicates=IntegerVectorDuplicateElimination(),
                   seed=seed, output=MultiObjectiveOutput())

    if handler == 'h4-cdp':
        return _HardGateNSGA2(hard_gate=gated, **common), True, handler_state

    if handler == 'h6-DSR':
        dsr = DominanceStochasticRanking(Pf=0.45, seed=seed)
        return (_HardGateNSGA2(hard_gate=gated, survival=dsr, **common),
                True, handler_state)

    if handler == 'h2-static_penalty':
        surv = _PenaltyRankingSurvival(weight_fn=lambda pop: penalty)
        return (_HardGateNSGA2(hard_gate=gated, survival=surv, **common),
                True, handler_state)

    if handler == 'h3-adaptive':
        # w0=penalty (not the class default 1.0): h3 self-corrects
        # multiplicatively every generation regardless of scale, but should
        # not START a run ~30x mis-scaled relative to the objective span
        # (see TAU.md, 'Penalty').
        pen = AdaptivePenaltyProblem(w0=penalty, target=0.5, c=1.2)
        handler_state['penalty_trajectory'] = []
        surv = _PenaltyRankingSurvival(weight_fn=_h3_weight_fn(pen, handler_state))
        return (_HardGateNSGA2(hard_gate=gated, survival=surv, **common),
                False, handler_state)

    if handler == 'h5-epsilon':
        eps = EpsilonRelaxation(n_gen_total=n_gen)
        handler_state['eps_trajectory'] = []
        surv = _EpsilonRelaxedSurvival(eps, handler_state)
        return (_HardGateNSGA2(hard_gate=gated, survival=surv, **common),
                False, handler_state)

    if handler == 'h1-rejection':
        algo = H1RejectionNSGA2(hard_gate=gated, resample_cap=resample_cap,
                                 handler_state=handler_state, **common)
        return algo, False, handler_state

    if handler == 'b0-as-obj':
        _, constr_pos = _b0_search_obj_indices(sid)
        algo = _HardGateNSGA2B0(
            hard_gate=gated, constr_pos=constr_pos, threshold=scenario['tau'],
            sense=SC.constr_sense(sid), **common)
        return algo, True, handler_state

    if handler == 'b1-unconstrained':
        # Plain NSGA-II over the two scoring objectives: build_problem hands
        # it an UnconstrainedGatedProblem (n_ieq_constr=0), so RankAndCrowding
        # has no feasibility split to make and there is no survival wiring to
        # add -- the SAME class as h4-cdp, differing only in the outer problem
        # it runs against and in the output (_FiniteOptOutput, which keeps the
        # hard gate's inf-masked rows out of pymoo's convergence metric -- see
        # its docstring). Counters read the hidden violation column
        # (_HardGateNSGA2._gate_violation), so waste is still measured.
        return (_HardGateNSGA2(hard_gate=gated, **{**common, 'output': _FiniteOptOutput()}),
                True, handler_state)

    raise ValueError(f'Unknown handler for nsga2: {handler!r}')


def _build_ctaea(handler, sid, benchmark, seed, pop_size, mode):
    """pymoo CTAEA over the real ConstrainedEvoXBenchProblem, with this
    campaign's integer operators (same sampler/crossover/mutation/dedup as
    _build_nsga2, so ctaea searches the same space with the same variation as
    every other method) in place of CTAEA's real-valued SBX/PM defaults.

    Constraint handling is intrinsic, so only the scenarios.fixed_handler slot
    is accepted -- passing any other handler is a wiring bug, not a
    configuration.

    Reference directions come from the Riesz s-energy generator rather than
    Das-Dennis: CTAEA derives ``pop_size = len(ref_dirs)``, and only the
    energy generator returns EXACTLY the requested count for any n_obj (the
    Das-Dennis simplex lattice can only hit the counts its partition number
    happens to produce -- 21, not 20, for 3 objectives). Seeding it off the
    run seed keeps the directions reproducible per run.

    output=MultiObjectiveOutput(): a fresh instance per build for the same
    shared-default-argument reason documented in _build_nsga2.
    """
    expected = SC.fixed_handlers('ctaea', mode)
    if handler not in expected:
        raise ValueError(
            f"ctaea's constraint handling is intrinsic (CA/DA archives + CV-first "
            f'restricted mating) -- there is no seam for a handler to act on, so it '
            f'only runs the {expected} slot, got handler={handler!r}')

    scenario = SC.SCENARIOS[sid]
    lb_override = SC.SEARCH_SPACE_LB_OVERRIDE.get(scenario['space'])
    xl, xu = bounds_with_override(benchmark, lb_override)
    ref_dirs = get_reference_directions('energy', len(SC.obj_indices(sid)),
                                        n_points=pop_size, seed=seed)

    algorithm = _HardGateCTAEA(
        ref_dirs=ref_dirs, hard_gate=(mode == 'hard'),
        sampling=EvoxBenchSampler(xl, xu),
        crossover=IntegerUniformCrossover(prob=0.9),
        mutation=IntegerPointMutation(xl, xu),
        eliminate_duplicates=IntegerVectorDuplicateElimination(),
        seed=seed, output=MultiObjectiveOutput(),
    )
    return algorithm, True, {}


def _build_ssansga2(handler, sid, benchmark, seed, pop_size, mode, n_doe,
                     n_infill, n_gen_inner, inner_pop_size):
    """pysamoo SSA-NSGA-II (_ssansga2.ScenarioSSANSGA2) over the same real
    problem, with this campaign's integer operators and XGBoost surrogates.

    Budget knobs are taken from the same defaults as method 'samos'
    (n_doe = n_infill = pop_size, inner GA of pop_size*10 for n_gen_inner
    generations) so the two surrogate-assisted methods differ in ALGORITHM,
    not in how much inner search each one buys per real evaluation.

    Two fixed slots (scenarios.fixed_handlers), differing only in whether the
    constraint is modelled at all:
      h4-cdp           : one extra XGBoost surrogate on the G column; the
                         inner NSGA-II's RankAndCrowding then splits on
                         predicted feasibility (Deb's CDP) with no extra
                         wiring, since the outer problem declares
                         n_ieq_constr=1.
      b1-unconstrained : no constraint surrogate, and build_problem hands it an
                         UnconstrainedGatedProblem (n_ieq_constr=0), so nothing
                         in the loop can select on feasibility.

    Surrogate seeds come off one RandomState(seed) draw sequence in the same
    order method 'samos' uses (objectives first, constraint last), so a given
    seed wires comparable models into both methods.
    """
    expected = SC.fixed_handlers('ssansga2', mode)
    if handler not in expected:
        raise ValueError(
            f"ssansga2's seam is what its surrogate models, not how survival ranks "
            f'-- it only runs the {expected} slots, got handler={handler!r}')

    scenario    = SC.SCENARIOS[sid]
    lb_override = SC.SEARCH_SPACE_LB_OVERRIDE.get(scenario['space'])
    xl, xu      = bounds_with_override(benchmark, lb_override)

    n_doe_    = n_doe if n_doe is not None else pop_size
    n_infill_ = n_infill if n_infill is not None else pop_size
    inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

    rng        = np.random.RandomState(seed)
    obj_models = [XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                  for _ in SC.obj_indices(sid)]
    constr_model = (XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                    if handler == 'h4-cdp' else None)

    algorithm = ScenarioSSANSGA2(
        obj_models=obj_models, constr_model=constr_model,
        hard_gate=(mode == 'hard'),
        sampling=EvoxBenchSampler(xl, xu),
        crossover=IntegerUniformCrossover(prob=0.9),
        mutation=IntegerPointMutation(xl, xu),
        eliminate_duplicates=IntegerVectorDuplicateElimination(),
        inner_seed=seed,
        n_initial_doe=n_doe_, n_infills=n_infill_,
        surr_pop_size=inner_ps, surr_n_gen=n_gen_inner,
    )
    return algorithm, True, {}


def build(method, handler, sid, benchmark, seed, pop_size, *,
          mode='hard', n_gen=30, n_doe=None, n_infill=None, n_gen_inner=20,
          inner_pop_size=None, penalty=1.0, resample_cap=10):
    """Build the (algorithm, copy_algorithm, handler_state) triple for one
    (method, handler, scenario) cell of the scenario_run campaign.

    Parameters
    ----------
    method : 'random' | 'nsga2' | 'ctaea' | 'ssansga2' | 'samos'
    handler : one of scenarios.ALL_HANDLERS (the 7-handler row plus the
        fixed-slot-only 'b1-unconstrained')
    sid : scenario id, e.g. 'S1'..'S6' (scenarios.SCENARIOS)
    benchmark : the evoxbench benchmark instance for sid's (suite, pid)
    mode : 'hard' (evaluability gate) | 'soft' (archive infeasible with real
        F/G) -- orthogonal to sid; scenarios.py shares tau across both.
    n_gen : total planned OUTER generations (drives h5's epsilon decay
        schedule and is forwarded to the delegated random/samos path).
    n_doe, n_infill, n_gen_inner, inner_pop_size, penalty : forwarded to
        run_constraint.build_algorithm for method in ('random', 'samos');
        penalty also used directly by nsga2's h2-static_penalty.
    resample_cap : nsga2's h1-rejection batch-draw cap per generation
        (default 10, i.e. 10*pop_size individual attempts).
    """
    if handler not in SC.ALL_HANDLERS:
        raise ValueError(f'Unknown handler: {handler!r}')

    if method in ('random', 'samos'):
        return _build_delegated(method, handler, sid, benchmark, seed, pop_size,
                                 mode, n_gen, n_doe, n_infill, n_gen_inner,
                                 inner_pop_size, penalty)
    if method == 'nsga2':
        return _build_nsga2(handler, sid, benchmark, seed, pop_size, mode,
                             n_gen, penalty, resample_cap)
    if method == 'ctaea':
        return _build_ctaea(handler, sid, benchmark, seed, pop_size, mode)
    if method == 'ssansga2':
        return _build_ssansga2(handler, sid, benchmark, seed, pop_size, mode,
                                n_doe, n_infill, n_gen_inner, inner_pop_size)
    raise ValueError(f'Unknown method: {method!r}')
