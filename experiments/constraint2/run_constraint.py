"""experiments/constraint2/run_constraint.py --- constraint-scenario campaign runner.

Four single-constraint scenarios instantiate the full hard/soft x
cheap/expensive taxonomy on a SHARED objective pair and (near-)shared
instance set, so every scenario pair isolates exactly one axis:

  s1 : #Params <= T, hard   (cheap constraint)
  s2 : latency <= T, hard   (expensive constraint)
  s3 : #Params <= T, soft   (cheap constraint)
  s4 : latency <= T, soft   (expensive constraint)

Eight multi-constraint scenarios stack every constraint-TYPE combination
(C = cheap/structural, E = expensive/predictor-backed) plus an
all-constraints scenario, each in both modes -- the stress test of the
single-constraint campaign's constraint-as-objective (b0) headline result,
whose Pareto-dominance search degrades with every added objective while
CDP-style handlers scale to multiple G rows trivially:

  s5  : #Params + FLOPs <= T, hard             (CC)
  s6  : #Params + FLOPs <= T, soft             (CC)
  s7  : #Params + latency <= T, hard           (CE)
  s8  : #Params + latency <= T, soft           (CE)
  s9  : latency + energy <= T, hard            (EE)
  s10 : latency + energy <= T, soft            (EE)
  s11 : #Params + FLOPs + latency + energy, hard  (ALL)
  s12 : #Params + FLOPs + latency + energy, soft  (ALL)

Objectives are {Err., FLOPs} everywhere (OBJ_METRICS). Constrained metrics
are resolved per instance: '#Params'/'FLOPs' are structural everywhere; the
latency/energy columns are device-specific per benchmark (LATENCY_METRIC /
ENERGY_METRIC). Feasible <=> EVERY constrained metric is at or below its own
~Q25 threshold; the JOINT design feasible fraction of a combination is a
measured property recorded per (scenario, instance)
(JOINT_DESIGN_FEASIBLE_FRACTION), never a design target.

Note on CC/ALL: FLOPs is both a scoring objective and a constrained metric
(a compute budget on top of FLOPs minimization). b0 rows never duplicate the
column: their search objectives are the scoring objectives plus the
constrained metrics NOT already among them (CC -> 3-obj {Err, FLOPs,
#Params}; the FLOPs budget is enforced by the gate/scoring only), recorded
in meta['search_obj_metrics'].

Mode axis (hard vs soft) is PHYSICAL, not a handler default:
  hard : evaluability gate. Every infeasible high-fidelity evaluation is
         counted as waste (n_hf_evaluated / n_hf_feasible on the algorithm,
         preferred by the callback) and DISCARDED -- its F is masked to
         np.inf by the outer problem and it never enters the archive, so
         objective surrogates train on feasible points only. The constraint
         surrogate trains on archive + rejection log (X and violation of
         every discarded point). The rejection log is also persisted to the
         pkl (``rejected_X`` / ``rejected_G``; RandomGA records rejected X
         only, so its ``rejected_G`` is NaN) so discarded evaluations are
         reconstructable post hoc. SAMOS2's DOE redraws counted batches
         until >= 2 feasible points exist. Models a solution that cannot
         run (e.g. too large for the device): its metrics are unobservable.
  soft : infeasible solutions are evaluated and archived with raw F and G;
         handlers decide how violation trades off against fitness. Models a
         solution that runs with degraded quality.

Instances
---------
Selected via --suite/--pid; the valid set depends on the scenario's
constraint type (SCENARIO_INSTANCES):

  cheap (s1/s3):     c10mop/4 (NATS, 5 vars), c10mop/5 (NB201, 6 vars),
                     in1kmop/9 (MobileNetV3, 21 vars), citysegmop/5
                     (MoSegNAS, 24 vars).
  expensive (s2/s4): the same four PLUS citysegmop/10 -- the same MoSegNAS
                     search space under a second deployment device (H2
                     latency instead of H1). citysegmop/10 is excluded from
                     the cheap scenarios: without the latency column it is
                     the identical experiment to citysegmop/5 (same
                     architectures, same Err/FLOPs/#Params values).
  cc/ce (s5-s8):     the cheap set (structural metrics exist everywhere;
                     citysegmop/10 stays excluded for the same duplication
                     reason -- for CC entirely, and CE keeps the frozen
                     cheap-set for seed-paired comparison against s1-s4).
  ee/all (s9-s12):   instances exposing >= 2 expensive metric columns
                     (latency AND energy): c10mop/5 (EdgeGPU), citysegmop/5
                     (H1) and citysegmop/10 (H2). c10mop/4 and in1kmop/9
                     expose latency only and cannot run EE/ALL.

All five instances share cheap FLOPs/#Params columns (BENCHMARK_META
cheap_obj_indices), so the samos-cheap split is uniform: Err. predicted,
FLOPs exact; constraint exact when cheap (s1/s3), predicted when expensive
(s2/s4). On citysegmop, Err. and the latency/energy columns are themselves
predictor-backed (RankNet / lookup table); FLOPs and #Params are structural.

Thresholds
----------
Per instance and metric, ~Q25 of a fixed-seed (0) 10k uniform random sample
in the space the problem operates in (benchmark.evaluate output,
normalize() applied only when not benchmark.normalized_objectives) -- so
roughly 25% of the space is feasible PER METRIC. Multi-constraint scenarios
keep the per-metric Q25 thresholds and record the measured JOINT fraction.
See experiments/constraint2/THRESHOLDS.md for exact values, provenance, and
the MoSegNAS exceptions: ~33.5% of random MoSegNAS genotypes collapse onto
one fallback architecture whose normalized metrics all equal 1.0, so no 25%
threshold exists for metrics whose lower quartile hits that spike; those
thresholds sit at the largest achieved value below it, the fallback arch is
infeasible on every axis, and the exact design feasible fraction(s) are
recorded per run in meta['design_feasible_fraction'] (plus
meta['design_feasible_fraction_joint'] for multi-constraint runs).

Methods
-------
  random       : RandomGA -- pure random search, exact evaluation, no
                 selection pressure, and no replacement: in hard mode an
                 infeasible draw just consumes budget. Runs only the
                 scenario-default handler (every other handler acts on
                 selection pressure it does not have; rejection sampling
                 would be a strategy, and for expensive constraints an
                 unbudgeted oracle).
  samos        : SAMOS2, both objectives predicted (XGBoost, per-seed
                 subseeded), constraint ALWAYS predicted (XGBoost).
  samos-cheap  : SAMOS2, Err. predicted / FLOPs exact; constraint exact for
                 s1/s3 (#Params is cheap), predicted for s2/s4.

Handlers
--------
Every scenario runs the SAME row (HANDLER_ROW) on samos/samos-cheap, so the
2x2 comparisons hold handler-for-handler; the scenario's mode picks only the
DEFAULT (hard -> h4-cdp, soft -> h2-penalty):

  h1-rejection       : rejection on the infill seam (inner candidates carry
                       the inner problem's G -- predicted or exact), inner
                       survival deliberately NOT feasibility-first so the
                       row isolates pure rejection.
  h2-penalty         : static penalty F + w*CV on the inner problem only
                       (_ConstraintsAsPenaltyMO), w = --penalty.
  h3-adaptive-penalty: multiplicative weight toward target feasible
                       fraction 0.5 (factor 1.2); in hard mode the signal
                       is the cumulative evaluated-feasible ratio (a gated
                       archive is all-feasible). w(t) saved per generation.
  h4-cdp             : native constrained-dominance (feasibility-first
                       RankAndCrowding via out['G']), zero extra wiring.
  h5-eps             : outer-clocked epsilon relaxation, eps0 = mean DOE CV
                       (in hard mode over ALL DOE evaluations including the
                       rejected ones), linear to 0 at 50% of --n_gen.
                       eps(t) saved per generation.
  h6-sr              : dominance-lifted stochastic ranking (Pf=0.45) as the
                       inner survival.
  h1-cdp-reject      : rejection on the infill seam PLUS the default
                       feasibility-first survival -- the practitioner
                       configuration (never spend an evaluation on a
                       candidate whose G says it cannot run).
  b0-as-obj          : constraint-as-objective baseline: unconstrained
                       3-objective search (objectives + constrained metric),
                       SMS-EMOA inner, no constraint surrogate; scored post
                       hoc on the feasible 2-objective projection like every
                       other row. In hard mode gating reads the constrained-
                       metric column of evaluated F.
  h4-cdp-sms         : control -- h4-cdp with SMS-EMOA inner (completes the
                       {formulation} x {inner GA} factorial with b0-nsga2).
  b0-nsga2           : control -- b0 formulation with NSGA-II inner.

Output layout
-------------
  {results_root}/{scenario}/{suite}/pid{pid}/{budget}/{method}/{handler}/seed_{N}.pkl

One fixed objective set, fresh tree -- no objective tag level. Every pkl is
self-describing via data['meta'] (instance, scenario, metrics, threshold,
mode, gate, handler, method, seed, budget, design feasible fraction);
data['handler_state'] carries h3's w(t) / h5's eps(t) where applicable.

Examples
--------
  python experiments/constraint2/run_constraint.py --scenario s1
  python experiments/constraint2/run_constraint.py --scenario s2 --suite citysegmop --pid 10
  python experiments/constraint2/run_constraint.py --scenario s3 --all_handlers
  python experiments/constraint2/run_constraint.py --scenario s4 --method samos --handler b0-nsga2
  python experiments/constraint2/run_constraint.py --scenario s1 --method samos-cheap \\
      --pop_size 8 --n_gen 3 --n_gen_inner 4 --results_root /tmp/smoke
"""

import argparse
import os
import pickle
import random
import sys
from functools import partial

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, _REPO_ROOT)

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.core.individual import calc_cv
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.misc import from_dict

from problem.evoxbench.benchmark_meta import BENCHMARK_META, metric_index
from problem.evoxbench.callbacks import FeasibilityAwareEvoxBenchCallback
from problem.evoxbench.constrained_problem import (
    ConstrainedEvoXBenchProblem,
    ConstrainedSurrogateProblemEvox,
)
from problem.evoxbench.utils import get_benchmark
from strategy.algorithm.algorithms import RandomGA
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

SUITE = 'c10mop'
PID   = 4

_DEFAULT_RESULTS_ROOT = os.path.join('results', 'constraint2')

# ─── campaign design (config, not code) ──────────────────────────────────────

OBJ_METRICS = ('Err.', 'FLOPs')      # fixed for the whole campaign

CHEAP_CONSTR_METRIC = '#Params'

# The expensive (latency/energy) columns are device-specific per instance.
LATENCY_METRIC = {
    ('c10mop', 4):      'Latency',
    ('c10mop', 5):      'EdgeGPU Lat.',
    ('in1kmop', 9):     'Latency',
    ('citysegmop', 5):  'H1 Lat.',
    ('citysegmop', 10): 'H2 Lat.',
}

# Second expensive metric (EE/ALL combos); only instances exposing both a
# latency AND an energy column qualify -- c10mop/4 and in1kmop/9 have
# latency only and therefore cannot run EE/ALL.
ENERGY_METRIC = {
    ('c10mop', 5):      'EdgeGPU En.',
    ('citysegmop', 5):  'H1 En.',
    ('citysegmop', 10): 'H2 En.',
}

SCENARIOS = {
    's1':  dict(constr='cheap',     mode='hard'),
    's2':  dict(constr='expensive', mode='hard'),
    's3':  dict(constr='cheap',     mode='soft'),
    's4':  dict(constr='expensive', mode='soft'),
    's5':  dict(constr='cc',        mode='hard'),
    's6':  dict(constr='cc',        mode='soft'),
    's7':  dict(constr='ce',        mode='hard'),
    's8':  dict(constr='ce',        mode='soft'),
    's9':  dict(constr='ee',        mode='hard'),
    's10': dict(constr='ee',        mode='soft'),
    's11': dict(constr='all',       mode='hard'),
    's12': dict(constr='all',       mode='soft'),
}

# Constraint types with more than one constrained metric (s5-s12).
MULTI_COMBOS = ('cc', 'ce', 'ee', 'all')

# citysegmop/10 is expensive-only: without its latency column it is the
# identical experiment to citysegmop/5 (same search space, same
# Err/FLOPs/#Params values) -- see module docstring, 'Instances'.
_CHEAP_SET = [('c10mop', 4), ('c10mop', 5), ('in1kmop', 9), ('citysegmop', 5)]
_EE_SET    = [('c10mop', 5), ('citysegmop', 5), ('citysegmop', 10)]

SCENARIO_INSTANCES = {
    'cheap':     _CHEAP_SET,
    'expensive': _CHEAP_SET + [('citysegmop', 10)],
    'cc':        _CHEAP_SET,
    'ce':        _CHEAP_SET,
    'ee':        _EE_SET,
    'all':       _EE_SET,
}

# ~Q25 thresholds (see THRESHOLDS.md for provenance and the MoSegNAS spike
# exceptions), in the space the constrained problem operates in. Every
# metric keeps ONE threshold regardless of which scenario constrains it;
# multi-constraint scenarios combine these per-metric values unchanged.
THRESHOLDS = {
    ('c10mop', 4):      {'#Params': 0.3078135998873715,  'Latency':      0.36311633657280196,
                         'FLOPs':   0.1814415868239766},
    ('c10mop', 5):      {'#Params': 0.11198208286674133, 'EdgeGPU Lat.': 0.5728975060318674,
                         'FLOPs':   0.10810810810810814, 'EdgeGPU En.':  0.544715316639718},
    ('in1kmop', 9):     {'#Params': 0.46727868434691233, 'Latency':      0.34752772487724964,
                         'FLOPs':   0.37192471140932504},
    ('citysegmop', 5):  {'#Params': 0.9997754995135822,  'H1 Lat.':      0.5579798323679069,
                         'FLOPs':   0.9988584474885844,  'H1 En.':       0.4869375522655389},
    ('citysegmop', 10): {'H2 Lat.': 0.9999087009935012,  '#Params':      0.9997754995135822,
                         'FLOPs':   0.9988584474885844,  'H2 En.':       0.9999647471796245},
}

# Fraction of the fixed-seed 10k random sample feasible at each threshold --
# 0.25 by quantile construction except at the MoSegNAS spike (recorded in
# every pkl's meta as design_feasible_fraction).
DESIGN_FEASIBLE_FRACTION = {
    ('c10mop', 4):      {'#Params': 0.25,   'Latency':      0.25,   'FLOPs': 0.25},
    ('c10mop', 5):      {'#Params': 0.254,  'EdgeGPU Lat.': 0.250,  'FLOPs': 0.2544,
                         'EdgeGPU En.': 0.25},
    ('in1kmop', 9):     {'#Params': 0.25,   'Latency':      0.25,   'FLOPs': 0.25},
    ('citysegmop', 5):  {'#Params': 0.1762, 'H1 Lat.':      0.250,  'FLOPs': 0.1646,
                         'H1 En.': 0.25},
    ('citysegmop', 10): {'H2 Lat.': 0.2015, '#Params':      0.1762, 'FLOPs': 0.1646,
                         'H2 En.': 0.2292},
}

# Measured JOINT fraction of the same fixed-seed 10k sample satisfying EVERY
# per-metric threshold of a combination -- a measured property per (combo,
# instance), never a design target (see THRESHOLDS.md, multi-constraint
# section; the MoSegNAS fallback arch is infeasible on every axis, so cs
# joint fractions sit at or below the tightest single fraction). Note
# cc @ c10mop/5: the #Params and FLOPs Q25-feasible sets coincide exactly
# on NB201 (joint == both marginals) -- CC there is effectively a single
# binding constraint.
JOINT_DESIGN_FEASIBLE_FRACTION = {
    ('cc',  ('c10mop', 4)):      0.1644,
    ('cc',  ('c10mop', 5)):      0.2544,
    ('cc',  ('in1kmop', 9)):     0.0825,
    ('cc',  ('citysegmop', 5)):  0.1424,
    ('ce',  ('c10mop', 4)):      0.1034,
    ('ce',  ('c10mop', 5)):      0.0990,
    ('ce',  ('in1kmop', 9)):     0.0837,
    ('ce',  ('citysegmop', 5)):  0.1167,
    ('ee',  ('c10mop', 5)):      0.2429,
    ('ee',  ('citysegmop', 5)):  0.2216,
    ('ee',  ('citysegmop', 10)): 0.1994,
    ('all', ('c10mop', 5)):      0.0977,
    ('all', ('citysegmop', 5)):  0.0966,
    ('all', ('citysegmop', 10)): 0.1121,
}


def constr_metrics_for(scenario, suite, pid):
    """The scenario's constrained metric NAME(s) at one instance, as a tuple
    (length 1 for s1-s4)."""
    constr = SCENARIOS[scenario]['constr']
    key = (suite, pid)
    if constr == 'cheap':
        return (CHEAP_CONSTR_METRIC,)
    if constr == 'expensive':
        return (LATENCY_METRIC[key],)
    if constr == 'cc':
        return ('#Params', 'FLOPs')
    if constr == 'ce':
        return ('#Params', LATENCY_METRIC[key])
    if constr == 'ee':
        return (LATENCY_METRIC[key], ENERGY_METRIC[key])
    if constr == 'all':
        return ('#Params', 'FLOPs', LATENCY_METRIC[key], ENERGY_METRIC[key])
    raise ValueError(f'Unknown constraint type: {constr!r}')


METHODS = ['random', 'samos', 'samos-cheap']

# One row for every scenario, so the 2x2 comparisons hold handler-for-handler.
HANDLER_ROW = ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp',
               'h5-eps', 'h6-sr', 'h1-cdp-reject', 'b0-as-obj']

# Inner-GA controls: run explicitly (--handler / the sbatch control block),
# not part of --all_handlers.
CONTROL_HANDLERS = ['h4-cdp-sms', 'b0-nsga2']

# Multi-constraint scenarios (s5-s12) run a reduced row: the two robust top
# handlers plus BOTH b0 inner-GA variants -- at 4-6 search objectives the
# inner GA may start to matter, so b0-nsga2 is a first-class row here, not a
# control. h1/h3/h5/h6 variants are not needed for the constraint-count
# hypothesis (explicit --handler still accepts them).
MULTI_HANDLER_ROW = ['h2-penalty', 'h4-cdp', 'b0-as-obj', 'b0-nsga2']

HANDLERS = HANDLER_ROW + CONTROL_HANDLERS

# Handlers that search the 3-column objective set (objectives + constrained
# metric) instead of defining G.
B0_HANDLERS = ('b0-as-obj', 'b0-nsga2')

DEFAULT_HANDLER = {'hard': 'h4-cdp', 'soft': 'h2-penalty'}


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
    split by whether the underlying benchmark column is cheap."""
    real_pos    = [pos for pos, col in enumerate(obj_indices) if col in cheap_cols]
    predict_pos = [pos for pos in range(len(obj_indices)) if pos not in real_pos]
    return predict_pos, real_pos


class B0ObjectiveProblem(Problem):
    """Outer problem for the b0 rows: UNCONSTRAINED search over
    ``obj_indices`` benchmark columns (the two objectives + the constrained
    metric, in that order) -- no ``G`` anywhere. Same evaluate-once /
    normalize-once / non-finite-guard convention as
    ``ConstrainedEvoXBenchProblem``, minus the constraint machinery. Hard-mode
    gating is algorithm-side only (SAMOS2.gate_g_fn reads the constrained-
    metric column of evaluated F); gated rows are discarded whole, so their F
    is never consumed by anything."""

    def __init__(self, benchmark, obj_indices, no_norm: bool = False):
        ss = benchmark.search_space
        self.obj_indices = list(obj_indices)
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
        out['F'] = F[:, self.obj_indices]
        self.n_eval_calls += len(X_int)


class B0SurrogateProblemEvox(Problem):
    """Inner problem for the b0 rows: the unconstrained 3-objective
    counterpart of ``ConstrainedSurrogateProblemEvox`` (same predict/real
    split over output-column POSITIONS into ``obj_indices``, same
    evaluate-once / normalize-once / non-finite-guard convention) -- minus
    the constraint. No constraint-surrogate seam: the constrained metric is
    an ordinary predicted-or-real objective column here."""

    def __init__(self, surrogates, obj_indices, predict_obj_indices,
                 real_obj_indices, benchmark, no_norm: bool = False):
        ss = benchmark.search_space
        self.obj_indices         = list(obj_indices)
        self.predict_obj_indices = list(predict_obj_indices)
        self.real_obj_indices    = list(real_obj_indices)
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

        out['F'] = F


def build_algorithm(method, benchmark, suite, pid, obj_indices, constr_indices,
                     thresholds, handler, seed, pop_size, n_doe, n_infill,
                     n_gen_inner, inner_pop_size, penalty, n_gen, gated):
    """Returns ``(algorithm, copy_algorithm, handler_state)``.
    ``copy_algorithm`` is False only for h3/h5 (their per-generation state
    objects read the live algorithm through the factory closure's
    ``algo_ref``; pymoo's default deepcopy would sever that handle).
    ``handler_state`` is a (possibly empty) dict of per-outer-generation
    handler-state trajectories (h3's penalty weight, h5's epsilon), appended
    live by the wrap_inner closures so any penalized/relaxed view is
    reconstructable from the stored pkl alone."""
    xl = np.asarray(benchmark.search_space.lb, dtype=int)
    xu = np.asarray(benchmark.search_space.ub, dtype=int)

    sampler   = EvoxBenchSampler(xl, xu)
    crossover = IntegerUniformCrossover(prob=0.9)
    mutation  = IntegerPointMutation(xl, xu)
    elim      = IntegerVectorDuplicateElimination()

    n_doe_    = n_doe if n_doe is not None else pop_size
    n_infill_ = n_infill if n_infill is not None else pop_size
    inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

    handler_state = {}

    if method == 'random':
        # Pure random search: no surrogate, no selection pressure, and no
        # replacement -- in hard mode an infeasible draw just consumes
        # budget (main() pairs random only with the scenario default).
        return RandomGA(pop_size=pop_size, sampling=sampler, eliminate_duplicates=elim,
                        hard_gate=gated), True, handler_state

    rng = np.random.RandomState(seed)

    if method == 'samos':
        predict_pos = list(range(len(obj_indices)))
        real_pos    = []
        surrogates       = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_pos]
        # b0 rows: no constraint surrogate -- the constrained metrics are
        # ordinary predicted objective columns here, never a G.
        constr_surrogate = (None if handler in B0_HANDLERS
                             else [XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                                   for _ in constr_indices])

    elif method == 'samos-cheap':
        cheap_cols = set(BENCHMARK_META[suite][pid].get('cheap_obj_indices', []))
        predict_pos, real_pos = _obj_split(obj_indices, cheap_cols)
        surrogates = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_pos]
        if handler in B0_HANDLERS:
            constr_surrogate = None   # same reason as the 'samos' branch above.
        else:
            # Per-slot split: cheap constraint columns are exact (None slot),
            # expensive ones get a surrogate -- a mixed exact/predicted
            # violation vector under CE.
            constr_surrogate = [None if ci in cheap_cols
                                else XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                                for ci in constr_indices]
            if not any(s is not None for s in constr_surrogate):
                constr_surrogate = None   # all exact -> no fit/factory seam needed

    else:
        raise ValueError(f'Unknown method: {method!r}')

    # Single constraint keeps the scalar surrogate object of the original
    # campaign path (bit-identical wiring); sequences are multi-only.
    if isinstance(constr_surrogate, list) and len(constr_surrogate) == 1:
        constr_surrogate = constr_surrogate[0]

    if handler in B0_HANDLERS:
        # Constraint-as-objective baseline: unconstrained 3-objective search,
        # no wrap_inner, no constraint surrogate. 'b0-as-obj' uses SMS-EMOA
        # as the inner algorithm (empirically the strongest at 3 objectives
        # on these instances, see experiments/constraint_deprecated/
        # prelim_3obj.py); 'b0-nsga2' is the inner-GA control.
        def factory(surrs):
            return B0SurrogateProblemEvox(surrs, obj_indices, predict_pos, real_pos, benchmark)

        # The b0 outer problem defines no G, so the gate/waste counters read
        # the constrained metrics straight off evaluated F -- every
        # constrained metric's benchmark column is among the search
        # objectives (run_single appends the ones not already there), so
        # each is located by position: an infeasible evaluation fails the
        # same way for every handler. Passed unconditionally: soft rows use
        # it for exact counters only.
        _constr_pos = np.array([obj_indices.index(ci) for ci in constr_indices])
        _thr        = np.asarray(thresholds, dtype=float)

        def b0_gate_g_fn(pop, _t=_thr, _p=_constr_pos):
            return (pop.get('F')[:, _p] - _t) / _t

        algorithm = SAMOS2(
            sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
            predict_obj_indices=predict_pos,
            crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
            n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
            inner_algorithm=SMSEMOA if handler == 'b0-as-obj' else NSGA2,
            hard_gate=gated, gate_g_fn=b0_gate_g_fn,
        )
        return algorithm, True, handler_state

    # ── handler wiring (scenario-independent) ─────────────────────────────────
    # wrap_inner: callable(inner) applied inside the factory each outer
    # generation, so only the inner surrogate problem is ever wrapped -- the
    # outer archive / saved pkls always keep unpenalized F and raw, un-relaxed
    # G. algo_ref is a late-bound handle to the live SAMOS2 instance for the
    # stateful handlers (h3/h5).
    samos2_kwargs  = {}
    wrap_inner     = None
    algo_ref       = []
    copy_algorithm = True

    if handler == 'h4-cdp':
        pass   # native CDP: out['G'] + RankAndCrowding do everything.

    elif handler == 'h4-cdp-sms':
        # Inner-GA control: identical constraint wiring to h4-cdp (pymoo's
        # SMSEMOA survival is also feasibility-first via the Survival base
        # class), only the inner algorithm changes.
        samos2_kwargs['inner_algorithm'] = SMSEMOA

    elif handler == 'h2-penalty':
        def wrap_inner(inner):
            # Static penalty on the inner problem only; _ConstraintsAsPenaltyMO,
            # not the plain pymoo class -- see its docstring.
            return _ConstraintsAsPenaltyMO(inner, penalty=penalty)

    elif handler == 'h3-adaptive-penalty':
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

    elif handler == 'h5-eps':
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

    elif handler == 'h6-sr':
        samos2_kwargs['inner_algorithm'] = partial(
            NSGA2, survival=DominanceStochasticRanking(Pf=0.45, seed=seed))

    elif handler == 'h1-rejection':
        # Rejection on the infill seam (inner candidates already carry the
        # inner problem's G -- predicted, or exact for cheap constraints),
        # with pure-rejection semantics: the inner survival must NOT be
        # feasibility-first. pymoo 0.6.1.1's RankAndCrowding constructor
        # hardcodes filter_infeasible=True (no kwarg), but it is a plain
        # instance attribute, so it is flipped post-construction; the single
        # instance is safely reused across outer generations (stateless).
        samos2_kwargs['infill_selector'] = RejectionInfillSelector()
        _surv = RankAndCrowding()
        _surv.filter_infeasible = False
        samos2_kwargs['inner_algorithm'] = partial(NSGA2, survival=_surv)

    elif handler == 'h1-cdp-reject':
        # h1's infill-seam rejection COMBINED with the default
        # feasibility-first RankAndCrowding survival -- the rejection+CDP
        # hybrid h1-rejection deliberately isolates away. The practitioner
        # row: never spend a real evaluation on a candidate whose (predicted,
        # or exact for cheap constraints) G says it cannot run, and stay
        # feasibility-first everywhere else.
        samos2_kwargs['infill_selector'] = RejectionInfillSelector()

    else:
        raise ValueError(f'Unknown handler: {handler!r}')

    def factory(surrs, fitted_constr_surrogate=None):
        inner = ConstrainedSurrogateProblemEvox(
            surrs, obj_indices, predict_pos, real_pos, benchmark,
            constr_indices, thresholds, constr_surrogate=fitted_constr_surrogate,
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


def run_single(method, scenario, suite, pid, handler, seed, pop_size, n_gen,
               n_doe, n_infill, n_gen_inner, inner_pop_size, penalty,
               compute_indicators=True):
    np.random.seed(seed)
    random.seed(seed)

    cfg            = SCENARIOS[scenario]
    benchmark      = get_benchmark(suite, pid)
    obj_indices    = [metric_index(suite, pid, m) for m in OBJ_METRICS]
    constr_metrics = constr_metrics_for(scenario, suite, pid)
    constr_indices = [metric_index(suite, pid, m) for m in constr_metrics]
    thresholds     = [THRESHOLDS[(suite, pid)][m] for m in constr_metrics]

    # The mode axis is physical: hard gates evaluability, soft archives
    # infeasible points with raw F/G (see module docstring).
    gated = cfg['mode'] == 'hard'

    if handler in B0_HANDLERS:
        # Unconstrained search over the scoring objectives PLUS each
        # constrained metric not already among them as an ordinary extra
        # objective -- no G, and no duplicated column when a constrained
        # metric (FLOPs under CC/ALL) is itself a scoring objective; its
        # budget is enforced by the gate/scoring only. The callback below
        # still only sees the two scoring objectives + the constraint
        # columns/thresholds, so indicators/feasibility scoring stay
        # identical to every other handler row (search space differs, the
        # scored space does not).
        extra_indices = [ci for ci in constr_indices if ci not in obj_indices]
        search_obj_indices = obj_indices + extra_indices
        problem = B0ObjectiveProblem(benchmark, search_obj_indices)
        assert problem.n_obj == len(OBJ_METRICS) + len(extra_indices), (
            f'{handler} search objective mismatch: n_obj={problem.n_obj} vs '
            f'{len(OBJ_METRICS)} + {len(extra_indices)} ({scenario}/{suite}/pid{pid})')
    else:
        search_obj_indices = obj_indices
        problem = ConstrainedEvoXBenchProblem(benchmark, obj_indices, constr_indices,
                                              thresholds, gate=gated)

    callback = FeasibilityAwareEvoxBenchCallback(
        benchmark, obj_indices, constr_indices, thresholds,
        compute_indicators=compute_indicators)

    algorithm, copy_algorithm, handler_state = build_algorithm(
        method, benchmark, suite, pid, search_obj_indices, constr_indices, thresholds,
        handler, seed, pop_size, n_doe, n_infill, n_gen_inner, inner_pop_size,
        penalty, n_gen, gated=gated)

    results = minimize(
        problem=problem, algorithm=algorithm, termination=('n_gen', n_gen),
        seed=seed, callback=callback, save_history=False, verbose=True,
        copy_algorithm=copy_algorithm,
    )
    data = results.algorithm.callback.data
    if handler_state:
        data['handler_state'] = handler_state

    # Hard mode: persist the rejection log so the discarded evaluations are
    # reconstructable post hoc (X for every gate-discarded architecture,
    # with its violation where the algorithm records one -- RandomGA keeps
    # rejected X only, so its violations are NaN and must be re-evaluated).
    algo = results.algorithm
    if gated:
        # rejected_G keeps the campaign's legacy 1-D shape for a single
        # constraint and one column per constraint otherwise.
        n_constr = len(constr_indices)
        g_shape  = (lambda n: (n,) if n_constr == 1 else (n, n_constr))
        if getattr(algo, '_rejected_X', None) is not None:
            data['rejected_X'] = np.asarray(algo._rejected_X)
            data['rejected_G'] = np.asarray(algo._rejected_G)
        elif getattr(algo, '_rejected_pop', None) is not None and len(algo._rejected_pop) > 0:
            X_rej = np.asarray(algo._rejected_pop.get('X'))
            data['rejected_X'] = X_rej
            data['rejected_G'] = np.full(g_shape(len(X_rej)), np.nan)
        else:
            data['rejected_X'] = np.empty((0, problem.n_var))
            data['rejected_G'] = np.empty(g_shape(0))
    return data


def _resolve_handlers(scenario, args):
    """Handler list for this invocation: explicit --handler > --all_handlers
    (the shared HANDLER_ROW; the reduced MULTI_HANDLER_ROW on s5-s12) > the
    scenario default."""
    default = DEFAULT_HANDLER[SCENARIOS[scenario]['mode']]
    if args.handler is not None:
        return [args.handler], default
    if args.all_handlers:
        row = (MULTI_HANDLER_ROW if SCENARIOS[scenario]['constr'] in MULTI_COMBOS
               else HANDLER_ROW)
        return list(row), default
    return [default], default


def _inner_ga(method, handler):
    """Which inner GA a run actually used: 'sms' for the two SMS-EMOA
    handlers, 'nsga2' for every other samos/samos-cheap handler, None for
    method 'random' (no inner GA at all)."""
    if method == 'random':
        return None
    return 'sms' if handler in ('b0-as-obj', 'h4-cdp-sms') else 'nsga2'


def main(args):
    budget_folder = f'B{args.n_gen * args.pop_size}_P{args.pop_size}'
    scenario = args.scenario
    suite, pid = args.suite, args.pid
    cfg = SCENARIOS[scenario]

    if (suite, pid) not in SCENARIO_INSTANCES[cfg['constr']]:
        print(f"ERROR: {suite}/pid{pid} is not part of scenario {scenario}'s instance set "
              f"({cfg['constr']} constraint) -- valid: "
              f"{SCENARIO_INSTANCES[cfg['constr']]}. See the module docstring, 'Instances'.")
        return 1

    constr_metrics = constr_metrics_for(scenario, suite, pid)
    multi          = len(constr_metrics) > 1
    thresholds     = tuple(THRESHOLDS[(suite, pid)][m] for m in constr_metrics)
    feas_fracs     = tuple(DESIGN_FEASIBLE_FRACTION[(suite, pid)][m] for m in constr_metrics)
    joint_frac     = (JOINT_DESIGN_FEASIBLE_FRACTION[(cfg['constr'], (suite, pid))]
                      if multi else None)

    # Single-constraint pkls keep their original scalar meta values; multi
    # runs carry tuples (plus the joint fraction added at meta-build time).
    constr_metric = constr_metrics if multi else constr_metrics[0]
    threshold     = thresholds     if multi else thresholds[0]
    feas_frac     = feas_fracs     if multi else feas_fracs[0]
    t_str         = ', '.join(f'{m}<={t:.4f}' for m, t in zip(constr_metrics, thresholds))

    handlers, default_handler = _resolve_handlers(scenario, args)

    # (method, handler) work list; random runs ONLY the scenario default.
    pairs = []
    for method in args.method:
        for handler in handlers:
            if method == 'random' and handler != default_handler:
                print(f'[SKIP] random x {handler}: random has no selection pressure or '
                      f'sampling strategy for a handler to act on -- it runs only the '
                      f'scenario default ({default_handler}).')
                continue
            pairs.append((method, handler))

    total_runs = len(args.seeds) * len(pairs)
    run_i = 0
    summary: dict = {}   # (method, handler) -> list of (final_hv, n_feasible, n_total)

    for seed in args.seeds:
        for method, handler in pairs:
            run_i += 1
            save_dir = os.path.join(
                args.results_root, scenario, suite, f'pid{pid}', budget_folder,
                method, handler)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f'seed_{seed}.pkl')

            if os.path.exists(out_path) and not args.overwrite:
                print(f'[SKIP {run_i}/{total_runs}] {scenario}/{suite}/pid{pid}/'
                      f'{method}/{handler}/seed_{seed} already exists')
                with open(out_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f'\n[RUN {run_i}/{total_runs}] scenario={scenario} {suite}/pid{pid} '
                      f'(mode={cfg["mode"]}, constr=[{cfg["constr"]}] {t_str})  '
                      f'method={method}  handler={handler}  '
                      f'seed={seed}  pop={args.pop_size}  n_gen={args.n_gen}')
                data = run_single(
                    method, scenario, suite, pid, handler, seed, args.pop_size,
                    args.n_gen, args.n_doe, args.n_infill, args.n_gen_inner,
                    args.inner_pop_size, args.penalty)
                # Self-describing pkl: everything needed to re-derive this
                # run's config without consulting the output path or the
                # SCENARIOS/THRESHOLDS state at read time. Multi-constraint
                # runs carry tuple-valued constr_metric/threshold/
                # design_feasible_fraction (parallel, one entry per
                # constraint) plus the measured joint fraction; b0 rows
                # additionally record their deduplicated search objective
                # set -- obj_metrics/constr_metric stay the scoring config
                # so analysis groups them with every other handler row.
                data['meta'] = dict(
                    suite=suite, pid=pid, scenario=scenario,
                    obj_metrics=OBJ_METRICS, constr_metric=constr_metric,
                    constr_type=cfg['constr'], threshold=threshold,
                    design_feasible_fraction=feas_frac,
                    mode=cfg['mode'], gate=(cfg['mode'] == 'hard'),
                    handler=handler, method=method, seed=seed,
                    pop_size=args.pop_size, n_gen=args.n_gen,
                    n_gen_inner=args.n_gen_inner, penalty=args.penalty,
                    inner_ga=_inner_ga(method, handler), config='c2',
                )
                if multi:
                    data['meta'].update(
                        constr_metrics=constr_metrics,
                        design_feasible_fraction_joint=joint_frac,
                    )
                if handler in B0_HANDLERS:
                    data['meta']['search_obj_metrics'] = OBJ_METRICS + tuple(
                        m for m in constr_metrics if m not in OBJ_METRICS)
                with open(out_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f'  Saved -> {out_path}')

            final_ind    = data['indicators'][-1] if data['indicators'] else {'hv': float('nan'), 'igd_plus': float('nan')}
            final_nfeas  = data['n_feasible'][-1] if data.get('n_feasible') else 0
            final_ntotal = data['n_total'][-1] if data.get('n_total') else 0
            summary.setdefault((method, handler), []).append(
                (final_ind['hv'], final_nfeas, final_ntotal))

    print('\n' + '=' * 88)
    print(f'Scenario {scenario} on {suite}/pid{pid} (mode={cfg["mode"]}, '
          f'constr: {t_str}) -- final feasible-HV summary '
          f'(mean +/- std over up to {len(args.seeds)} seeds)')
    for method, handler in pairs:
        rows = summary.get((method, handler))
        if not rows:
            continue
        hv_arr     = np.array([r[0] for r in rows], dtype=float)
        nfeas_arr  = np.array([r[1] for r in rows], dtype=float)
        ntotal_arr = np.array([r[2] for r in rows], dtype=float)
        print(f'  {method:>12s} x {handler:<19s} : hv={np.nanmean(hv_arr):.4f} +/- {np.nanstd(hv_arr):.4f}  '
              f'n_feasible={np.nanmean(nfeas_arr):.1f}/{np.nanmean(ntotal_arr):.1f}  (n={len(rows)})')
    print('=' * 88)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', required=True, choices=list(SCENARIOS),
                    help='Which of s1-s4 to run (see module docstring).')
    p.add_argument('--suite', default=SUITE, choices=['c10mop', 'in1kmop', 'citysegmop'],
                    help='Benchmark suite (instance must be in the scenario\'s '
                         'SCENARIO_INSTANCES set).')
    p.add_argument('--pid', type=int, default=PID,
                    help='Benchmark problem id within --suite.')
    p.add_argument('--method', nargs='+', default=list(METHODS), choices=METHODS)
    p.add_argument('--handler', default=None, choices=HANDLERS,
                    help='Constraint handler. Default: the scenario default '
                         '(hard -> h4-cdp, soft -> h2-penalty).')
    p.add_argument('--all_handlers', action='store_true',
                    help='Run the shared handler row (HANDLER_ROW; the two '
                         'inner-GA controls are excluded -- run those via '
                         '--handler). Ignored when --handler is given.')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--pop_size', type=int, default=20)
    p.add_argument('--n_gen', type=int, default=60)
    p.add_argument('--n_doe', type=int, default=None,
                    help='SAMOS2: initial DOE size (default: pop_size)')
    p.add_argument('--n_infill', type=int, default=None,
                    help='SAMOS2: real evaluations per outer generation (default: pop_size)')
    p.add_argument('--n_gen_inner', type=int, default=20,
                    help='SAMOS2: inner GA generations')
    p.add_argument('--inner_pop_size', type=int, default=None,
                    help='SAMOS2: inner GA population size (default: pop_size x 10)')
    p.add_argument('--results_root', default=_DEFAULT_RESULTS_ROOT,
                    help='Output root for per-seed pkls. Point at a dedicated '
                         'smoke-test folder when testing -- never write test '
                         'data into results/ (see CLAUDE.md).')
    p.add_argument('--penalty', type=float, default=1.0,
                    help='Static penalty weight for h2-penalty.')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    sys.exit(main(args))
