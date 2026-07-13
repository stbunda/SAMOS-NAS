"""experiments/constraint/run_constraint.py --- S1-S4 constraint-scenario campaign runner. 
Cloned from experiments/samos/compare_methods.py's run_single/main structure and output conventions 
(skip-if-exists per-seed pkl, EvoxBenchCallback-style ``data`` dict, budget-folder naming), but on
top of the constrained problem pair (problem/evoxbench/constrained_problem.py) and the 
feasibility-aware callback (problem/evoxbench/callbacks.py ::FeasibilityAwareEvoxBenchCallback).

See text/plans/EXPERIMENT_PLAN_constraint_scenarios.md for the campaign design and 
experiments/constraint/THRESHOLDS.md for threshold provenance.

Instances
---------
Selected via --suite/--pid (default c10mop/4, so every pre-existing
invocation behaves identically). Every instance's benchmark column order is
(Err, #Params, FLOPs[, Latency]); objective and constraint columns are
resolved by metric NAME (metric_index), never a hardcoded position, so one
scenario definition below applies unchanged across all of them.

  4-metric (Err/#Params/FLOPs/Latency) -- s1 and s2 (s2's own physical runs;
  s4 reads them too, see Scenarios below):
    c10mop/pid4  : NATS, 5 vars.
    in1kmop/pid9 : MobileNetV3, 21 vars.

  3-metric (Err/#Params/FLOPs) -- s1 and s3. Each of these problems already
  drops one of {#Params, FLOPs} as an objective; that dropped metric becomes
  the natural constraint (s1 keeps FLOPs as an objective and constrains
  #Params, s3 does the reverse):
    c10mop/pid2  : NB101, 26 vars.        c10mop/pid3  : NATS, 5 vars.
    c10mop/pid9  : DARTS, 32 vars.        in1kmop/pid3 : ResNet-50D, 25 vars.
    in1kmop/pid6 : Transformer, 34 vars.  in1kmop/pid8 : MobileNetV3, 21 vars.

All eight share BENCHMARK_META cheap_obj_indices=[1, 2] (#Params, FLOPs both
cheap), so the samos-cheap split logic is identical everywhere. They differ
in normalized_objectives (True for c10mop/pid2,3,4; False otherwise), which
ConstrainedEvoXBenchProblem's normalize-once convention already absorbs --
the constraint metric always lives in the space THRESHOLDS.md computed the
per-instance thresholds in (verified empirically for pid9: ~49% of a 2k
random sample is feasible at each threshold, matching the ~50% design
intent; the six 3-metric instances were checked against their own 10k
samples the same way, see THRESHOLDS.md).

  citysegmop (MoSegNAS, 24 vars) -- s5 and s6 -- both objectives sets keep
  BENCHMARK_META's one cheap structural column (FLOPs/#Params respectively,
  cheap_obj_indices=[2]) so the samos-cheap split still works; UNLIKE
  c10mop/in1kmop, Err. is ALSO predictor-backed here (RankNet/lookup-table
  surrogate), not just the constraint metric:
    citysegmop/pid2 : Err/H1 Lat./FLOPs.    citysegmop/pid3 : Err/H1 Lat./#Params.

Scenarios
---------
Scenario definitions (objectives + constraint metric + handler mode) are
instance-independent; only the numeric threshold varies per instance
(THRESHOLDS dict below). Objectives are two metric NAMES per scenario
(SCENARIOS dict's obj_metrics), resolved to benchmark columns per instance:

  s1 : objectives {Err, FLOPs},   #Params <= T_params, hard  (cheap constraint)
  s2 : objectives {Err, #Params}, Latency <= T_latency, hard (expensive constraint)
  s3 : objectives {Err, #Params}, FLOPs   <= T_flops,   soft (cheap constraint)
  s4 : objectives {Err, #Params}, Latency <= T_latency, soft (expensive constraint)
       -- an ANALYSIS VIEW of s2, not a run of its own: identical objectives,
       constraint, threshold and handler set (only the hard/soft framing
       differs), and the shared feasibility indicator treats the threshold
       as a hard cutoff either way, so a separate s4 run would duplicate s2's
       compute byte-for-byte. --scenario s4 refuses here with a message
       pointing at s2; analyse_constraint.py reads s2's pkls under an s4
       view instead.
  s5 : objectives {Err, FLOPs},   H1 Lat. <= T_h1lat, hard (citysegmop/pid2)
  s6 : objectives {Err, #Params}, H1 Lat. <= T_h1lat, hard (citysegmop/pid3)
       -- s2's role on the citysegmop suite: a hard constraint on a genuinely
       expensive, predictor-backed metric (H1 Lat.), with the objective pair
       chosen per pid so the OTHER objective is that pid's cheap structural
       column (s5 keeps FLOPs for pid2, s6 keeps #Params for pid3), mirroring
       s2's cheap-objectives / expensive-constraint split even though Err. is
       predictor-backed here too (see Instances above).

Handlers
--------
--handler adds the handler axis on top of scenario x method x instance.
Handler wiring is SCENARIO-INDEPENDENT: each handler fully defines its
mechanism; the scenario only picks metric/threshold and (via its hard/soft
mode) the DEFAULT handler. Omitting --handler runs the scenario default
(hard -> h4-cdp, soft -> h2-penalty); --all_handlers runs the
scenario's full row (SCENARIO_HANDLERS) instead. No validation rejects
an off-default scenario x handler combination: stress / wrong-model rows
(h2/h3 penalties on hard scenarios, h1/h4 on soft ones) are deliberate.

  h4-cdp             : native CDP, zero extra wiring. Both constrained
                       problems define out['G'], so pymoo's RankAndCrowding
                       survival (outer archive selection AND the SAMOS2
                       inner NSGA-II) is feasibility-first natively -- see
                       problem/evoxbench/constrained_problem.py.
  h2-penalty         : static penalty. Wraps ONLY the inner surrogate
                       problem in _ConstraintsAsPenaltyMO (pymoo 0.6.1.1's
                       own ConstraintsAsPenalty mis-handles n_obj>1, see its
                       docstring) at the --penalty weight; outer problem and
                       saved pkls keep unpenalized F / raw G.
  h3-adaptive-penalty: strategy.constraints.AdaptivePenaltyProblem. One
                       controller per run; each outer generation the factory
                       adapt()s the weight on the live archive's feasible
                       fraction (target 0.5, c=1.2), then wrap()s the fresh
                       inner problem. Needs copy_algorithm=False (the
                       factory closure reads the live algorithm's archive;
                       pymoo's default deepcopy severs that link).
  h5-eps             : strategy.constraints.EpsilonRelaxation. Outer-clocked
                       epsilon schedule (eps0 = mean DOE CV, linear to 0 at
                       50% of --n_gen); per outer generation the factory does
                       maybe_init_eps0 / wrap / advance. Needs
                       copy_algorithm=False (same live-archive reason as h3).
  h6-sr              : strategy.constraints.DominanceStochasticRanking
                       (Pf=0.45, seeded per run) as the inner NSGA-II
                       survival, via partial(NSGA2, survival=...).
  h1-rejection       : rejection. random -> RejectionSampling around the
                       base sampler with an exact make_benchmark_g_fn;
                       samos/samos-cheap -> RejectionInfillSelector on the
                       C3 infill seam (rejects on the inner problem's G:
                       predicted for samos, exact for cheap constraints),
                       AND the inner NSGA-II survival is a RankAndCrowding
                       whose filter_infeasible is set False so the row
                       isolates pure rejection rather than a rejection+CDP
                       hybrid. pymoo 0.6.1.1's RankAndCrowding
                       constructor hardcodes filter_infeasible=True (no
                       kwarg), but it is a plain instance attribute
                       (Survival.__init__ sets it), so it is flipped
                       post-construction. ponytail: NSGA2's binary_
                       tournament mating selection still compares CV-first
                       when a parent is infeasible -- only the survival
                       mechanism is replaced, so that residual feasibility
                       pressure is accepted and documented, not patched.
  b0-as-obj          : constraint-as-objective baseline (B0). No G anywhere
                       in the search: the outer problem is UNCONSTRAINED,
                       3-objective (the scenario's two objectives plus its
                       constrained metric appended as an ordinary third
                       objective, resolved by name; asserted == 3 in
                       run_single). samos -> all 3 objectives predicted (3
                       surrogates); samos-cheap -> the instance's cheap
                       columns among those 3 evaluated exactly, the rest
                       predicted (same _obj_split split as every other
                       handler, just over 3 columns instead of 2). No
                       constr_surrogate for either method -- the constrained
                       metric IS an objective here. Search mechanism is
                       SMS-EMOA (SAMOS2's inner_algorithm), chosen by
                       preliminary study (D31, experiments/constraint/
                       prelim_3obj.py): plain NSGA-II's crowding-distance
                       selection degrades noticeably at 3 objectives on
                       these instances, SMS-EMOA did not. Evaluation/
                       scoring is unaffected: the callback still gets the
                       scenario's TWO objective columns + the usual
                       constr_index/threshold, so indicators stay comparable
                       to every other handler row (the 3-obj archive is
                       projected onto the feasible 2-obj front post hoc, see
                       problem/evoxbench/callbacks.py). SAMOS2's archive-
                       seeding RankAndCrowding (top_pop in _infill) still
                       runs plain (feasibility-agnostic) crowding at 3
                       objectives for this handler -- that degradation is
                       deliberately left in place; it is part of what B0
                       measures, not a bug to fix here. Output path gets a
                       distinct 3-objective objtag (_objtag_b0 below) so a
                       b0-as-obj run can never be mistaken for a 2-objective
                       row by path alone; its meta additionally records the
                       3 search objectives next to the 2 scoring objectives
                       (see 'Output layout').

For method 'random', only h1-rejection and the scenario default are run:
every other handler -- including b0-as-obj -- acts purely on selection
pressure that RandomGA does not have, so those runs would be byte-identical
duplicates of its default run -- the runner prints [SKIP] instead and
analysis replicates the row.

Output layout (adds an objective-set tag level on top of the handler level):
  {results_root}/{scenario}/{suite}/pid{pid}/{objtag}/{budget}/{method}/{handler}/seed_{N}.pkl
objtag identifies the scenario's objective set ('obj-err-flops' for s1,
'obj-err-params' for s2/s3/s4 -- see _objtag()), so the same (scenario,
suite, pid) can never silently collide across an objective-set change. This
module writes and skip-if-exists-checks ONLY the tagged path; paths without
an objtag level are pre-existing history from before objectives became
per-scenario and are never read or rewritten here (analyse_constraint.py
still reads them, as legacy data -- see its own docstring). Round-1 flat
pkls ({method}/seed_N.pkl, no handler or objtag level at all) are migrated
by COPY -- never moved or deleted -- into the scenario-default handler
subdir via --migrate (s1/s2 -> h4-cdp, s3/s4 -> h2-penalty); that migration
predates the objtag level and is unaffected by it. Every new-phase pkl also
carries a 'meta' dict (suite/pid/scenario/objtag/objectives/constraint/
threshold/mode/handler/method/seed/pop_size/n_gen/n_gen_inner/config) so it
is self-describing without consulting the writing code's current SCENARIOS/
THRESHOLDS state; legacy pkls predate this key.

b0-as-obj's objtag is DIFFERENT from its scenario's normal objtag: it
encodes the 3-column search-objective set instead (_objtag_b0() below,
e.g. 'obj-err-flops-params-b0' for s1), so the path alone makes an
unconstrained 3-objective run impossible to confuse with a 2-objective
handler row even though the handler-subdir level would already prevent a
literal collision. meta['obj_metrics'] / meta['constr_metric'] still record
the scenario's normal 2-objective scoring config (unchanged, so
analyse_constraint.py's config-signature groups b0-as-obj rows with every
other handler row of the same scenario -- they are scored in the same
2-objective space); meta gains a 'search_obj_metrics' key (3 names) on
b0-as-obj rows only, recording what was actually searched.

Methods
-------
  random       : RandomGA (no surrogate, no selection pressure) on
                 ConstrainedEvoXBenchProblem, exact benchmark G. Handler mode
                 (hard/soft) is irrelevant to it -- it has no ranking to
                 penalize or CDP-filter differently -- so it runs identically
                 in hard and soft scenarios; results should therefore be
                 near-identical between e.g. s1 and s3 for this method (any
                 difference is RNG-stream noise from the extra ConstraintsAs
                 Penalty wrapper never being constructed for this method, not
                 a real behavioural change).
  samos        : SAMOS2, ALL output objectives predicted (predict=[0, 1],
                 real=[]), XGBoost surrogates (one per predicted position,
                 per-seed subseeded like compare_methods.py). The constraint
                 is ALWAYS predicted for this method (constr_surrogate=
                 XGBoost), regardless of scenario.
  samos-cheap  : SAMOS2 cheap-real split, derived from BENCHMARK_META's
                 cheap_obj_indices for the instance ({1, 2} = #Params, FLOPs
                 for every supported instance): predict=[0] (Err), real=[1]
                 (the scenario's other objective -- FLOPs for s1, #Params for
                 s2/s3/s4 -- always a cheap column since both #Params and
                 FLOPs are). Constraint: exact (constr_surrogate=None) when
                 the constraint metric is itself a cheap column (s1's
                 #Params, s3's FLOPs) -- this is H7 (exact cheap filter) for
                 free; predicted (constr_surrogate=XGBoost) otherwise (s2/s4,
                 Latency, expensive).

Known simplifications:
  (a) SAMOS2's archive-level RankAndCrowding seeding (top_pop in _infill)
      stays feasibility-first even in soft scenarios -- it operates on the
      *outer* archive, which always carries raw (unpenalized) G.
  (b) Infill/subset selection compares penalized candidate F (from the
      ConstraintsAsPenalty-wrapped inner problem, soft scenarios only)
      against unpenalized archive F (the outer archive) when ranking for
      diversity. Both are accepted for round 1; see the experiment plan.

Examples
--------
  python experiments/constraint/run_constraint.py --scenario s1
  python experiments/constraint/run_constraint.py --scenario s1 --suite c10mop --pid 3
  python experiments/constraint/run_constraint.py --scenario s3 --suite in1kmop --pid 8
  python experiments/constraint/run_constraint.py --scenario s2 --method samos --handler h3-adaptive-penalty
  python experiments/constraint/run_constraint.py --scenario s1 --all_handlers
  python experiments/constraint/run_constraint.py --scenario s1 --method samos --handler b0-as-obj \\
      --suite c10mop --pid 3
  python experiments/constraint/run_constraint.py --scenario s4              # refuses: see --scenario s2
  python experiments/constraint/run_constraint.py --migrate                     # copy round-1 flat pkls
  python experiments/constraint/run_constraint.py --scenario s2 --method samos-cheap \\
      --pop_size 8 --n_gen 3 --n_gen_inner 4 --results_root /tmp/smoke
  python experiments/constraint/run_constraint.py --scenario s1 --threshold_set q25 \\
      --results_root results/constraint_25   # ~25%-feasible thresholds (THRESHOLDS_Q25)
"""

import argparse
import glob
import os
import pickle
import random
import shutil
import sys
from functools import partial

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'samos'))  # for _common
sys.path.insert(0, _REPO_ROOT)

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.core.individual import calc_cv
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.misc import from_dict

from _common import _make_surrogate

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
    RejectionSampling,
    feasible_fraction,
    make_benchmark_g_fn,
)
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.sampler import EvoxBenchSampler
from strategy.surrogate.models import XGBoost
from strategy.surrogate.samos2 import SAMOS2

# ─── default instance (module-level, used only as the --suite/--pid CLI
# defaults so every pre-existing invocation keeps behaving identically) ───────
SUITE = 'c10mop'
PID   = 4

# --results_root default; also used in main()'s --threshold_set guard (a
# non-q50 --threshold_set may never write into this tree, see main()).
_DEFAULT_RESULTS_ROOT = os.path.join('results', 'constraint')

# ─── per-instance thresholds (config, not code) ──────────────────────────────
# Provenance: experiments/constraint/THRESHOLDS.md -- per instance, the
# median of the constraint metric over a fixed-seed (0) 10k random sample, in
# the space the constrained problem operates in (benchmark.evaluate() output,
# benchmark.normalize() applied ONLY when not benchmark.normalized_objectives
# -- normalized natively for c10mop/pid2,3,4, via that convention for
# everything else). ~50% of the sample is feasible at each threshold by
# construction. Keyed by (suite, pid), then by the same metric names
# SCENARIOS uses.
THRESHOLDS = {
    ('c10mop', 4): {
        '#Params': 0.4383147965648318,
        'Latency': 0.45870737801039463,
    },
    ('in1kmop', 9): {
        '#Params': 0.554389375817845,
        'Latency': 0.4606726558251625,
    },
    ('c10mop', 2): {
        '#Params': 0.07970941037337388,
        'FLOPs':   0.0816152005844248,
    },
    ('c10mop', 3): {
        '#Params': 0.4383147965648318,
        'FLOPs':   0.34256897616596726,
    },
    ('c10mop', 9): {
        '#Params': 0.418475,
        'FLOPs':   0.4018467755799336,
    },
    ('in1kmop', 3): {
        '#Params': 0.36461655406484417,
        'FLOPs':   0.27081408509201554,
    },
    ('in1kmop', 6): {
        '#Params': 0.4490521985657902,
        'FLOPs':   0.450714583214995,
    },
    ('in1kmop', 8): {
        '#Params': 0.5542505032341564,
        'FLOPs':   0.49379317155236824,
    },
    ('citysegmop', 2): {
        'H1 Lat.': 0.785908199618216,
    },
    ('citysegmop', 3): {
        'H1 Lat.': 0.785300373420361,
    },
}

# Q25 variant of THRESHOLDS above: same (suite, pid) keys, same metric names,
# but each value is the Q25 (not median) row of the quartile tables in
# experiments/constraint/THRESHOLDS.md -- the SAME fixed-seed (0) 10k random
# samples the medians came from, just the lower quartile instead of the
# midpoint, so ~25% of the sample is feasible at each threshold instead of
# ~50%. Selected via --threshold_set q25 (see THRESHOLD_SETS below); default
# behavior (--threshold_set q50, THRESHOLDS above) is unaffected.
THRESHOLDS_Q25 = {
    ('c10mop', 4): {
        '#Params': 0.3078135998873715,
        'Latency': 0.36311633657280196,
    },
    ('in1kmop', 9): {
        '#Params': 0.46727868434691233,
        'Latency': 0.34752772487724964,
    },
    ('c10mop', 2): {
        '#Params': 0.047260473500094415,
        'FLOPs':   0.04731894337649006,
    },
    ('c10mop', 3): {
        '#Params': 0.3078135998873715,
        'FLOPs':   0.1814415868239766,
    },
    ('c10mop', 9): {
        '#Params': 0.3468375,
        'FLOPs':   0.3282217003685359,
    },
    ('in1kmop', 3): {
        '#Params': 0.2617528196589368,
        'FLOPs':   0.16534922735000696,
    },
    ('in1kmop', 6): {
        '#Params': 0.2839906468663558,
        'FLOPs':   0.28835792673038946,
    },
    ('in1kmop', 8): {
        '#Params': 0.46660078570385494,
        'FLOPs':   0.37242751750291636,
    },
    ('citysegmop', 2): {
        'H1 Lat.': 0.5561883179426053,
    },
    ('citysegmop', 3): {
        'H1 Lat.': 0.5569287648808446,
    },
}

# Lookup by --threshold_set flag value.
THRESHOLD_SETS = {'q50': THRESHOLDS, 'q25': THRESHOLDS_Q25}

# SCENARIOS is data, not branching code, and is instance-independent: the
# objective metric names, constraint metric and handler mode define the
# scenario; the numeric threshold is looked up per instance in THRESHOLDS,
# the benchmark column indices per instance in obj_indices_for()/metric_index.
# s4 has no runs of its own -- 'view_of' names the scenario whose physical
# output it reads instead (see module docstring, 'Scenarios').
SCENARIOS = {
    's1': dict(obj_metrics=('Err.', 'FLOPs'),   constr_metric='#Params', mode='hard'),
    's2': dict(obj_metrics=('Err.', '#Params'), constr_metric='Latency', mode='hard'),
    's3': dict(obj_metrics=('Err.', '#Params'), constr_metric='FLOPs',   mode='soft'),
    's4': dict(obj_metrics=('Err.', '#Params'), constr_metric='Latency', mode='soft', view_of='s2'),
    's5': dict(obj_metrics=('Err.', 'FLOPs'),   constr_metric='H1 Lat.', mode='hard'),   # citysegmop/pid2
    's6': dict(obj_metrics=('Err.', '#Params'), constr_metric='H1 Lat.', mode='hard'),   # citysegmop/pid3
}

# Output-path tag per scenario's objective set (see module docstring, 'Output
# layout'). Metric names are mapped to short slugs; every combination in
# SCENARIOS today is covered explicitly, with a generic fallback for future
# additions. 'H1 Lat.' needs its own entry -- the generic fallback would
# produce a slug containing a space (path-unsafe).
_METRIC_SLUG = {'Err.': 'err', '#Params': 'params', 'FLOPs': 'flops', 'Latency': 'latency',
                'H1 Lat.': 'h1lat'}


def _objtag(obj_metrics):
    return 'obj-' + '-'.join(_METRIC_SLUG.get(m, m.lower().strip('.#')) for m in obj_metrics)


def _objtag_b0(search_obj_metrics):
    """b0-as-obj's own objtag: the 3-column SEARCH objective set (scenario's
    2 objectives + its constrained metric), suffixed so it can never be
    mistaken for a 2-objective row by path alone (see module docstring,
    'Output layout'). E.g. s1 -> 'obj-err-flops-params-b0'."""
    return _objtag(search_obj_metrics) + '-b0'


def obj_indices_for(suite, pid, scenario):
    """Benchmark output-column indices for a scenario's objectives at one
    instance, resolved by metric NAME (KeyError if this instance does not
    define one of them)."""
    return [metric_index(suite, pid, m) for m in SCENARIOS[scenario]['obj_metrics']]


METHODS = ['random', 'samos', 'samos-cheap']

# ─── handler axis ─────────────────────────────────────────────────────────────
HANDLERS = ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty',
            'h4-cdp', 'h5-eps', 'h6-sr', 'b0-as-obj',
            # Inner-GA control rows (not part of any scenario's default row
            # set): they complete the {formulation} x {inner GA} factorial so
            # b0-as-obj's advantage can be attributed to the formulation or
            # to SMS-EMOA. Run explicitly via --handler / the sbatch block.
            'h4-cdp-sms',   # constrained CDP, SMS-EMOA inner GA
            'b0-nsga2']     # constraint-as-objective, NSGA-II inner GA

# Handlers that search the 3-column objective set (scenario objectives +
# constrained metric) instead of defining G.
B0_HANDLERS = ('b0-as-obj', 'b0-nsga2')

# Scenario-default handler = the round-1 wiring, keyed by the scenario's mode.
DEFAULT_HANDLER = {'hard': 'h4-cdp', 'soft': 'h2-penalty'}

# Scenario -> handler-row matrix (config, not code). Includes each
# scenario's default; --all_handlers runs exactly this row. S2 carries the
# h2/h3 penalty stress rows; s3/s4 carry the h1/h4 wrong-model controls.
# b0-as-obj (D31) is on every row -- s4 stays a view of s2, so its row just
# documents the same set s2 physically runs.
SCENARIO_HANDLERS = {
    's1': ['h1-rejection', 'h2-penalty', 'h4-cdp', 'h5-eps', 'h6-sr', 'b0-as-obj'],
    's2': ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp', 'h5-eps', 'h6-sr', 'b0-as-obj'],
    's3': ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp', 'h6-sr', 'b0-as-obj'],
    's4': ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp', 'h6-sr', 'b0-as-obj'],
    's5': ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp', 'h5-eps', 'h6-sr', 'b0-as-obj'],
    's6': ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp', 'h5-eps', 'h6-sr', 'b0-as-obj'],
}

# Handlers that act on selection pressure only are meaningless for RandomGA:
# for method 'random' the runner executes only the scenario default
# and h1-rejection, and prints [SKIP] for the rest.
RANDOM_VALID_EXTRA = {'h1-rejection'}


class _ConstraintsAsPenaltyMO(ConstraintsAsPenalty):
    """Deviation from the plan's plain ``ConstraintsAsPenalty`` (found during
    verification): pymoo 0.6.1.1's ``ConstraintsAsPenalty.do()`` computes
    ``F + penalty * np.reshape(CV, F.shape)``. ``calc_cv`` returns one
    aggregated violation per individual, shape ``(n,)`` -- that reshape only
    works when ``F`` is single-column (``n.size == n_obj==1 * n``). Our inner
    problems have ``n_obj=len(obj_indices)=2`` (every current scenario picks
    exactly two objectives) with one constraint, so ``CV.size == n`` but
    ``F.size == 2n``, and pymoo's own
    reshape raises ``ValueError: cannot reshape array of size n into shape
    (n,2)`` (confirmed against the installed build). This override is
    otherwise identical to pymoo's; it only replaces the exact reshape with
    ``CV.reshape(-1, 1)`` broadcasting, applying the same per-individual
    penalty to every objective column -- the standard multi-objective
    generalization of static penalty (H2)."""

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)
        F, G, H = from_dict(out, 'F', 'G', 'H')
        out['__F__'], out['__G__'], out['__H__'] = F, G, H
        CV = calc_cv(G=G, H=H)
        out['F'] = F + self.penalty * CV.reshape(-1, 1)
        del out['G']
        del out['H']
        return out


def _obj_split(obj_indices, cheap_cols):
    """(predict_pos, real_pos): output-column POSITIONS into obj_indices,
    split by whether the underlying benchmark column is cheap."""
    real_pos    = [pos for pos, col in enumerate(obj_indices) if col in cheap_cols]
    predict_pos = [pos for pos in range(len(obj_indices)) if pos not in real_pos]
    return predict_pos, real_pos


class B0ObjectiveProblem(Problem):
    """Outer problem for the b0-as-obj handler: UNCONSTRAINED search over
    ``obj_indices`` benchmark columns (the scenario's 2 objectives + its
    constrained metric, in that order) -- no ``G`` anywhere. Same
    evaluate-once / normalize-once / non-finite-guard convention as
    ``ConstrainedEvoXBenchProblem`` (problem/evoxbench/constrained_problem.py),
    minus the constraint machinery."""

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
    """Inner problem for the b0-as-obj handler: the unconstrained,
    3-objective counterpart of ``ConstrainedSurrogateProblemEvox`` (same
    predict/real split over output-column POSITIONS into ``obj_indices``,
    same evaluate-once / normalize-once / non-finite-guard convention) --
    minus the constraint. No ``constr_surrogate`` seam: the constrained
    metric is an ordinary predicted-or-real objective column here."""

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


def build_algorithm(method, benchmark, suite, pid, obj_indices, constr_index,
                     threshold, handler, seed, pop_size, n_doe, n_infill,
                     n_gen_inner, inner_pop_size, penalty, n_gen):
    """Returns ``(algorithm, copy_algorithm)``; the second element is the
    ``copy_algorithm`` value ``minimize`` must be called with. It is False
    only for h3/h5 (their per-generation state objects read the live
    algorithm's archive through the factory closure's ``algo_ref``; pymoo's
    default algorithm deepcopy would sever that handle) and pymoo's default
    True everywhere else."""
    xl = np.asarray(benchmark.search_space.lb, dtype=int)
    xu = np.asarray(benchmark.search_space.ub, dtype=int)

    sampler   = EvoxBenchSampler(xl, xu)
    crossover = IntegerUniformCrossover(prob=0.9)
    mutation  = IntegerPointMutation(xl, xu)
    elim      = IntegerVectorDuplicateElimination()

    n_doe_    = n_doe if n_doe is not None else pop_size
    n_infill_ = n_infill if n_infill is not None else pop_size
    inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

    if method == 'random':
        # No surrogate, no selection pressure: every handler except
        # h1-rejection is a no-op for this method (main() [SKIP]s them).
        # h1 swaps the sampler for exact rejection sampling; DOE and every
        # infill batch RandomGA draws are then rejection-resampled.
        if handler == 'h1-rejection':
            g_fn    = make_benchmark_g_fn(benchmark, constr_index, threshold)
            sampler = RejectionSampling(sampler, g_fn)
        return RandomGA(pop_size=pop_size, sampling=sampler, eliminate_duplicates=elim), True

    rng = np.random.RandomState(seed)

    if method == 'samos':
        predict_pos = list(range(len(obj_indices)))
        real_pos    = []
        surrogates       = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_pos]
        # b0 rows: no constr_surrogate -- the constrained metric is an
        # ordinary predicted objective column here, never a G.
        constr_surrogate = (None if handler in B0_HANDLERS
                             else XGBoost(100, seed=rng.randint(0, 2**31 - 1)))

    elif method == 'samos-cheap':
        cheap_cols = set(BENCHMARK_META[suite][pid].get('cheap_obj_indices', []))
        predict_pos, real_pos = _obj_split(obj_indices, cheap_cols)
        surrogates = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_pos]
        if handler in B0_HANDLERS:
            constr_surrogate = None   # same reason as the 'samos' branch above.
        else:
            exact_constr = constr_index in cheap_cols   # H7: cheap constraint -> exact
            constr_surrogate = None if exact_constr else XGBoost(100, seed=rng.randint(0, 2**31 - 1))

    else:
        raise ValueError(f'Unknown method: {method!r}')

    if handler in B0_HANDLERS:
        # B0 baseline: unconstrained ``len(obj_indices)``-objective (== 3,
        # enforced by the caller) search, no wrap_inner, no constr_surrogate.
        # 'b0-as-obj' uses SMS-EMOA as the inner algorithm -- empirically the
        # strongest of {NSGA-II, SMS-EMOA, NSGA-III} at 3 objectives on these
        # instances (experiments/constraint/prelim_3obj.py); 'b0-nsga2' is
        # the inner-GA control (same formulation, NSGA-II inner). SAMOS2's
        # archive-seeding RankAndCrowding (top_pop in _infill) still runs
        # plain crowding at 3 objectives here -- deliberately left as part
        # of what this baseline measures, see module docstring.
        def factory(surrs):
            return B0SurrogateProblemEvox(surrs, obj_indices, predict_pos, real_pos, benchmark)

        algorithm = SAMOS2(
            sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
            predict_obj_indices=predict_pos,
            crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
            n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
            inner_algorithm=SMSEMOA if handler == 'b0-as-obj' else NSGA2,
        )
        return algorithm, True

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
        # Inner-GA control for the b0 comparison: identical constraint
        # wiring to h4-cdp (native CDP -- pymoo's SMSEMOA survival is also
        # feasibility-first via the Survival base class), only the inner
        # algorithm changes.
        samos2_kwargs['inner_algorithm'] = SMSEMOA

    elif handler == 'h2-penalty':
        def wrap_inner(inner):
            # Static penalty on the inner problem only. _ConstraintsAsPenaltyMO,
            # not the plain pymoo class -- see its docstring (multi-objective
            # reshape bug in pymoo 0.6.1.1).
            return _ConstraintsAsPenaltyMO(inner, penalty=penalty)

    elif handler == 'h3-adaptive-penalty':
        pen = AdaptivePenaltyProblem(w0=1.0, target=0.5, c=1.2)

        def wrap_inner(inner):
            arc = algo_ref[0]._archive    # live archive (copy_algorithm=False)
            if len(arc) > 0:
                pen.adapt(feasible_fraction(arc))
            return pen.wrap(inner)

        copy_algorithm = False            # live-archive handle, see docstring

    elif handler == 'h5-eps':
        eps = EpsilonRelaxation(n_gen_total=n_gen)

        def wrap_inner(inner):
            eps.maybe_init_eps0(algo_ref[0]._archive)   # eps0 = mean DOE CV
            wrapped = eps.wrap(inner)                   # snapshots epsilon(t)
            eps.advance()                               # t+1 for next generation
            return wrapped

        copy_algorithm = False            # live-archive handle, see docstring

    elif handler == 'h6-sr':
        samos2_kwargs['inner_algorithm'] = partial(
            NSGA2, survival=DominanceStochasticRanking(Pf=0.45, seed=seed))

    elif handler == 'h1-rejection':
        # Rejection on the C3 infill seam (inner candidates already carry the
        # inner problem's G -- predicted for samos, exact for cheap
        # constraints), PLUS pure-rejection semantics: the inner
        # NSGA-II survival must NOT be feasibility-first. pymoo 0.6.1.1's
        # RankAndCrowding constructor hardcodes filter_infeasible=True (no
        # kwarg), but it is a plain instance attribute set by
        # Survival.__init__, so it is flipped post-construction; the single
        # instance is safely reused across outer generations (stateless).
        samos2_kwargs['infill_selector'] = RejectionInfillSelector()
        _surv = RankAndCrowding()
        _surv.filter_infeasible = False
        samos2_kwargs['inner_algorithm'] = partial(NSGA2, survival=_surv)

    else:
        raise ValueError(f'Unknown handler: {handler!r}')

    def factory(surrs, fitted_constr_surrogate=None):
        inner = ConstrainedSurrogateProblemEvox(
            surrs, obj_indices, predict_pos, real_pos, benchmark,
            constr_index, threshold, constr_surrogate=fitted_constr_surrogate,
        )
        return wrap_inner(inner) if wrap_inner is not None else inner

    algorithm = SAMOS2(
        sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
        predict_obj_indices=predict_pos,
        crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
        n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
        constr_surrogate=constr_surrogate,
        **samos2_kwargs,
    )
    algo_ref.append(algorithm)   # late-bind the live handle for h3/h5 closures
    return algorithm, copy_algorithm


def run_single(method, scenario, suite, pid, handler, seed, pop_size, n_gen,
               n_doe, n_infill, n_gen_inner, inner_pop_size, penalty,
               threshold_set='q50', compute_indicators=True):
    np.random.seed(seed)
    random.seed(seed)

    cfg          = SCENARIOS[scenario]
    benchmark    = get_benchmark(suite, pid)
    obj_indices  = obj_indices_for(suite, pid, scenario)   # scenario's 2 scoring objectives -- always
    constr_index = metric_index(suite, pid, cfg['constr_metric'])
    threshold    = THRESHOLD_SETS[threshold_set][(suite, pid)][cfg['constr_metric']]

    if handler in B0_HANDLERS:
        # B0: unconstrained search over the scenario's 2 objectives
        # PLUS the constrained metric as an ordinary 3rd objective -- no G.
        # The callback below still only sees the scenario's 2 objectives +
        # constr_index/threshold, so indicators/feasibility scoring stay
        # identical to every other handler row (search space differs, the
        # scored space does not -- see module docstring).
        search_obj_indices = obj_indices + [constr_index]
        problem = B0ObjectiveProblem(benchmark, search_obj_indices)
        assert problem.n_obj == 3, (
            f'{handler} requires exactly 3 search objectives, got {problem.n_obj} '
            f'({scenario}/{suite}/pid{pid})')
    else:
        search_obj_indices = obj_indices
        problem = ConstrainedEvoXBenchProblem(benchmark, obj_indices, constr_index, threshold)

    callback = FeasibilityAwareEvoxBenchCallback(
        benchmark, obj_indices, constr_index, threshold,
        compute_indicators=compute_indicators)

    algorithm, copy_algorithm = build_algorithm(
        method, benchmark, suite, pid, search_obj_indices, constr_index, threshold,
        handler, seed, pop_size, n_doe, n_infill, n_gen_inner, inner_pop_size,
        penalty, n_gen)

    results = minimize(
        problem=problem, algorithm=algorithm, termination=('n_gen', n_gen),
        seed=seed, callback=callback, save_history=False, verbose=True,
        copy_algorithm=copy_algorithm,
    )
    return results.algorithm.callback.data


def migrate(args):
    """Copy round-1 FLAT pkls ({...}/{method}/seed_N.pkl) into their
    scenario-default handler subdir ({...}/{method}/{h4-cdp|h2-penalty}/
    seed_N.pkl). Copies only -- originals are NEVER moved or deleted
    (CLAUDE.md / plan Round 2). Idempotent: existing destinations are left
    untouched. Works on whatever instances/budgets exist under
    --results_root."""
    n_copied = n_skipped = 0
    for scenario, cfg in SCENARIOS.items():
        default_handler = DEFAULT_HANDLER[cfg['mode']]
        # Fixed-depth flat layout: scenario/suite/pidX/budget/method/seed_N.pkl.
        # Handler-subdir pkls live one level deeper and cannot match this glob.
        pattern = os.path.join(args.results_root, scenario, '*', 'pid*', '*', '*', 'seed_*.pkl')
        for src in sorted(glob.glob(pattern)):
            method = os.path.basename(os.path.dirname(src))
            if method not in METHODS:
                continue   # not a flat method dir (defensive)
            dst_dir = os.path.join(os.path.dirname(src), default_handler)
            dst = os.path.join(dst_dir, os.path.basename(src))
            if os.path.exists(dst):
                n_skipped += 1
                print(f'[MIGRATE-SKIP] exists: {dst}')
                continue
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst)
            n_copied += 1
            print(f'[MIGRATE] {src}  ->  {dst}')
    print(f'[MIGRATE] done: {n_copied} copied, {n_skipped} already present '
          f'(originals left in place).')
    return 0


def _resolve_handlers(scenario, args):
    """Handler list for this invocation: explicit --handler > --all_handlers
    (the scenario's full SCENARIO_HANDLERS row) > the scenario default."""
    default = DEFAULT_HANDLER[SCENARIOS[scenario]['mode']]
    if args.handler is not None:
        return [args.handler], default
    if args.all_handlers:
        return list(SCENARIO_HANDLERS[scenario]), default
    return [default], default


def _inner_ga(method, handler):
    """meta['inner_ga']: which inner GA a run actually used (see
    build_algorithm) -- 'sms' for the two SMS-EMOA handlers (b0-as-obj,
    h4-cdp-sms), 'nsga2' for every other samos/samos-cheap handler, None for
    method 'random' (no inner GA at all)."""
    if method == 'random':
        return None
    return 'sms' if handler in ('b0-as-obj', 'h4-cdp-sms') else 'nsga2'


def main(args):
    budget_folder = f'B{args.n_gen * args.pop_size}_P{args.pop_size}'
    scenario = args.scenario
    suite, pid = args.suite, args.pid
    cfg = SCENARIOS[scenario]

    if 'view_of' in cfg:
        print(f"ERROR: --scenario {scenario} has no runs of its own -- it is an analysis "
              f"view of --scenario {cfg['view_of']}'s physical output (identical objectives, "
              f"constraint, threshold and handler set; only the hard/soft framing differs, "
              f"see module docstring, 'Scenarios'). Run --scenario {cfg['view_of']} instead; "
              f"analyse_constraint.py reads its pkls under an {scenario} view too.")
        return 1

    if args.threshold_set != 'q50' and args.results_root == _DEFAULT_RESULTS_ROOT:
        print(f"ERROR: --threshold_set {args.threshold_set} must not write into the default "
              f"--results_root ({_DEFAULT_RESULTS_ROOT}) -- that tree holds q50 threshold data. "
              f"Point --results_root at a dedicated tree for this threshold set (e.g. "
              f"results/constraint_25 for q25) -- threshold sets must never be mixed in one "
              f"results tree.")
        return 1

    thresholds = THRESHOLD_SETS[args.threshold_set]
    if (suite, pid) not in thresholds:
        print(f'ERROR: no thresholds defined for {suite}/pid{pid} '
              f'(available: {sorted(thresholds)}) -- see THRESHOLDS.md.')
        return 1
    if cfg['constr_metric'] not in thresholds[(suite, pid)]:
        print(f"ERROR: {suite}/pid{pid} has no {cfg['constr_metric']!r} threshold "
              f"(scenario {scenario} constrains it) -- available: "
              f"{sorted(thresholds[(suite, pid)])}; see THRESHOLDS.md.")
        return 1
    threshold = thresholds[(suite, pid)][cfg['constr_metric']]

    try:
        obj_indices_for(suite, pid, scenario)   # validate objectives resolve before any runs start
    except KeyError as exc:
        print(f'ERROR: {exc}')
        return 1

    objtag = _objtag(cfg['obj_metrics'])
    # b0-as-obj's own 3-column objtag (scenario objectives + constrained
    # metric); only used for handler == 'b0-as-obj' below, see module
    # docstring ('Output layout').
    b0_search_obj_metrics = tuple(cfg['obj_metrics']) + (cfg['constr_metric'],)
    b0_objtag = _objtag_b0(b0_search_obj_metrics)
    handlers, default_handler = _resolve_handlers(scenario, args)

    # (method, handler) work list; RandomGA runs only its default + h1.
    pairs = []
    for method in args.method:
        for handler in handlers:
            if method == 'random' and handler != default_handler \
                    and handler not in RANDOM_VALID_EXTRA:
                print(f'[SKIP] random x {handler}: handler acts on selection '
                      f'pressure RandomGA does not have -- would be byte-identical '
                      f'to random x {default_handler} (analysis replicates the row).')
                continue
            pairs.append((method, handler))

    total_runs = len(args.seeds) * len(pairs)
    run_i = 0
    summary: dict = {}   # (method, handler) -> list of (final_hv, n_feasible, n_total)

    for seed in args.seeds:
        for method, handler in pairs:
            run_i += 1
            # b0 rows write under their own 3-objective objtag; every other
            # handler keeps the scenario's normal (2-objective) objtag.
            this_objtag = b0_objtag if handler in B0_HANDLERS else objtag
            save_dir = os.path.join(
                args.results_root, scenario, suite, f'pid{pid}', this_objtag, budget_folder,
                method, handler)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f'seed_{seed}.pkl')

            if os.path.exists(out_path) and not args.overwrite:
                print(f'[SKIP {run_i}/{total_runs}] {scenario}/{suite}/pid{pid}/{this_objtag}/'
                      f'{method}/{handler}/seed_{seed} already exists')
                with open(out_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f'\n[RUN {run_i}/{total_runs}] scenario={scenario} {suite}/pid{pid} '
                      f'(mode={cfg["mode"]}, objectives={"/".join(cfg["obj_metrics"])}, '
                      f'constr={cfg["constr_metric"]}, T={threshold:.4f})  '
                      f'method={method}  handler={handler}  seed={seed}  '
                      f'pop={args.pop_size}  n_gen={args.n_gen}')
                data = run_single(
                    method, scenario, suite, pid, handler, seed, args.pop_size,
                    args.n_gen, args.n_doe, args.n_infill, args.n_gen_inner,
                    args.inner_pop_size, args.penalty, args.threshold_set)
                # Self-describing pkl: everything needed to re-derive this
                # run's config without consulting the output path or
                # SCENARIOS/THRESHOLDS at whatever version they are when the
                # pkl is read later. Legacy (untagged-path) pkls predate this
                # key; analyse_constraint.py falls back to its
                # path+LEGACY_CONFIG map when it is absent.
                # obj_metrics/constr_metric/mode stay the scenario's normal
                # 2-objective SCORING config even for b0 rows (so their
                # config-signature groups with the scenario's other handler
                # rows in analyse_constraint.py); search_obj_metrics records
                # what was actually searched, b0 rows only.
                data['meta'] = dict(
                    suite=suite, pid=pid, scenario=scenario, objtag=this_objtag,
                    obj_metrics=tuple(cfg['obj_metrics']), constr_metric=cfg['constr_metric'],
                    threshold=threshold, mode=cfg['mode'], handler=handler, method=method,
                    seed=seed, pop_size=args.pop_size, n_gen=args.n_gen,
                    n_gen_inner=args.n_gen_inner, config='r3',
                    threshold_set=args.threshold_set, inner_ga=_inner_ga(method, handler),
                    **({'search_obj_metrics': b0_search_obj_metrics} if handler in B0_HANDLERS else {}),
                )
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
          f'constr={cfg["constr_metric"]}, T={threshold:.4f}) -- final feasible-HV summary '
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
    p.add_argument('--scenario', default=None, choices=list(SCENARIOS),
                    help='Which of s1-s4 to run (see module docstring). '
                         'Required unless --migrate.')
    p.add_argument('--suite', default=SUITE, choices=['c10mop', 'in1kmop', 'citysegmop'],
                    help='Benchmark suite (instance must have an entry in THRESHOLDS).')
    p.add_argument('--pid', type=int, default=PID,
                    help='Benchmark problem id within --suite.')
    p.add_argument('--method', nargs='+', default=list(METHODS), choices=METHODS)
    p.add_argument('--handler', default=None, choices=HANDLERS,
                    help='Constraint handler (round 2, see module docstring). '
                         'Default: the scenario default (hard -> h4-cdp, '
                         'soft -> h2-penalty) = round-1 behavior.')
    p.add_argument('--all_handlers', action='store_true',
                    help='Run the scenario\'s full handler row '
                         '(SCENARIO_HANDLERS) instead of a single handler. '
                         'Ignored when --handler is given.')
    p.add_argument('--migrate', action='store_true',
                    help='No runs: copy round-1 flat {method}/seed_N.pkl files '
                         'under --results_root into their scenario-default '
                         'handler subdir (s1/s2 -> h4-cdp, s3/s4 -> h2-penalty). '
                         'Copies only; originals stay in place.')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--pop_size', type=int, default=20)
    p.add_argument('--n_gen', type=int, default=60)
    p.add_argument('--n_doe', type=int, default=None,
                    help='SAMOS2: initial DOE size (default: pop_size)')
    p.add_argument('--n_infill', type=int, default=None,
                    help='SAMOS2: real evaluations per outer generation (default: pop_size)')
    p.add_argument('--n_gen_inner', type=int, default=20,
                    help='SAMOS2: inner NSGA-II generations')
    p.add_argument('--inner_pop_size', type=int, default=None,
                    help='SAMOS2: inner NSGA-II population size (default: pop_size x 10)')
    p.add_argument('--results_root', default=_DEFAULT_RESULTS_ROOT,
                    help='Output root for per-seed pkls. Point at a dedicated '
                         'smoke-test folder when testing -- never write test '
                         'data into results/ (see CLAUDE.md). --threshold_set '
                         'q25 refuses this default (see --threshold_set).')
    p.add_argument('--threshold_set', default='q50', choices=['q50', 'q25'],
                    help='Constraint threshold table: q50 (median, ~50%% '
                         'feasible, default, THRESHOLDS) or q25 (~25%% '
                         'feasible, tighter, THRESHOLDS_Q25). q25 requires '
                         'its own --results_root (e.g. results/constraint_25) '
                         '-- never mixed with q50 data in one tree.')
    p.add_argument('--penalty', type=float, default=1.0,
                    help='ConstraintsAsPenalty weight for soft scenarios (s3/s4). '
                         'Unused for hard scenarios (s1/s2).')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.migrate:
        sys.exit(migrate(args))
    if args.scenario is None:
        p.error('--scenario is required (unless --migrate is given).')
    sys.exit(main(args))
