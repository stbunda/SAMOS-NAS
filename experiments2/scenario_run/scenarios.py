"""Scenario configurations for scenario_run campaign.

Six scenarios (S1-S6) at a single 10% feasibility operating point, with tau
shared by hard and soft modes (differing only in treatment: gate vs archive).
Tau lives in the normalized benchmark-evaluate space (when the benchmark is
not already normalized, affine scaling is applied as ConstrainedEvoXBenchProblem
does).

``tau``, ``ref_point`` and ``penalty`` are all derived from the stage-1 sample
by derive_tau.py (``--check`` re-derives and compares); see TAU.md.

``penalty`` is the static-penalty weight for h2-static_penalty and the starting
weight for h3-adaptive. It is scaled per scenario so that a median-violating
architecture is penalised by one objective range::

    CV        = max(0, sense * (metric - tau) / tau)   # per sampled architecture
    CV_median = median CV over the sample
    span      = max_j (ref_point[j] - min(objective_j))
    penalty   = span / CV_median

G is a *relative* violation (it divides by tau), so a tight tau inflates CV
mechanically. A flat weight of 1.0 therefore varies ~30x in effective pressure
across the suite -- 0.3x of the objective span on S3 up to 8x on S4, whose
tau of 0.028 sits on the NB201 #Params tie plateau -- which collapses h2 toward
unconstrained search at one end and toward h4-cdp at the other. Only h2 needs
this: h5's eps0 is the mean DOE CV so it self-scales, h6's Pf is a probability,
and h1/h4/b0 never read CV magnitude.
"""

from problem.evoxbench.benchmark_meta import metric_index as _metric_index

SCENARIOS = {
    'S1': dict(
        space='NATS', suite='c10mop', pid=4,
        obj_metrics=('Err.', '#Params'), constr_metric='FLOPs',
        direction='min', tau=0.10096685215830803, feasible_fraction=0.100006103515625,
        ref_point=(1.05, 1.05), penalty=0.42952973692843516
    ),
    'S2': dict(
        space='ResNet-50D', suite='in1kmop', pid=3,
        obj_metrics=('Err.', 'FLOPs'), constr_metric='#Params',
        direction='min', tau=0.19399442988274224, feasible_fraction=0.1,
        ref_point=(1.05, 1.05), penalty=1.6487484881442727
    ),
    'S3': dict(
        space='MobileNetV3', suite='in1kmop', pid=9,
        obj_metrics=('Err.', 'FLOPs'), constr_metric='#Params',
        direction='min', tau=0.39204589380971877, feasible_fraction=0.100001,
        ref_point=(1.05, 1.05), penalty=3.7689575570537506
    ),
    'S4': dict(
        space='NB201', suite='c10mop', pid=7,
        obj_metrics=('Err.', 'EdgeGPU Lat.'), constr_metric='#Params',
        direction='min', tau=0.02799552120268345, feasible_fraction=0.139712,
        ref_point=(1.05, 1.05), penalty=0.12096774942757076
    ),
    'S5': dict(
        space='NB201', suite='c10mop', pid=7,
        obj_metrics=('Err.', '#Params'), constr_metric='EdgeGPU Lat.',
        direction='min', tau=0.32976839542388914, feasible_fraction=0.100032,
        ref_point=(1.05, 1.05), penalty=1.1675723922069405
    ),
    'S6': dict(
        space='NB201', suite='c10mop', pid=7,
        obj_metrics=('Err.', 'Eyeriss Lat.'), constr_metric='Eyeriss AI',
        direction='max', tau=0.8965967893600464, feasible_fraction=0.101504,
        ref_point=(1.05, 1.05), penalty=2.3834047043612734
    ),
}

# EvoXBench looks NB101/NATS/NB201 metrics up in a table and produces every
# other space's through trained predictors (MLP accuracy predictors for
# ResNet-50D/MobileNetV3/Transformer, a RankNet for MoSegNAS). The distinction
# matters for analysis, not just provenance: on a tabular space the search
# space is small enough to enumerate, so the true feasible front and its HV are
# exact; on a surrogate space they can only be approximated from a sample, and
# the objective values themselves carry predictor error.
EVAL_KIND_BY_SPACE = {
    'NB101': 'tabular', 'NATS': 'tabular', 'NB201': 'tabular',
    'DARTS': 'surrogate', 'ResNet-50D': 'surrogate', 'Transformer': 'surrogate',
    'MobileNetV3': 'surrogate', 'MoSegNAS': 'surrogate',
}

MODES = ('hard', 'soft')

HANDLERS = ['h1-rejection', 'h2-static_penalty', 'h3-adaptive', 'h4-cdp',
            'h5-epsilon', 'h6-DSR', 'b0-as-obj']

# Slots kept OUT of HANDLERS so the 7-handler row (--all_handlers, the
# handler-pair statistics, the per-handler plots) is unchanged. They are still
# runnable by any method that supports them: ssansga2 runs b1 as one of its two
# fixed slots, and nsga2/samos build it on an explicit --handler (never via
# --all_handlers), where it is the unconstrained-search reference for their
# seven handled cells:
#   b1-unconstrained : the constraint is hidden from the algorithm entirely --
#       it optimizes the two scoring objectives and nothing else, and
#       feasibility is scored post hoc by the callback. Distinct from
#       b0-as-obj, which SHOWS the algorithm the constrained metric (as an
#       extra objective). Under the hard gate the outer problem still masks
#       infeasible F to inf, so 'hidden' means hidden from SELECTION, never
#       'the gate is off' (mode governs the outer problem, not the slot).
EXTRA_HANDLER_SLOTS = ['b1-unconstrained']

ALL_HANDLERS = HANDLERS + EXTRA_HANDLER_SLOTS

METHODS = ['random', 'nsga2', 'ctaea', 'ssansga2', 'samos']

DEFAULT_HANDLER = {'hard': 'h4-cdp', 'soft': 'h2-static_penalty'}

# Methods whose constraint handling is INTRINSIC: they run a FIXED, short list
# of handler slots instead of the 7-handler row. The value is that list, or
# None for "the scenario/mode default".
#   random   : no selection pressure and no sampling strategy for a handler to
#              act on, so only the default slot means anything.
#   ctaea    : the convergence/diversity archive pair and the CV-first
#              restricted-mating tournament ARE the handler -- there is no seam
#              to swap another one into. What they implement is
#              feasibility-first constraint domination, so ctaea is filed under
#              'h4-cdp' in BOTH modes, which also lands it next to nsga2/samos
#              x h4-cdp in the analysis grid (the mode axis still applies: it
#              governs gating vs archiving in the outer problem, not the
#              handler).
#   ssansga2 : surrogate-assisted NSGA-II (pysamoo). Its seam is what the
#              surrogate MODELS, not how survival ranks, so it runs the two
#              slots that seam can take: 'h4-cdp' (the constraint is surrogated
#              alongside the objectives and the inner NSGA-II ranks it by
#              constraint domination) and 'b1-unconstrained' (the constraint is
#              not modelled or seen at all). That pair is the whole point of
#              including it -- it isolates "does handing a surrogate-assisted
#              baseline the constraint help?" at fixed algorithm and budget.
FIXED_HANDLER_METHODS = {
    'random': None,
    'ctaea': ['h4-cdp'],
    'ssansga2': ['h4-cdp', 'b1-unconstrained'],
}

SEARCH_SPACE_LB_OVERRIDE = {'MoSegNAS': {0: 1}}


def eval_kind(sid):
    """'tabular' or 'surrogate' -- how this scenario's benchmark produces its
    metrics (see EVAL_KIND_BY_SPACE)."""
    return EVAL_KIND_BY_SPACE[SCENARIOS[sid]['space']]


def fixed_handlers(method, mode):
    """The handler slot(s) ``method`` runs under ``mode`` as a list, or None
    when it runs the full HANDLERS row (nsga2, samos). See
    FIXED_HANDLER_METHODS."""
    if method not in FIXED_HANDLER_METHODS:
        return None
    slots = FIXED_HANDLER_METHODS[method]
    return list(slots) if slots else [DEFAULT_HANDLER[mode]]


def constr_sense(sid):
    """+1 for a 'metric <= tau' ceiling, -1 for a 'metric >= tau' floor
    (used for G = sense * (metric - tau) / tau)."""
    return 1 if SCENARIOS[sid]['direction'] == 'min' else -1


def obj_indices(sid):
    """Tuple of benchmark metric indices for this scenario's objectives."""
    scenario = SCENARIOS[sid]
    return tuple(_metric_index(scenario['suite'], scenario['pid'], name)
                 for name in scenario['obj_metrics'])


def constr_index(sid):
    """Benchmark metric index for this scenario's constrained metric."""
    scenario = SCENARIOS[sid]
    return _metric_index(scenario['suite'], scenario['pid'],
                         scenario['constr_metric'])
