"""Scenario configurations for scenario_run campaign.

Eight scenarios (S1-S8) at a single 10% feasibility operating point, with tau
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
across the suite -- 0.2x of the objective span on S6/S7 up to 6.0x on S4, whose
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
        space='MoSegNAS', suite='citysegmop', pid=15,
        obj_metrics=('Err.', '#Params'), constr_metric='H1 Lat.',
        direction='min', tau=0.36661331274859266, feasible_fraction=0.10000119865989823,
        ref_point=(1.05, 2.626431190600913), penalty=3.690633954979857
    ),
    'S7': dict(
        space='MoSegNAS', suite='citysegmop', pid=15,
        obj_metrics=('Err.', '#Params'), constr_metric='H2 Lat.',
        direction='min', tau=0.7076894651685864, feasible_fraction=0.10000119865989823,
        ref_point=(1.05, 2.626431190600913), penalty=3.681586082300617
    ),
    'S8': dict(
        space='NB201', suite='c10mop', pid=7,
        obj_metrics=('Err.', 'Eyeriss Lat.'), constr_metric='Eyeriss AI',
        direction='max', tau=0.8965967893600464, feasible_fraction=0.101504,
        ref_point=(1.05, 1.05), penalty=2.3834047043612734
    ),
}

MODES = ('hard', 'soft')

HANDLERS = ['h1-rejection', 'h2-static_penalty', 'h3-adaptive', 'h4-cdp',
            'h5-epsilon', 'h6-DSR', 'b0-as-obj']

METHODS = ['random', 'nsga2', 'samos']

HANDLER_CODE = {
    'h1-rejection':      'h1-rejection',
    'h2-static_penalty': 'h2-penalty',
    'h3-adaptive':       'h3-adaptive-penalty',
    'h4-cdp':            'h4-cdp',
    'h5-epsilon':        'h5-eps',
    'h6-DSR':            'h6-sr',
    'b0-as-obj':         'b0-nsga2',
}

DEFAULT_HANDLER = {'hard': 'h4-cdp', 'soft': 'h2-static_penalty'}

SEARCH_SPACE_LB_OVERRIDE = {'MoSegNAS': {0: 1}}


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
