"""experiments/constraint2/analyse_constraint.py --- Post-hoc feasibility-region
metrics for the constraint2 2x2 campaign (no re-runs needed: everything is
recomputed from the saved archives).

Consumes the per-seed pickles written by ``run_constraint.py``:

  {results_root}/{scenario}/{suite}/pid{pid}/{budget}/{method}/{handler}/seed_{N}.pkl

Each is a ``FeasibilityAwareEvoxBenchCallback.data`` dict with keys
``var_pop``, ``obj_pop``, ``var_archive``, ``obj_archive``,
``test_obj_archive``, ``indicators``, ``n_feasible``, ``n_total``, ``time``,
the exact outer-loop counters ``n_evaluated``/``n_feasible_evaluated``, plus
a ``meta`` dict self-describing the run's exact config (instance, scenario,
objectives, constrained metric, threshold, design feasible fraction, mode,
gate, handler, method, seed, budget, inner GA -- see run_constraint.py).
Seven feasibility-region metrics are computed entirely offline -- no runner
changes, no re-runs. Every (suite, pid) instance found under
``results_root`` is analysed; instances without a resolvable threshold, or
with missing/partial data, are skipped with a message rather than failing.

Objectives are {Err., FLOPs} for the whole campaign; the constrained metric
is per (scenario, instance): '#Params' for the cheap scenarios (s1/s3), the
instance's own latency column for the expensive ones (s2/s4). Each run's
config comes from its pkl's own ``meta`` dict (cross-checked against the
campaign constants derived from the path, warning on any mismatch); absent
``meta`` -- which should not happen in this campaign -- the path +
run_constraint.py constants are the fallback. A run's config SIGNATURE
(objectives + constraint metric + mode) decides whether it shares its
reference data / attainment plot with other runs of the same
scenario+instance; with the campaign's single fixed 'c2' config this is one
group per (scenario, instance), and the ``config`` CSV column is always
'c2'. Unlike the deprecated campaign, s4 is a PHYSICAL run set (the mode
axis is physical -- hard gates evaluability, see run_constraint.py), so
there is no view-of-s2 indirection anywhere.

Multi-constraint scenarios (s5-s12): ``constr_metric`` is a TUPLE in both
the pkl meta and the config signature, and the metric layer generalizes as
feasible = ALL constraints satisfied; violation = MAX over the
per-constraint normalized violations; M6/M8 slack = MIN over constraints
(the binding one); M7's true front = feasible-under-all (enumerable
instances: the design cross-check uses the measured JOINT fraction). The
constraint-vs-error boundary plots are single-constraint only for now
(multi configs are skipped there; the appended constraint column of
final_front_c/final_infeas_c holds the max violation instead of a raw
metric for those runs). The headline multi-constraint deliverable is the
b0-vs-CDP win/loss trend across constraint COUNT, read from
constraint_metrics.csv / handler_stats.csv across s1-s4 (1 constraint),
s5-s10 (2) and s11-s12 (3-4).

Metrics
-------
M1  Feasibility-ratio trajectory: ``n_feasible / n_total`` per generation,
    straight from the saved arrays.
M2  Evals-to-first-feasible: 1-based index, into the FINAL generation's
    ``var_archive`` (the only cumulative, insertion-ordered record of
    evaluated architectures the pkl carries -- see "Evaluation-order caveat"
    below), of the first finite architecture whose re-evaluated constraint
    metric is <= threshold. ``inf`` (censored) if none.
M3  Infeasible-evaluation waste: cumulative fraction of the (finite) archive
    that is infeasible (``= 1 - M1``), plus a per-generation fraction
    reconstructed from consecutive differences of the ``n_total`` /
    ``n_feasible`` trajectories (``NaN`` where the archive did not grow that
    generation -- can't be decomposed from counts alone). CSV-only
    ``exact_waste`` is a separate, exact companion, not part of M3: M3 is
    archive-based and understates true waste (dominated points are dropped
    from ``var_archive``), while ``exact_waste = 1 - n_feasible_evaluated[-1]
    / n_evaluated[-1]`` counts every architecture the outer loop ever
    evaluated, from the run callback's own cumulative counters (``NaN`` when
    those keys are absent -- pkls written before this counter existed -- or
    ``n_evaluated[-1] == 0``).
M4  Feasible non-dominated front size per generation. ``test_obj_archive`` IS
    this front already (``FeasibilityAwareEvoxBenchCallback.notify()``
    computes ND-among-feasible in the run's own true-eval objective space
    every generation) -- no re-evaluation needed, just ``len(...)`` per
    generation.
M5  Mean constraint violation of infeasible archive points per generation:
    ``(metric - T) / T`` averaged over infeasible, finite archive members;
    ``NaN`` if none. Needs a true-eval re-evaluation of every generation's
    ``var_archive`` (the constraint column is not itself saved).
M6  Boundary slack of the FINAL feasible ND front: ``(T - metric) / T``
    over the front's members; reports min and median.
M7  Ground-truth reference (ENUMERABLE instances only, guarded on
    ``prod(ub - lb + 1) <= --enum_limit`` [default 200_000]; c10mop/pid4 =
    8**5 = 32768 and c10mop/pid5 = 5**6 = 15625 qualify, everything else does
    not): exhaustively enumerate the space (``itertools.product`` over
    ``benchmark.search_space.lb/ub`` -- bounds read from the benchmark, not
    hardcoded), evaluate once, build the TRUE feasible Pareto front per
    (scenario, config) in that config's own 2-objective true-eval space, then
    report each run's final feasible front's exact IGD+ (pymoo ``IGDPlus``)
    against it and HV-ratio = HV(run front) / HV(true front) using a
    ref-point of ``np.ones(2) * 1.05`` (every current scenario config has
    exactly 2 objectives -- same convention ``FeasibilityAwareEvoxBenchCallback``
    uses). Enumerated fresh every invocation -- no cache file. For
    NON-enumerable instances M7 IGD+/HV-ratio are reported as NaN, marked
    'n/a (non-enumerable)' in the summary -- a random sample is deliberately
    NOT passed off as a true front. M1-M6 and the raw feasible HV stay fully
    available.
M8  Best feasible error: the minimum true error over the run's final
    FEASIBLE solutions (``m8_best_err``), plus the boundary slack of that
    best point (``m8_best_err_slack``, ``(T - metric)/T``) -- how far the
    handler pushes accuracy right up against the constraint limit, as
    opposed to HV's whole-front view. Lower error is better (the
    significance tests handle the direction); ``m8_err_true`` (enumerable
    instances only) is the ground-truth minimum feasible error for
    reference.

Soft-HV
-------
Violation-graded hypervolume of a run's FINAL archive (every finite point of
the final-generation ``var_archive``, not just the feasible ND front),
re-evaluated in the config's own true-eval 2-objective space with the same
convention as everything else in this module: each point's normalized
constraint violation ``max(0, (metric - T) / T)`` (0 for feasible points) is
added to BOTH objective columns, the non-dominated front of this shifted set
is taken, and its HV is computed against the same ``np.ones(2) * 1.05`` ref
point (points shifted outside the ref box contribute nothing, as HV does
naturally). Reported as the ``soft_hv`` CSV column for every run regardless
of scenario mode -- it coincides with the ordinary feasible HV only in the
all-feasible limit, so comparing it across handlers is the soft-scenario
story.

Significance testing
---------------------
Per (scenario, instance, config signature, method): pairwise two-sided
Wilcoxon SIGNED-RANK (``scipy.stats.wilcoxon``, paired by seed)
between handler pairs, on final feasible HV (``m7_hv_run``), HV-ratio
(enumerable instances only), and ``soft_hv``. Pairs are grouped into
comparison FAMILIES so the priority comparison holds the inner GA fixed
(see _stat_families): 'nsga2' (the shared handler row + b0-nsga2, all
NSGA-II inners), 'sms' (b0-sms vs h4-cdp-sms, the formulation comparison
under SMS-EMOA), and 'ga-effect' (the separate inner-GA study: h4-cdp vs
h4-cdp-sms and b0-nsga2 vs b0-sms -- same formulation, different GA).
Cross-GA cross-formulation pairs are not tested (two factors at once,
each covered cleanly by its own family). Note the b0-as-obj result
directories are labeled 'b0-sms' throughout (HANDLER_LABEL). Runs sharing
a seed share their RNG streams and (for the same method) their evaluated
DOE (run_constraint.py seeds np.random and minimize identically per seed),
so seeds are matched blocks: the paired test removes between-seed variance
(e.g. a lucky DOE lifting every handler that seed) that an unpaired
rank-sum would pool into noise. Only seeds present with finite values
on BOTH sides pair up; a pair with fewer than 5 common seeds is skipped
(noted, excluded from that family's Holm correction) rather than tested.
All-zero difference vectors (byte-identical runs, e.g. replicated random
rows) are p = 1.0 ties by definition (scipy's wilcoxon rejects them). Raw
p-values are Holm-corrected within each (scenario, instance, config
signature, method, metric, family) unit. A comparison counts as a win for
the handler with the better (higher) median over the COMMON seeds when its
Holm-adjusted p < 0.05, otherwise a tie; only handler pairs sharing the
same method and config signature are ever compared (never across objective
spaces).

Attainment plots
----------------
Per (scenario x instance), ``{scenario}_{suite}_pid{pid}_attainment.png``
in ``{output_dir}/plots``: one SUBPLOT PER HANDLER (every handler present,
not just the scenario default), each showing the attainable objective cloud
in the true-eval plane, split feasible/infeasible (from the exhaustive
enumeration where enumerable, else a fixed-seed [np.random.seed(0)] 10k
uniform random sample, labeled 'sampled (10k)' since it is an
approximation); the true feasible Pareto front (enumerable only) as a
staircase PLUS its member points as markers, so constraint-induced gaps in
the front are visible rather than bridged by the staircase; each method's
empirical attainment surface of FEASIBLE solutions as a minimization
staircase -- with 1 seed the final feasible ND front, with n > 1 seeds the
median (ceil(n/2)-out-of-n) attainment surface with a shaded best/worst
band -- and random's median attainment as a grey reference in every
subplot. ``--self_test`` runs a unit-style check of the multi-seed
attainment computation on synthetic 2-D points and exits.

Evaluation-order caveat (M2)
-----------------------------
``var_archive`` is the cumulative NON-DOMINATED (by search objectives only --
feasibility does not gate membership, see callbacks.py) archive, not a literal
log of every evaluated architecture: a point later dominated in objective
space is dropped and never reappears. Its final-generation snapshot is
insertion-ordered (new points are appended; removals never reorder survivors)
and is the only cross-generation, evaluation-order-correlated record the pkl
contains, so M2 uses it as instructed by the analysis brief. This can
overstate evals-to-first-feasible if an early feasible point was later
objective-dominated out of the archive before being counted here; accepted
as a known limitation of post-hoc analysis on saved archives.

Re-evaluation conventions (must match callbacks.py / constrained_problem.py
EXACTLY so numbers are comparable): round X to int, ``benchmark.evaluate(X,
true_eval=True)``, then ``benchmark.normalize()`` ONLY when
``not benchmark.normalized_objectives`` (natively normalized for
c10mop/pid2,3,4; ``normalize()`` applied exactly once for every other
instance, mirroring the callback/runner convention). This convention is
about the METRIC SPACE only and is instance-, not objective-set-, dependent,
so it is unaffected by which two columns a config picks as objectives.
Non-finite rows dropped (``finite_mask``, matching the callback exactly).

Output
------
  {output_dir}/constraint_metrics.csv    one row per (scenario, suite, pid, method, handler, config, seed)
                                          (includes ``soft_hv``, see above)
  {output_dir}/constraint_trajectories.json   per-generation M1/M3/M4/M5 (+ raw
                                               feasible-HV and M7 HV-ratio trajectories)
  {output_dir}/handler_stats.csv         one row per (scenario, instance, config
                                          signature, method, metric, handler pair)
                                          Wilcoxon signed-rank comparison (raw p,
                                          Holm p, medians, significance flag)
  {output_dir}/plot_data.pkl             everything the plot/table layer needs
                                          (clouds, refs, per-run fronts), so
                                          --plots_only regenerates figures and
                                          tables in seconds
  {output_dir}/plots/{scenario}_{feasibility_ratio,feasible_hv,hv_ratio,waste}.png
          per-(scenario, metric) grids: instance subplots, colour = handler,
          linestyle = method, one legend cell; titles carry the scenario's
          mode + constraint type and each instance's constrained metric +
          threshold (hv_ratio: enumerable instances only; waste: cumulative
          exact infeasible-evaluation fraction from the run counters)
  {output_dir}/plots/{scenario}_waste_bars.png   final exact waste as grouped
          bars (instance subplots, x = handler, bars = samos/samos-cheap
          mean +- std, random as a dotted reference line)
  {output_dir}/plots/{scenario}_{suite}_pid{pid}_attainment.png   (per-handler
          subplot grids, see 'Attainment plots')
  {output_dir}/plots/constr_vs_err/{scenario}_{suite}_pid{pid}_constr_vs_err.png
          boundary diagnostics: per-handler subplots in the (error,
          constrained metric) plane -- neutral context cloud + threshold
          line, per-method feasible front members (dots) and infeasible
          final-archive members (x markers; hard mode: only true-eval limit
          crossers -- gate-discarded evaluations are not recorded) (see
          plot_constr_attainment_grid; NOT a Pareto view)
  {output_dir}/plots/constr_vs_err_modes/{cheap|expensive}_{suite}_pid{pid}_hard_vs_soft.png
          hard-vs-soft comparison of the same diagnostic: rows = handlers,
          left column = hard scenario, right = soft (identical instance,
          metric, threshold; one figure per constraint type x instance)
  {output_dir}/tables/constraint2_tables.tex   booktabs LaTeX: per-scenario
          feasible-HV (mean +- std) tables, the same-GA W/L/T significance
          summary, and the GA-effect study table
  stdout: plain-text summary table (mean over seeds where n_seeds > 1), one
          block per (scenario, instance, config) actually present, followed
          by a per (scenario, instance) win/tie/loss matrix block from the
          significance tests above

Examples
--------
  python experiments/constraint2/analyse_constraint.py
  python experiments/constraint2/analyse_constraint.py --self_test
  python experiments/constraint2/analyse_constraint.py --scenarios s1 s2 --seeds 0 1
"""

import argparse
import glob
import itertools
import json
import math
import os
import pickle
import sys

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, _THIS_DIR)     # for run_constraint (also fixes its own _common path)
sys.path.insert(0, _REPO_ROOT)

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import wilcoxon

from problem.evoxbench.benchmark_meta import get_obj_names, metric_index
from problem.evoxbench.utils import get_benchmark
# Reuse the runner's exact scenario/threshold constants -- no re-derivation,
# so numbers are guaranteed comparable.
from run_constraint import (METHODS, SCENARIOS, THRESHOLDS, OBJ_METRICS,
                            DESIGN_FEASIBLE_FRACTION,
                            JOINT_DESIGN_FEASIBLE_FRACTION, constr_metrics_for)

# Same ref-point convention as FeasibilityAwareEvoxBenchCallback: every
# scenario config picks exactly 2 objectives.
REF_POINT = np.ones(2) * 1.05


def _path_config(scenario, suite, pid):
    """Campaign-constant config (obj_metrics/constr_metric/mode) for one
    (scenario, instance), from run_constraint.py's own definitions -- the
    cross-check / fallback for a pkl's self-describing ``meta``. None when
    the scenario or the instance's constrained metric doesn't resolve.
    ``constr_metric`` is a scalar name for the single-constraint scenarios
    (matching their pkls' meta exactly) and a tuple for s5-s12."""
    cfg = SCENARIOS.get(scenario)
    if cfg is None:
        return None
    try:
        constr_metrics = constr_metrics_for(scenario, suite, pid)
    except KeyError:
        return None
    constr_metric = constr_metrics if len(constr_metrics) > 1 else constr_metrics[0]
    return dict(obj_metrics=tuple(OBJ_METRICS), constr_metric=constr_metric,
                mode=cfg['mode'])


def _constraint_lists(cfg):
    """(metric names tuple, thresholds tuple) from a config whose
    ``constr_metric``/``threshold`` may be scalar (single constraint) or
    tuple-valued (multi)."""
    cm = cfg['constr_metric']
    names = tuple(cm) if isinstance(cm, (list, tuple)) else (cm,)
    thr = cfg['threshold']
    thrs = (tuple(float(t) for t in thr) if isinstance(thr, (list, tuple))
            else (float(thr),) * len(names))
    assert len(thrs) == len(names), (names, thrs)
    return names, thrs


def _max_violation(F_rows, constr_indices, thresholds):
    """Per-row MAX over the per-constraint normalized violations
    ``(metric_j - T_j) / T_j`` -- the campaign's multi-constraint violation
    convention. Feasible <=> value <= 0; ``-value`` is the binding-constraint
    slack (the MIN slack over constraints). Reduces to the single-constraint
    violation when one constraint is given."""
    F_rows = np.asarray(F_rows, dtype=float)
    return np.max(np.column_stack(
        [(F_rows[:, ci] - t) / t for ci, t in zip(constr_indices, thresholds)]), axis=1)


def _config_signature(cfg):
    """Identity used to decide whether two configs are the SAME experiment
    (their rows/plots merge) or DIFFERENT ones (kept apart)."""
    return (tuple(cfg['obj_metrics']), cfg['constr_metric'], cfg['mode'])


def _run_config(scenario, suite, pid, meta):
    """Full config for one discovered run -- obj_metrics/constr_metric/mode/
    threshold -- preferring its own ``meta`` dict (every constraint2 pkl is
    self-describing, so a run's numbers never drift from a later change to
    the campaign constants) over the path-derived lookup, with a mismatch
    warning if both are available and disagree. Falls back to the path +
    run_constraint.py constants when ``meta`` is absent (should not happen
    in this campaign) or its constraint metric has no threshold for this
    instance. None if nothing usable resolves."""
    path_cfg = _path_config(scenario, suite, pid)
    if meta is not None:
        cm = meta['constr_metric']
        cm = tuple(cm) if isinstance(cm, (list, tuple)) else cm
        thr = meta['threshold']
        thr = (tuple(float(t) for t in thr) if isinstance(thr, (list, tuple))
               else float(thr))
        meta_cfg = dict(obj_metrics=tuple(meta['obj_metrics']), constr_metric=cm,
                         mode=meta['mode'], threshold=thr)
        if path_cfg is not None and _config_signature(meta_cfg) != _config_signature(path_cfg):
            print(f"  [analyse] WARN: {scenario}/{suite}/pid{pid}: pkl meta config "
                  f"{_config_signature(meta_cfg)} disagrees with the path-derived config "
                  f"{_config_signature(path_cfg)}; trusting meta (the pkl's own record).")
        return meta_cfg
    if path_cfg is None:
        return None
    inst_thresholds = THRESHOLDS.get((suite, pid), {})
    cm = path_cfg['constr_metric']
    names = cm if isinstance(cm, tuple) else (cm,)
    if any(m not in inst_thresholds for m in names):
        return None
    thr = (tuple(inst_thresholds[m] for m in names) if isinstance(cm, tuple)
           else inst_thresholds[cm])
    return dict(path_cfg, threshold=thr)


# Reporting label map: the 'b0-as-obj' result DIRECTORIES (never renamed on
# disk) are labeled 'b0-sms' in every output of this module, so each handler
# name that involves a non-default inner GA carries that GA explicitly
# ('b0-sms'/'b0-nsga2', 'h4-cdp'/'h4-cdp-sms') and same-GA rows compare at a
# glance. Applied once at discovery (discover_runs); everything downstream
# (CSV, stats, trajectories, plots, stdout) sees only the display name.
HANDLER_LABEL = {'b0-as-obj': 'b0-sms'}

# Inner GA per display handler name, the fallback when a pkl's meta lacks
# ``inner_ga`` (should not happen in this campaign): everything runs NSGA-II
# except the two SMS-EMOA rows.
_SMS_HANDLERS = ('b0-sms', 'h4-cdp-sms')

# Same-formulation groups across the inner-GA axis: the {formulation} x
# {inner GA} factorial pairs for the GA-effect study.
FORMULATION = {'b0-sms': 'b0', 'b0-nsga2': 'b0',
               'h4-cdp': 'h4-cdp', 'h4-cdp-sms': 'h4-cdp'}

# Display order for stdout summary rows: the same-GA (NSGA-II) handler row
# first -- the priority comparison -- then the two SMS-EMOA rows.
HANDLER_ORDER = ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp',
                 'h5-eps', 'h6-sr', 'h1-cdp-reject', 'b0-nsga2',
                 'h4-cdp-sms', 'b0-sms']


def _handler_sort_key(handler):
    """HANDLER_ORDER position (unknown names go last, alphabetically)."""
    try:
        return (HANDLER_ORDER.index(handler), handler)
    except ValueError:
        return (len(HANDLER_ORDER), handler)


def _method_ga(method, handler, inner_ga=None):
    """Method+inner-GA label: 'random' for method random; otherwise
    f'{method}-sms' when the inner GA is SMS-EMOA (handler in
    _SMS_HANDLERS), else f'{method}-nsga2' (NSGA-II, the samos default;
    handler 'b0-nsga2' is also NSGA-II). *inner_ga* ('sms'/'nsga2'), when a
    pkl's own meta carries it, overrides the handler rule."""
    if method == 'random':
        return 'random'
    if inner_ga in ('sms', 'nsga2'):
        return f'{method}-{inner_ga}'
    return f'{method}-sms' if handler in _SMS_HANDLERS else f'{method}-nsga2'


# Okabe-Ito hues (colorblind-safe), fixed assignment per method -- never cycled.
METHOD_COLOURS = {
    'random':      '#0072B2',
    'samos':       '#D55E00',
    'samos-cheap': '#009E73',
}
CLOUD_FEASIBLE_COLOUR   = '#b8d4ea'   # light cool
CLOUD_INFEASIBLE_COLOUR = '#f4c7b8'   # light warm

# Trajectory plots: color = HANDLER (the comparison axis), linestyle =
# method, so every (method, handler) line is visually unique. Fixed
# assignment (tab10), never cycled.
HANDLER_COLOURS = {
    'h1-rejection':        '#1f77b4',
    'h2-penalty':          '#ff7f0e',
    'h3-adaptive-penalty': '#2ca02c',
    'h4-cdp':              '#d62728',
    'h5-eps':              '#9467bd',
    'h6-sr':               '#8c564b',
    'h1-cdp-reject':       '#e377c2',
    'b0-nsga2':            '#7f7f7f',
    'h4-cdp-sms':          '#bcbd22',
    'b0-sms':              '#17becf',
}
METHOD_LINESTYLE = {'samos': '-', 'samos-cheap': '--', 'random': ':'}

# Reader-facing scenario names: figures and LaTeX tables say hard/soft x
# constraint-type words instead of sN codes (filenames and the results tree
# keep the sN codes -- those mirror physical directories).
_CONSTR_WORD = {'cheap': 'cheap', 'expensive': 'expensive',
                'cc': 'cheap+cheap', 'ce': 'cheap+expensive',
                'ee': 'expensive+expensive', 'all': 'all-constraints'}
SCENARIO_LABEL = {sc: f"{cfg['mode']}/{_CONSTR_WORD[cfg['constr']]}"
                  for sc, cfg in SCENARIOS.items()}
SCENARIO_SLUG  = {sc: f"{cfg['mode']}-{_CONSTR_WORD[cfg['constr']].replace('+', '-')}"
                  for sc, cfg in SCENARIOS.items()}


def _scen_long(scenario):
    cfg = SCENARIOS.get(scenario)
    if cfg is None:
        return scenario
    word = _CONSTR_WORD.get(cfg['constr'], cfg['constr'])
    plural = 's' if cfg['constr'] in ('cc', 'ce', 'ee', 'all') else ''
    return f"{cfg['mode']} mode, {word} constraint{plural}"


# Short instance names for figure titles / LaTeX table headers.
INSTANCE_SHORT = {
    ('c10mop', 4):      'NATS',
    ('c10mop', 5):      'NB201',
    ('in1kmop', 9):     'MNV3',
    ('citysegmop', 5):  'MoSeg-H1',
    ('citysegmop', 10): 'MoSeg-H2',
}


def _inst_short(inst_key):
    suite, pid_s = inst_key.split('/pid')
    return INSTANCE_SHORT.get((suite, int(pid_s)), inst_key)

CLOUD_SAMPLE_N    = 10_000
CLOUD_SAMPLE_SEED = 0

# ─── significance-testing constants ────────────────────────────────────────
STATS_ALPHA     = 0.05
STATS_MIN_SEEDS = 5    # per side; families below this are skipped, not tested
STATS_METRICS   = ('m7_hv_run', 'm7_hv_ratio', 'soft_hv', 'm8_best_err')
STATS_METRIC_LABEL = {'m7_hv_run': 'feasible HV', 'm7_hv_ratio': 'HV-ratio',
                      'soft_hv': 'soft-HV', 'm8_best_err': 'best feasible Err'}
# +1: higher median is better; -1: lower is better (error).
STATS_METRIC_DIRECTION = {'m7_hv_run': 1, 'm7_hv_ratio': 1, 'soft_hv': 1, 'm8_best_err': -1}


# ─── re-evaluation helpers ─────────────────────────────────────────────────

def _norm_once(benchmark, F):
    """The one shared normalization convention (callbacks.py / runner /
    constrained_problem.py): benchmark.normalize() ONLY when the benchmark
    does not already return normalized objectives. NEVER stacked twice."""
    if not benchmark.normalized_objectives:
        F = benchmark.normalize(F)
    return np.where(np.isfinite(F), F, np.nan)


def _full_reeval(benchmark, X, cache: dict):
    """True-eval every row of *X* (n, n_var), via *cache* keyed by rounded-int
    row (shared across generations/runs/scenarios of ONE instance -- a hit is
    exact regardless of which run asked for it). Mirrors
    FeasibilityAwareEvoxBenchCallback.notify()'s conventions exactly (see
    module docstring). Returns (F_full, finite_mask), both length len(X).
    """
    if len(X) == 0:
        return np.empty((0, benchmark.evaluator.n_objs)), np.array([], dtype=bool)
    X_int = np.round(np.asarray(X, dtype=float)).astype(int)
    keys = [tuple(row) for row in X_int]
    missing = sorted(set(i for i, k in enumerate(keys) if k not in cache))
    if missing:
        F_new = _norm_once(benchmark, benchmark.evaluate(X_int[missing], true_eval=True))
        for i, row in zip(missing, F_new):
            cache[keys[i]] = np.asarray(row, dtype=float)
    F_full = np.array([cache[k] for k in keys], dtype=float)
    finite_mask = np.isfinite(F_full).all(axis=1)
    return F_full, finite_mask


def space_size(benchmark) -> int:
    """Cardinality of the integer search space, from the benchmark's own
    bounds (never hardcoded). Python ints -- no overflow for huge spaces."""
    lb = [int(v) for v in benchmark.search_space.lb]
    ub = [int(v) for v in benchmark.search_space.ub]
    return math.prod(u - l + 1 for l, u in zip(lb, ub))


def enumerate_full_space(benchmark):
    """Exhaustive true-eval of the full space (only call when enumerable)."""
    lb = np.asarray(benchmark.search_space.lb, dtype=int)
    ub = np.asarray(benchmark.search_space.ub, dtype=int)
    ranges = [range(int(l), int(u) + 1) for l, u in zip(lb, ub)]
    X_all = np.array(list(itertools.product(*ranges)), dtype=int)
    F_all = _norm_once(benchmark, benchmark.evaluate(X_all, true_eval=True))
    finite_mask = np.isfinite(F_all).all(axis=1)
    return X_all[finite_mask], F_all[finite_mask]


def sample_cloud(benchmark, n=CLOUD_SAMPLE_N, seed=CLOUD_SAMPLE_SEED):
    """Fixed-seed uniform random sample of the space, true-evaled with the
    shared convention -- an APPROXIMATE attainable cloud for non-enumerable
    instances (plot background only; never used as an M7 'true front').
    Sampling convention matches THRESHOLDS.md / EvoxBenchSampler:
    np.random.seed(seed) then per-column randint(lb, ub + 1)."""
    lb = [int(v) for v in benchmark.search_space.lb]
    ub = [int(v) for v in benchmark.search_space.ub]
    np.random.seed(seed)
    X = np.column_stack([np.random.randint(lo, hi + 1, size=n)
                         for lo, hi in zip(lb, ub)]).astype(int)
    F = _norm_once(benchmark, benchmark.evaluate(X, true_eval=True))
    finite_mask = np.isfinite(F).all(axis=1)
    return X[finite_mask], F[finite_mask]


def true_feasible_front(F_all, obj_indices, constr_indices, thresholds):
    """(feasible_fraction, feasible-under-ALL ND front in {obj_indices}
    space, same front with a constraint column appended -- (n, 3), for the
    boundary-diagnostic constraint-vs-error plots). The appended column is
    the raw constrained metric for a single constraint (legacy plot
    convention) and the per-point MAX normalized violation for multiple
    constraints (negative = binding-constraint slack)."""
    vmax = _max_violation(F_all, constr_indices, thresholds)
    feasible_mask = vmax <= 0
    frac = float(feasible_mask.mean())
    F_feas_full = F_all[feasible_mask]
    F_feas = F_feas_full[:, obj_indices]
    if len(F_feas) == 0:
        return frac, np.empty((0, len(obj_indices))), np.empty((0, len(obj_indices) + 1))
    nd_idx = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)
    c_col = (F_feas_full[nd_idx, constr_indices[0]] if len(constr_indices) == 1
             else vmax[feasible_mask][nd_idx])
    front_c = np.column_stack([F_feas[nd_idx], c_col])
    return frac, F_feas[nd_idx], front_c


def _hv(F, ref_point):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or len(F) == 0:
        return 0.0
    return float(HV(ref_point=ref_point)(F))


# ─── empirical attainment surfaces (2-D minimization) ──────────────────────

def attainment_surfaces(fronts, ks):
    """k-out-of-n empirical attainment surfaces for 2-D minimization fronts.

    Parameters
    ----------
    fronts : list of (m_i, 2) arrays (one per seed/run).
    ks : iterable of int in [1, n] -- k=1 is the best (attained by >= 1 run),
         k=n the worst (attained by all), k=ceil(n/2) the median surface.

    Returns
    -------
    (xs, {k: ys}) where xs is the sorted union of all fronts' x-coordinates
    and ys[j] is the y-level attained (weakly dominated) by at least k of the
    n fronts at x = xs[j]: the k-th smallest of the per-front minimal y over
    points with x' <= xs[j]. np.inf where fewer than k fronts attain any
    point at that x. Drawn as a staircase with ``step(..., where='post')``.
    """
    fronts = [np.asarray(f, dtype=float) for f in fronts]
    assert fronts and all(f.ndim == 2 and f.shape[1] == 2 and len(f) > 0 for f in fronts), \
        'attainment_surfaces needs >= 1 non-empty (m, 2) front'
    n = len(fronts)
    xs = np.unique(np.concatenate([f[:, 0] for f in fronts]))
    Y = np.full((n, xs.size), np.inf)
    for i, f in enumerate(fronts):
        order = np.argsort(f[:, 0], kind='stable')
        fx, fy = f[order, 0], f[order, 1]
        cummin = np.minimum.accumulate(fy)
        idx = np.searchsorted(fx, xs, side='right') - 1
        valid = idx >= 0
        Y[i, valid] = cummin[idx[valid]]
    Ys = np.sort(Y, axis=0)
    return xs, {int(k): Ys[int(k) - 1, :] for k in ks}


def _self_test_attainment():
    """Unit-style check of the multi-seed attainment path on synthetic 2-D
    points (real multi-seed data does not exist yet)."""
    inf = np.inf
    # Three single-point fronts on an anti-diagonal: every k-level is known.
    fronts = [np.array([[0., 3.]]), np.array([[1., 2.]]), np.array([[2., 1.]])]
    xs, surf = attainment_surfaces(fronts, ks=[1, 2, 3])
    assert np.allclose(xs, [0., 1., 2.]), xs
    assert np.allclose(surf[1], [3., 2., 1.]), surf[1]            # best
    assert np.array_equal(surf[2], [inf, 3., 2.]), surf[2]        # median
    assert np.array_equal(surf[3], [inf, inf, 3.]), surf[3]       # worst
    # Single front reduces to its own staircase (cumulative min over x),
    # including a dominated interior point.
    f = np.array([[0., 2.], [0.5, 3.], [1., 1.]])
    xs1, s1 = attainment_surfaces([f], ks=[1])
    assert np.allclose(xs1, [0., 0.5, 1.]) and np.allclose(s1[1], [2., 2., 1.]), (xs1, s1)
    # Median of identical fronts equals the front's staircase.
    xs2, s2 = attainment_surfaces([f, f.copy(), f.copy()], ks=[2])
    assert np.allclose(xs2, [0., 0.5, 1.]) and np.allclose(s2[2], [2., 2., 1.]), (xs2, s2)
    # Two overlapping multi-point fronts: k=2 (worst) = pointwise max.
    fa = np.array([[0., 2.], [2., 0.]])
    fb = np.array([[1., 1.]])
    xs3, s3 = attainment_surfaces([fa, fb], ks=[1, 2])
    assert np.allclose(xs3, [0., 1., 2.]), xs3
    assert np.allclose(s3[1], [2., 1., 0.]), s3[1]
    assert np.array_equal(s3[2], [inf, 2., 1.]), s3[2]
    print('[self-test] attainment_surfaces: all assertions passed.')


# ─── per-run metrics ────────────────────────────────────────────────────────

def analyse_run(data: dict, obj_indices, constr_indices, thresholds, benchmark, cache: dict):
    """Per-run metrics. Multi-constraint conventions: feasible = ALL
    constraints satisfied; violation = MAX over per-constraint normalized
    violations (matches the pymoo-CV direction of aggregation used by the
    campaign); slack (M6/M8) = MIN over constraints (the binding one). The
    constraint column appended to final_front_c/final_infeas_c is the raw
    metric for a single constraint, the max violation otherwise (see
    true_feasible_front)."""
    single = len(constr_indices) == 1
    var_archive       = data.get('var_archive', [])
    test_obj_archive  = data.get('test_obj_archive', [])
    n_feasible        = np.asarray(data.get('n_feasible', []), dtype=float)
    n_total           = np.asarray(data.get('n_total', []), dtype=float)
    n_gen             = len(var_archive)

    # M1: straight from saved arrays.
    m1_ratio = np.where(n_total > 0, n_feasible / np.where(n_total > 0, n_total, 1), np.nan)

    # M3: cumulative fraction (= 1 - M1) + per-generation reconstruction from
    # consecutive differences of the n_total / n_feasible trajectories.
    m3_cumulative = np.where(n_total > 0, (n_total - n_feasible) / np.where(n_total > 0, n_total, 1), np.nan)
    delta_total    = np.diff(n_total, prepend=0.0)
    delta_feasible = np.diff(n_feasible, prepend=0.0)
    m3_per_gen = np.where(
        delta_total > 0,
        (delta_total - delta_feasible) / np.where(delta_total > 0, delta_total, 1),
        np.nan)

    # M4: the saved feasible ND front IS this metric already.
    m4_front_size = [len(g) for g in test_obj_archive]

    # M5 (+ per-generation re-evaluations reused below for M2/M6).
    m5_mean_violation = []
    full_evals_per_gen = []
    for g in range(n_gen):
        F_full, finite_mask = _full_reeval(benchmark, var_archive[g], cache)
        full_evals_per_gen.append((F_full, finite_mask))
        F_fin = F_full[finite_mask]
        if len(F_fin) == 0:
            m5_mean_violation.append(float('nan'))
            continue
        vmax = _max_violation(F_fin, constr_indices, thresholds)
        infeasible = vmax > 0
        if not infeasible.any():
            m5_mean_violation.append(float('nan'))
        else:
            m5_mean_violation.append(float(np.mean(vmax[infeasible])))

    # M2: 1-based index into the FINAL var_archive (evaluation-order proxy,
    # see module docstring), among finite entries only (matches n_total's own
    # finite-row convention).
    if n_gen > 0:
        F_last, finite_last = full_evals_per_gen[-1]
        F_last_fin = F_last[finite_last]
        vmax_last = (_max_violation(F_last_fin, constr_indices, thresholds)
                     if len(F_last_fin) else np.zeros(0))
        feasible_seq = vmax_last <= 0
        if feasible_seq.any():
            m2_evals_to_first_feasible = int(np.argmax(feasible_seq)) + 1
            m2_censored = False
        else:
            m2_evals_to_first_feasible = float('inf')
            m2_censored = True
        m2_n_evaluated = int(len(F_last_fin))
    else:
        F_last_fin = np.empty((0, benchmark.evaluator.n_objs))
        vmax_last = np.zeros(0)
        m2_evals_to_first_feasible = float('inf')
        m2_censored = True
        m2_n_evaluated = 0

    # M6: boundary slack over the FINAL feasible ND front (slack = MIN over
    # constraints = -max violation; the binding constraint's slack).
    final_front = np.asarray(test_obj_archive[-1]) if n_gen > 0 and len(test_obj_archive[-1]) > 0 \
        else np.empty((0, len(obj_indices)))
    feasible_mask_last = vmax_last <= 0 if len(F_last_fin) else np.zeros(0, dtype=bool)
    # Infeasible members of the final archive, with the constraint column
    # appended ((n, 3)) -- soft mode: the ND-surviving infeasible samples the
    # run kept; hard mode: members whose TRUE evaluation crosses the limit
    # even though the gate admitted them (fidelity gap at the boundary). The
    # gate-discarded evaluations themselves are not recorded in the pkls.
    if len(F_last_fin) and (~feasible_mask_last).any():
        F_inf = F_last_fin[~feasible_mask_last]
        c_inf = (F_inf[:, constr_indices[0]] if single
                 else vmax_last[~feasible_mask_last])
        final_infeas_c = np.column_stack([F_inf[:, obj_indices], c_inf])
    else:
        final_infeas_c = np.empty((0, len(obj_indices) + 1))
    if len(final_front) > 0 and feasible_mask_last.any():
        F_feas_last = F_last_fin[feasible_mask_last]
        vmax_feas_last = vmax_last[feasible_mask_last]
        obj_feas_last = F_feas_last[:, obj_indices]
        nd_idx = NonDominatedSorting().do(obj_feas_last, only_non_dominated_front=True)
        slack = -vmax_feas_last[nd_idx]
        m6_min_slack, m6_median_slack = float(np.min(slack)), float(np.median(slack))
        # Front members with their re-evaluated constraint column appended
        # ((n, 3)) -- the boundary-diagnostic constraint-vs-error view.
        c_front = (F_feas_last[nd_idx, constr_indices[0]] if single
                   else vmax_feas_last[nd_idx])
        final_front_c = np.column_stack([obj_feas_last[nd_idx], c_front])
        # M8: best (minimum) true error over the final FEASIBLE solutions,
        # plus the boundary slack of that best point -- how well the run
        # exploits right up against the constraint limit (the error-optimal
        # feasible architecture typically sits near the boundary).
        best_i = int(np.argmin(F_feas_last[:, obj_indices[0]]))
        m8_best_err = float(F_feas_last[best_i, obj_indices[0]])
        m8_best_err_slack = float(-vmax_feas_last[best_i])
    else:
        # No re-evaluated feasible point in the final archive (a run may end
        # all-infeasible, or the callback-time and re-evaluated feasibility
        # can disagree at the boundary).
        m6_min_slack = m6_median_slack = float('nan')
        final_front_c = np.empty((0, len(obj_indices) + 1))
        m8_best_err = m8_best_err_slack = float('nan')

    # Soft-HV: violation-graded HV of the FULL final archive (feasible
    # points unchanged, infeasible points pushed away from the origin in
    # BOTH objectives by their normalized constraint violation) -- defined
    # for every run regardless of scenario mode (see module docstring).
    if len(F_last_fin) > 0:
        violation = np.maximum(0.0, vmax_last)
        shifted = F_last_fin[:, obj_indices] + violation[:, None]
        nd_idx_soft = NonDominatedSorting().do(shifted, only_non_dominated_front=True)
        soft_hv = _hv(shifted[nd_idx_soft], REF_POINT)
    else:
        soft_hv = 0.0

    # Per-generation raw feasible HV (ratio against the true front, where one
    # exists, is taken once per scenario/instance in main()).
    hv_traj = [_hv(g, REF_POINT) for g in test_obj_archive]

    # exact_waste: exact companion to M3, see module docstring. n_evaluated /
    # n_feasible_evaluated are new run-callback keys (absent from every pkl
    # written before this counter existed).
    n_evaluated_raw = data.get('n_evaluated')
    n_feasible_evaluated_raw = data.get('n_feasible_evaluated')
    if n_evaluated_raw and n_feasible_evaluated_raw and n_evaluated_raw[-1]:
        exact_waste = 1.0 - n_feasible_evaluated_raw[-1] / n_evaluated_raw[-1]
        ne = np.asarray(n_evaluated_raw, dtype=float)
        nf = np.asarray(n_feasible_evaluated_raw, dtype=float)
        exact_waste_traj = np.where(ne > 0, 1.0 - nf / np.where(ne > 0, ne, 1), np.nan).tolist()
    else:
        exact_waste = float('nan')
        exact_waste_traj = []

    return dict(
        m1_ratio=m1_ratio.tolist(),
        m3_cumulative=m3_cumulative.tolist(),
        m3_per_gen=m3_per_gen.tolist(),
        m4_front_size=m4_front_size,
        m5_mean_violation=m5_mean_violation,
        hv_traj=hv_traj,
        m2_evals_to_first_feasible=m2_evals_to_first_feasible,
        m2_censored=m2_censored,
        m2_n_evaluated=m2_n_evaluated,
        m3_final_cumulative=float(m3_cumulative[-1]) if n_gen > 0 else float('nan'),
        m4_final=m4_front_size[-1] if m4_front_size else 0,
        m6_min_slack=m6_min_slack,
        m6_median_slack=m6_median_slack,
        m8_best_err=m8_best_err,
        m8_best_err_slack=m8_best_err_slack,
        soft_hv=soft_hv,
        exact_waste=exact_waste,
        exact_waste_traj=exact_waste_traj,
        final_front=final_front,
        final_front_c=final_front_c,
        final_infeas_c=final_infeas_c,
        n_gen=n_gen,
    )


# ─── discovery / IO ─────────────────────────────────────────────────────────

def discover_instances(results_root, scenarios):
    """Every (suite, pid) with at least one seed pkl under any of
    *scenarios* (pid_dir/{budget}/{method}/{handler}/seed_*.pkl)."""
    found = set()
    for scenario in scenarios:
        for pid_dir in glob.glob(os.path.join(results_root, scenario, '*', 'pid*')):
            if not os.path.isdir(pid_dir):
                continue
            suite = os.path.basename(os.path.dirname(pid_dir))
            try:
                pid = int(os.path.basename(pid_dir)[len('pid'):])
            except ValueError:
                continue
            if glob.glob(os.path.join(pid_dir, '*', '*', '*', 'seed_*.pkl')):
                found.add((suite, pid))
    return sorted(found)


def _parse_seed(fname):
    try:
        return int(fname[len('seed_'):-len('.pkl')])
    except ValueError:
        return None


def discover_runs(results_root, scenario, suite, pid, method, seeds):
    """(seed, handler, path, 'c2') tuples for one (scenario, suite, pid,
    method): pid_dir/{budget}/{method}/{handler}/seed_N.pkl -- the one
    layout run_constraint.py writes. The handler is the DISPLAY name
    (HANDLER_LABEL applied), not necessarily the directory name."""
    pid_dir = os.path.join(results_root, scenario, suite, f'pid{pid}')
    out = []
    for path in sorted(glob.glob(os.path.join(pid_dir, '*', method, '*', 'seed_*.pkl'))):
        rel = os.path.relpath(path, pid_dir).split(os.sep)
        if len(rel) != 4:
            continue
        _budget, _method, handler, fname = rel
        seed = _parse_seed(fname)
        if seed is None or (seeds is not None and seed not in seeds):
            continue
        out.append((seed, HANDLER_LABEL.get(handler, handler), path, 'c2'))
    return out


def _pad_mean(rows):
    """Elementwise nanmean over ragged lists of floats (right-padded with
    NaN) -- averages per-generation trajectories over seeds whose run lengths
    might differ slightly."""
    if not rows:
        return []
    max_len = max(len(r) for r in rows)
    arr = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    with np.errstate(invalid='ignore'):
        return np.nanmean(arr, axis=0).tolist()


# ─── plotting ───────────────────────────────────────────────────────────────
# Everything below consumes only the finished analysis artifacts
# (trajectories dict, plot_cache, metrics/stats DataFrames), so plots and
# LaTeX tables can be regenerated via --plots_only without redoing any
# true-eval work.

# (metric key in trajectories, y-label, filename stem, guard, y-scale)
TRAJ_METRICS = [
    ('m1_ratio',         'feasibility ratio (n_feasible / n_total)',       'feasibility_ratio', None,         1.0),
    ('hv_traj',          'feasible HV (raw, ref=1.05 per obj)',            'feasible_hv',       None,         1.0),
    ('m7_hv_ratio_traj', 'HV-ratio (run / true feasible front)',           'hv_ratio',          'enumerable', 1.0),
    ('exact_waste_traj', 'wasted HF evaluations so far [% of all HF evals]', 'waste',           None,         100.0),
]

# 'Wasted' = the share of ALL high-fidelity evaluations (DOE + infill) spent
# on architectures that turned out infeasible, from the runs' exact counters.
_WASTE_NOTE = {'hard': 'evaluated & DISCARDED by the hard limit',
               'soft': 'evaluated & archived, but infeasible'}


def _split_label(label):
    """'{method}:{handler}' -> (method, handler)."""
    method, _, handler = label.partition(':')
    return method, handler


def _traj_legend(ax, seen_handlers, seen_methods, plt):
    """Handler-colour + method-linestyle legends drawn into a spare
    (turned-off) grid cell."""
    from matplotlib.lines import Line2D
    ax.axis('off')
    handler_handles = [Line2D([0], [0], color=HANDLER_COLOURS.get(h, '#333333'), lw=2, label=h)
                       for h in HANDLER_ORDER if h in seen_handlers]
    method_handles = [Line2D([0], [0], color='black', ls=METHOD_LINESTYLE.get(m, '-'),
                             lw=1.5, label=m)
                      for m in METHODS if m in seen_methods]
    leg1 = ax.legend(handles=handler_handles, loc='upper left', fontsize=7, ncol=2,
                     title='handler (colour)', title_fontsize=8, frameon=False,
                     bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, loc='lower left', fontsize=7, ncol=3,
              title='method (linestyle)', title_fontsize=8, frameon=False,
              bbox_to_anchor=(0.0, 0.0))


def plot_trajectory_grids(trajectories, plot_cache, plot_dir, plt):
    """One figure per (scenario, trajectory metric): instance subplots on a
    shared grid, colour = handler, linestyle = method, one legend cell.
    Titles carry the scenario's mode + constraint type and each instance's
    constrained metric and threshold."""
    written = []
    for scenario, by_inst in sorted(trajectories['runs'].items()):
        mode = SCENARIOS[scenario]['mode']
        refs = plot_cache['refs'].get(scenario, {})
        for metric_key, ylabel, stem, guard, yscale in TRAJ_METRICS:
            inst_keys = [k for k in sorted(by_inst)
                         if guard != 'enumerable'
                         or trajectories['instances'].get(k, {}).get('enumerable')]
            inst_keys = [k for k in inst_keys
                         if any(v.get(metric_key) for sd in by_inst[k].values() for v in sd.values())]
            if not inst_keys:
                continue
            ncols = 3
            nrows = math.ceil((len(inst_keys) + 1) / ncols)   # +1 legend cell
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                                     squeeze=False)
            axes = axes.ravel()
            seen_handlers, seen_methods = set(), set()
            for ax, inst_key in zip(axes, inst_keys):
                for label in sorted(by_inst[inst_key]):
                    seeds_dict = by_inst[inst_key][label]
                    series = [v[metric_key] for v in seeds_dict.values() if v.get(metric_key)]
                    if not series:
                        continue
                    method, handler = _split_label(label)
                    seen_handlers.add(handler)
                    seen_methods.add(method)
                    mean_series = np.asarray(_pad_mean(series)) * yscale
                    ax.plot(range(1, len(mean_series) + 1), mean_series,
                            color=HANDLER_COLOURS.get(handler, '#333333'),
                            ls=METHOD_LINESTYLE.get(method, '-'), lw=1.2)
                ref = refs.get(inst_key)
                thr_note = (f"  ({ref['cfg']['constr_metric']} $\\leq$ {ref['threshold']:.3f})"
                            if ref else '')
                ax.set_title(f'{_inst_short(inst_key)} [{inst_key}]{thr_note}', fontsize=9)
                ax.set_xlabel('generation', fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.tick_params(labelsize=8)
            _traj_legend(axes[len(inst_keys)], seen_handlers, seen_methods, plt)
            for ax in axes[len(inst_keys) + 1:]:
                ax.axis('off')
            if stem == 'waste':
                fig.suptitle(f'{_scen_long(scenario)}: wasted '
                             f'high-fidelity evaluations\nshare of all HF evaluations '
                             f'(DOE + infill) spent on infeasible architectures '
                             f'[{_WASTE_NOTE[mode]}]', fontsize=11)
            else:
                fig.suptitle(f'{_scen_long(scenario)}: {ylabel}', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.93 if stem == 'waste' else 0.96))
            out_png = os.path.join(plot_dir, f'{scenario}_{stem}.png')
            fig.savefig(out_png, dpi=130)
            plt.close(fig)
            written.append(out_png)
    return written


def _draw_front_set(ax, fronts, colour, label, lw=1.6, band=True):
    """Median attainment staircase (+ best/worst band when n > 1 and *band*)
    for one set of per-seed feasible fronts."""
    fronts = [np.asarray(f, dtype=float) for f in fronts if f is not None and len(f) > 0]
    if not fronts:
        return
    if len(fronts) == 1:
        f = fronts[0][np.argsort(fronts[0][:, 0], kind='stable')]
        ax.step(f[:, 0], np.minimum.accumulate(f[:, 1]), where='post',
                color=colour, lw=lw, label=label)
        return
    n = len(fronts)
    k_med = math.ceil(n / 2)
    xs, surf = attainment_surfaces(fronts, ks=[1, k_med, n])
    to_nan = lambda a: np.where(np.isfinite(a), a, np.nan)
    ax.step(xs, to_nan(surf[k_med]), where='post', color=colour, lw=lw,
            label=f'{label} (median, n={n})')
    if band:
        ax.fill_between(xs, to_nan(surf[1]), to_nan(surf[n]), step='post',
                        color=colour, alpha=0.15, linewidth=0)


_CLOUD_PLOT_MAX = 6000    # per-subplot scatter budget (fixed-seed subsample)


def plot_attainment_grid(scenario, inst_key, inst_cache, ref, fronts_by_mh,
                         out_png, plt):
    """One figure per (scenario, instance): a subplot per HANDLER (all of
    them, not just the scenario default), each showing the feasible/
    infeasible attainable cloud, the true feasible front (staircase PLUS its
    actual member points as markers, so constraint-induced gaps in the front
    are visible), the per-method attainment staircases for that handler, and
    random's attainment as a grey reference."""
    handlers = [h for h in HANDLER_ORDER
                if any(mh[1] == h and mh[0] != 'random' for mh in fronts_by_mh)]
    if not handlers:
        return False
    random_fronts = [f for (m, _h), fr in fronts_by_mh.items() if m == 'random' for f in fr]

    F_cloud   = inst_cache['F_cloud']
    obj_names = inst_cache['obj_names']
    obj_idx   = ref['obj_indices']
    cidxs     = ref.get('constr_indices', [ref['constr_index']])
    thrs      = ref.get('thresholds', [ref['threshold']])
    cfg       = ref['cfg']

    obj  = F_cloud[:, obj_idx]
    feas = np.all([F_cloud[:, i] <= t for i, t in zip(cidxs, thrs)], axis=0)
    if len(F_cloud) > _CLOUD_PLOT_MAX:
        sub = np.random.RandomState(0).choice(len(F_cloud), _CLOUD_PLOT_MAX, replace=False)
    else:
        sub = np.arange(len(F_cloud))
    tf = None
    if ref['true_front'] is not None and len(ref['true_front']) > 0:
        tf = np.asarray(ref['true_front'], dtype=float)
        tf = tf[np.argsort(tf[:, 0], kind='stable')]

    ncols = 4
    nrows = math.ceil((len(handlers) + 1) / ncols)   # +1 legend cell
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.1 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax, handler in zip(axes, handlers):
        s_feas, s_inf = feas[sub], ~feas[sub]
        ax.scatter(obj[sub][s_inf, 0], obj[sub][s_inf, 1], s=2, c=CLOUD_INFEASIBLE_COLOUR,
                   alpha=0.4, linewidths=0, rasterized=True)
        ax.scatter(obj[sub][s_feas, 0], obj[sub][s_feas, 1], s=2, c=CLOUD_FEASIBLE_COLOUR,
                   alpha=0.4, linewidths=0, rasterized=True)
        if tf is not None:
            ax.step(tf[:, 0], np.minimum.accumulate(tf[:, 1]), where='post',
                    color='black', lw=1.0)
            ax.plot(tf[:, 0], tf[:, 1], '.', color='black', ms=3.5)
        _draw_front_set(ax, random_fronts, '#999999', 'random', lw=1.0, band=False)
        for method in ('samos', 'samos-cheap'):
            _draw_front_set(ax, fronts_by_mh.get((method, handler), []),
                            METHOD_COLOURS[method], method)
        ax.set_title(handler, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(f'{obj_names[obj_idx[0]]} (true-eval, norm.)', fontsize=7)
        ax.set_ylabel(f'{obj_names[obj_idx[1]]} (true-eval, norm.)', fontsize=7)

    # Legend cell: proxy artists for every element used above.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leg_ax = axes[len(handlers)]
    leg_ax.axis('off')
    handles = [
        Patch(color=CLOUD_FEASIBLE_COLOUR, label=f"feasible cloud ({inst_cache['cloud_label']})"),
        Patch(color=CLOUD_INFEASIBLE_COLOUR, label='infeasible cloud'),
    ]
    if tf is not None:
        handles.append(Line2D([0], [0], color='black', marker='.', lw=1.0,
                              label='true feasible front (points = members)'))
    handles.append(Line2D([0], [0], color='#999999', lw=1.0, label='random (median attainment)'))
    handles += [Line2D([0], [0], color=METHOD_COLOURS[m], lw=1.6,
                       label=f'{m} (median attainment $\\pm$ best/worst)')
                for m in ('samos', 'samos-cheap')]
    leg_ax.legend(handles=handles, loc='center left', fontsize=8, frameon=False)
    for ax in axes[len(handlers) + 1:]:
        ax.axis('off')

    cloud_note = '' if inst_cache['enumerable'] else \
        '   [cloud sampled (10k) -- approximation, not the true region]'
    constr_names, _ = _constraint_lists(cfg)
    constr_str = ', '.join(f'{m} $\\leq$ {t:.4f}' for m, t in zip(constr_names, thrs))
    fig.suptitle(f"{_scen_long(scenario)}  --  {inst_key} ({_inst_short(inst_key)}): "
                 f"{'/'.join(cfg['obj_metrics'])} objectives, "
                 f"constraint {constr_str}{cloud_note}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


_CLOUD_CONTEXT_COLOUR = '#d9d9d9'   # neutral context cloud (threshold line marks the split)


def _stack_fronts(by_mh, mh):
    pts = [f for f in (by_mh or {}).get(mh, []) if f is not None and len(f) > 0]
    return np.vstack(pts) if pts else None


def _draw_constr_panel(ax, x_cloud, y_cloud, sub, thr, tf_c, handler,
                       fronts_c_by_mh, fronts_ic_by_mh):
    """One (error, constrained metric) panel: grey context cloud, threshold
    line, true-front members, random reference (dots + x), and one handler's
    per-method feasible front members (dots) / infeasible archive members
    (x). Shared by the per-scenario grid and the hard-vs-soft variant."""
    ax.scatter(x_cloud[sub], y_cloud[sub], s=2, c=_CLOUD_CONTEXT_COLOUR,
               alpha=0.4, linewidths=0, rasterized=True)
    ax.axhline(thr, color='black', ls='--', lw=1.0)
    if tf_c is not None and len(tf_c) > 0:
        ax.plot(tf_c[:, 0], tf_c[:, 2], '.', color='black', ms=3.5)
    random_mh = [(m, h) for (m, h) in fronts_c_by_mh if m == 'random']
    if random_mh:
        rinf = _stack_fronts(fronts_ic_by_mh, random_mh[0])
        if rinf is not None:
            ax.scatter(rinf[:, 0], rinf[:, 2], s=10, c='#999999',
                       alpha=0.5, marker='x', linewidths=0.8)
        rpts = _stack_fronts(fronts_c_by_mh, random_mh[0])
        if rpts is not None:
            ax.scatter(rpts[:, 0], rpts[:, 2], s=5, c='#999999',
                       alpha=0.5, linewidths=0)
    for method in ('samos', 'samos-cheap'):
        inf = _stack_fronts(fronts_ic_by_mh, (method, handler))
        if inf is not None:
            ax.scatter(inf[:, 0], inf[:, 2], s=10, c=METHOD_COLOURS[method],
                       alpha=0.45, marker='x', linewidths=0.8)
        pts = _stack_fronts(fronts_c_by_mh, (method, handler))
        if pts is not None:
            ax.scatter(pts[:, 0], pts[:, 2], s=6, c=METHOD_COLOURS[method],
                       alpha=0.55, linewidths=0)


def _constr_panel_legend_handles(inst_cache, constr_name, thr, tf_present,
                                 x_label='infeasible archive members'):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(color=_CLOUD_CONTEXT_COLOUR, label=f"attainable cloud ({inst_cache['cloud_label']})"),
        Line2D([0], [0], color='black', ls='--', lw=1.0,
               label=f'threshold ({constr_name} = {thr:.4f})'),
    ]
    if tf_present:
        handles.append(Line2D([0], [0], color='black', marker='.', ls='none',
                              label='true feasible front members'))
    for colour, name in [('#999999', 'random'), (METHOD_COLOURS['samos'], 'samos'),
                         (METHOD_COLOURS['samos-cheap'], 'samos-cheap')]:
        handles.append(Line2D([0], [0], color=colour, marker='o', ls='none', ms=4,
                              label=f'{name}: feasible front members (all seeds)'))
        handles.append(Line2D([0], [0], color=colour, marker='x', ls='none', ms=5,
                              label=f'{name}: {x_label}'))
    return handles


def plot_constr_attainment_grid(scenario, inst_key, inst_cache, ref,
                                fronts_c_by_mh, fronts_ic_by_mh, out_png, plt):
    """Boundary diagnostic, one figure per (scenario, instance): a subplot
    per handler in the (error, constrained metric) plane. NOT a Pareto view
    (the constraint is a search objective only for the b0 rows). Per
    handler: a neutral grey context cloud (the threshold line marks the
    feasible half), each method's FINAL feasible-front members as dots, and
    each method's INFEASIBLE final-archive members as x markers -- in soft
    mode the ND-surviving infeasible samples the run kept, in hard mode
    archive members whose TRUE evaluation crosses the limit even though the
    gate admitted them (the discarded evaluations themselves are not
    recorded in the pkls). True-front members (enumerable only) in black."""
    if len(ref.get('constr_indices', [ref['constr_index']])) > 1:
        # Multi-constraint configs need a per-constraint variant (or a
        # max-violation y-axis) of this diagnostic -- deliberately deferred;
        # the CSV/stats layer carries the multi-constraint story.
        return False
    handlers = [h for h in HANDLER_ORDER
                if any(mh[1] == h and mh[0] != 'random' for mh in fronts_c_by_mh)]
    if not handlers:
        return False
    fronts_ic_by_mh = fronts_ic_by_mh or {}

    F_cloud = inst_cache['F_cloud']
    obj_idx = ref['obj_indices']
    cidx    = ref['constr_index']
    thr     = ref['threshold']
    cfg     = ref['cfg']
    err_name    = cfg['obj_metrics'][0]
    constr_name = cfg['constr_metric']

    x_cloud = F_cloud[:, obj_idx[0]]
    y_cloud = F_cloud[:, cidx]
    if len(F_cloud) > _CLOUD_PLOT_MAX:
        sub = np.random.RandomState(0).choice(len(F_cloud), _CLOUD_PLOT_MAX, replace=False)
    else:
        sub = np.arange(len(F_cloud))
    tf_c = ref.get('true_front_c')

    ncols = 4
    nrows = math.ceil((len(handlers) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.1 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax, handler in zip(axes, handlers):
        _draw_constr_panel(ax, x_cloud, y_cloud, sub, thr, tf_c, handler,
                           fronts_c_by_mh, fronts_ic_by_mh)
        ax.set_title(handler, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(f'{err_name} (true-eval, norm.)', fontsize=7)
        ax.set_ylabel(f'{constr_name} (true-eval, norm.)', fontsize=7)

    x_label = ('infeasible on re-evaluation (admitted during search)'
               if cfg['mode'] == 'hard' else 'infeasible archive members')
    leg_ax = axes[len(handlers)]
    leg_ax.axis('off')
    leg_ax.legend(handles=_constr_panel_legend_handles(
        inst_cache, constr_name, thr, tf_c is not None and len(tf_c) > 0,
        x_label=x_label),
        loc='center left', fontsize=7, frameon=False)
    for ax in axes[len(handlers) + 1:]:
        ax.axis('off')

    if cfg['mode'] == 'hard':
        inf_note = ('  [x = admitted as feasible when evaluated during the search, '
                    'infeasible on re-evaluation; discarded evaluations not shown]')
    else:
        inf_note = '  [x = infeasible solutions retained in the archive]'
    fig.suptitle(f"{_scen_long(scenario)}  --  {inst_key} ({_inst_short(inst_key)}): "
                 f"final archive members in the ({err_name}, {constr_name}) plane\n"
                 f"boundary diagnostic, not a Pareto view{inf_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


# Hard/soft scenario pairs sharing an instance set, metric and threshold,
# per constraint type -- the mode-comparison figures below.
MODE_PAIRS = {'cheap': ('s1', 's3'), 'expensive': ('s2', 's4')}

_MODE_COL_NOTE = {
    'hard': 'x = admitted as feasible when evaluated during\nthe search, infeasible '
            'on re-evaluation\n(evaluations the limit discarded are not shown)',
    'soft': 'x = infeasible solutions retained in the archive\n(evaluated and kept -- '
            'soft mode never discards)',
}

# Evaluation-budget composition categories for the per-panel inset bars.
_PORTION_COLOURS = [('feasible', '#a6cee3'),
                    ('infeasible, archived', '#fdbf6f'),
                    ('infeasible, not archived', '#e0605e')]

# The campaign is uniformly B1200_P20; per-run evaluations = pop_size x
# n_gen with pop_size 20 (run_constraint.py default, fixed on the sbatch).
_POP_SIZE = 20


def _eval_portions(metrics_df, scenario, suite, pid, method, handler, ic_by_mh, ic_key):
    """Mean (feasible, infeasible-archived, infeasible-not-archived) split
    of one row's HF-evaluation budget, from the exact waste counters
    (metrics CSV) and the per-seed infeasible-archive sizes (plot cache).
    None when the waste counters are absent."""
    rows = metrics_df[(metrics_df.scenario == scenario) & (metrics_df.suite == suite)
                      & (metrics_df.pid == pid) & (metrics_df.method == method)]
    if handler is not None:
        rows = rows[rows.handler == handler]
    rows = rows[np.isfinite(rows.exact_waste)]
    if rows.empty:
        return None
    waste  = float(rows.exact_waste.mean())
    n_eval = _POP_SIZE * float(rows.n_gen.mean())
    inf_arrays = (ic_by_mh or {}).get(ic_key, [])
    inf_arch = float(np.mean([len(a) for a in inf_arrays])) if inf_arrays else 0.0
    p_arch = min(inf_arch / n_eval, waste) if n_eval > 0 else 0.0
    return (1.0 - waste, p_arch, waste - p_arch)


def _draw_portion_bars(ax, portions):
    """Normalized stacked bars on a dedicated (panel-height) axes: one bar
    per method in *portions* (label, triple-or-None), category colours from
    _PORTION_COLOURS."""
    labels = []
    for i, (label, p) in enumerate(portions):
        labels.append(label)
        if p is None:
            continue
        bottom = 0.0
        for frac, (_name, colour) in zip(p, _PORTION_COLOURS):
            ax.bar(i, frac, bottom=bottom, color=colour, width=0.72)
            bottom += frac
    ax.set_xticks(range(len(portions)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_xlim(-0.6, len(portions) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(labelsize=6, length=2)


def plot_constr_mode_pairs(plot_cache, metrics_df, plot_dir, plt):
    """Hard-vs-soft comparison in the (error, constrained metric) plane: one
    figure per (constraint type, instance), rows = handlers, left column =
    the hard scenario, right column = the soft scenario (identical instance,
    metric and threshold -- only the mode differs). Panels share axes per
    figure so the two modes compare directly. Each panel carries a
    normalized stacked-bar inset (random / samos / samos-cheap) with the
    mean evaluation-budget split: feasible, infeasible-but-archived,
    infeasible not archived (hard mode: discarded)."""
    written = []
    fronts_c  = plot_cache.get('fronts_c') or {}
    fronts_ic = plot_cache.get('fronts_ic') or {}
    pair_dir  = os.path.join(plot_dir, 'constr_vs_err_modes')
    for ctype, (sc_hard, sc_soft) in MODE_PAIRS.items():
        common = sorted(set(fronts_c.get(sc_hard, {})) & set(fronts_c.get(sc_soft, {})))
        for inst_key in common:
            inst_cache = plot_cache['instances'].get(inst_key)
            refs = {'hard': plot_cache['refs'].get(sc_hard, {}).get(inst_key),
                    'soft': plot_cache['refs'].get(sc_soft, {}).get(inst_key)}
            if inst_cache is None or refs['hard'] is None or refs['soft'] is None:
                continue
            by_mode_c  = {'hard': fronts_c[sc_hard][inst_key], 'soft': fronts_c[sc_soft][inst_key]}
            by_mode_ic = {'hard': (fronts_ic.get(sc_hard, {}) or {}).get(inst_key, {}),
                          'soft': (fronts_ic.get(sc_soft, {}) or {}).get(inst_key, {})}
            handlers = [h for h in HANDLER_ORDER
                        if any(mh[1] == h and mh[0] != 'random'
                               for mode in ('hard', 'soft') for mh in by_mode_c[mode])]
            if not handlers:
                continue

            ref = refs['hard']   # instance geometry is mode-independent
            F_cloud = inst_cache['F_cloud']
            obj_idx = ref['obj_indices']
            cidx    = ref['constr_index']
            thr     = ref['threshold']
            err_name    = ref['cfg']['obj_metrics'][0]
            constr_name = ref['cfg']['constr_metric']
            x_cloud = F_cloud[:, obj_idx[0]]
            y_cloud = F_cloud[:, cidx]
            if len(F_cloud) > _CLOUD_PLOT_MAX:
                sub = np.random.RandomState(0).choice(len(F_cloud), _CLOUD_PLOT_MAX, replace=False)
            else:
                sub = np.arange(len(F_cloud))

            os.makedirs(pair_dir, exist_ok=True)
            fig, axes = plt.subplots(len(handlers), 4,
                                     figsize=(11.0, 2.5 * len(handlers)),
                                     squeeze=False,
                                     gridspec_kw={'width_ratios': [1, 0.26, 1, 0.26]})
            suite, pid_s = inst_key.split('/pid')
            pid = int(pid_s)
            scen_of = {'hard': sc_hard, 'soft': sc_soft}
            for row, handler in enumerate(handlers):
                for mode_i, mode in enumerate(('hard', 'soft')):
                    ax, bar_ax = axes[row, 2 * mode_i], axes[row, 2 * mode_i + 1]
                    _draw_constr_panel(ax, x_cloud, y_cloud, sub, thr,
                                       refs[mode].get('true_front_c'), handler,
                                       by_mode_c[mode], by_mode_ic[mode])
                    random_mh = [(m, h) for (m, h) in by_mode_c[mode] if m == 'random']
                    portions = [('rnd', _eval_portions(metrics_df, scen_of[mode], suite, pid,
                                                       'random', None, by_mode_ic[mode],
                                                       random_mh[0] if random_mh else None)),
                                ('s', _eval_portions(metrics_df, scen_of[mode], suite, pid,
                                                     'samos', handler, by_mode_ic[mode],
                                                     ('samos', handler))),
                                ('sc', _eval_portions(metrics_df, scen_of[mode], suite, pid,
                                                      'samos-cheap', handler, by_mode_ic[mode],
                                                      ('samos-cheap', handler)))]
                    _draw_portion_bars(bar_ax, portions)
                    ax.tick_params(labelsize=7)
                    if row == 0:
                        ax.set_title(f'{mode} mode\n[{_MODE_COL_NOTE[mode]}]', fontsize=8)
                        bar_ax.set_title('HF budget\nsplit', fontsize=8)
                    if mode_i == 0:
                        ax.set_ylabel(f'{handler}\n{constr_name} (norm.)', fontsize=8)
                    if row == len(handlers) - 1:
                        ax.set_xlabel(f'{err_name} (true-eval, norm.)', fontsize=8)

            from matplotlib.patches import Patch
            handles = _constr_panel_legend_handles(
                inst_cache, constr_name, thr,
                any(refs[m].get('true_front_c') is not None for m in refs),
                x_label='infeasible members (meaning differs by mode -- see column notes)')
            handles += [Patch(color=colour, label=f'bars: {name} (mean share of HF budget)')
                        for name, colour in _PORTION_COLOURS]
            fig.legend(handles=handles, loc='lower center', fontsize=7,
                       frameon=False, ncol=3)
            fig.suptitle(f'{ctype} constraint ({constr_name}) -- {inst_key} '
                         f'({_inst_short(inst_key)}): hard vs soft mode\nfinal archive '
                         f'members in the ({err_name}, {constr_name}) plane '
                         f'[boundary diagnostic, not a Pareto view]', fontsize=10)
            fig.tight_layout(rect=(0, 0.05, 1, 0.965))
            out_png = os.path.join(pair_dir, f'{ctype}_{suite}_pid{pid_s}_hard_vs_soft.png')
            fig.savefig(out_png, dpi=130)
            plt.close(fig)
            written.append(out_png)
    return written


def plot_final_bars(metrics_df, plot_dir, plt, value_col, yscale, ylabel,
                    stem, suptitle_fn):
    """Final-value grouped bars for one CSV column, one figure per scenario:
    instance subplots, x = handler (HANDLER_ORDER), bars = samos /
    samos-cheap (mean over seeds, std as error bars), random's mean as a
    dotted reference line. NaN rows are dropped; a scenario with no finite
    values gets no figure. *suptitle_fn(scenario, mode, ctype)* builds the
    figure title."""
    written = []
    for scenario in sorted(metrics_df.scenario.unique()):
        sub = metrics_df[(metrics_df.scenario == scenario) & np.isfinite(metrics_df[value_col])]
        if sub.empty:
            continue
        mode  = SCENARIOS[scenario]['mode']
        ctype = SCENARIOS[scenario]['constr']
        inst_keys = sorted(f'{s}/pid{p}' for s, p in
                           sub[['suite', 'pid']].drop_duplicates().itertuples(index=False))
        ncols = 3
        nrows = math.ceil((len(inst_keys) + 1) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                                 squeeze=False)
        axes = axes.ravel()
        for ax, inst_key in zip(axes, inst_keys):
            suite, pid_s = inst_key.split('/pid')
            d = sub[(sub.suite == suite) & (sub.pid == int(pid_s))]
            handlers = [h for h in HANDLER_ORDER if (d.handler == h).any()]
            x = np.arange(len(handlers))
            width = 0.38
            for off, method in ((-width / 2, 'samos'), (width / 2, 'samos-cheap')):
                g = d[d.method == method].groupby('handler')[value_col]
                mu = [g.mean().get(h, np.nan) * yscale for h in handlers]
                sd = [g.std(ddof=0).get(h, 0.0) * yscale for h in handlers]
                ax.bar(x + off, mu, width, yerr=sd, capsize=2,
                       color=METHOD_COLOURS[method], error_kw=dict(lw=0.8))
            rnd = d[d.method == 'random'][value_col]
            if not rnd.empty:
                ax.axhline(rnd.mean() * yscale, color=METHOD_COLOURS['random'], ls=':', lw=1.4)
            ax.set_xticks(x)
            ax.set_xticklabels(handlers, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f'{_inst_short(inst_key)} [{inst_key}]', fontsize=9)
            ax.tick_params(labelsize=7)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        leg_ax = axes[len(inst_keys)]
        leg_ax.axis('off')
        leg_ax.legend(handles=[
            Patch(color=METHOD_COLOURS['samos'], label='samos (mean $\\pm$ std)'),
            Patch(color=METHOD_COLOURS['samos-cheap'], label='samos-cheap (mean $\\pm$ std)'),
            Line2D([0], [0], color=METHOD_COLOURS['random'], ls=':', lw=1.4,
                   label='random (mean)'),
        ], loc='center left', fontsize=9, frameon=False)
        for ax in axes[len(inst_keys) + 1:]:
            ax.axis('off')
        fig.suptitle(suptitle_fn(scenario, mode, ctype), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_png = os.path.join(plot_dir, f'{scenario}_{stem}.png')
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        written.append(out_png)
    return written


def plot_waste_bars(metrics_df, plot_dir, plt):
    return plot_final_bars(
        metrics_df, plot_dir, plt, value_col='exact_waste', yscale=100.0,
        ylabel='wasted HF evaluations [% of all HF evals]', stem='waste_bars',
        suptitle_fn=lambda sc, mode, ctype: (
            f'{_scen_long(sc)}: wasted high-fidelity '
            f'evaluations at the end of the run\nshare of all HF evaluations '
            f'(DOE + infill) spent on infeasible architectures [{_WASTE_NOTE[mode]}]'))


def plot_best_err_bars(metrics_df, plot_dir, plt):
    return plot_final_bars(
        metrics_df, plot_dir, plt, value_col='m8_best_err', yscale=1.0,
        ylabel='best feasible Err. (true-eval, norm.)', stem='best_err_bars',
        suptitle_fn=lambda sc, mode, ctype: (
            f'{_scen_long(sc)}: best feasible error (M8) '
            f'-- lower is better\nminimum true error over each run\'s final '
            f'FEASIBLE solutions: how far each handler pushes accuracy within '
            f'the constraint limit'))


# ─── LaTeX tables ───────────────────────────────────────────────────────────

def _tex_escape(s):
    return str(s).replace('#', r'\#').replace('%', r'\%').replace('_', r'\_')


def write_latex_tables(metrics_df, stats_df, tables_dir):
    """Report-ready booktabs tables (one .tex file, one table env per
    block, \\input-able individually via % ---- markers):

      1. Per scenario: final feasible HV, mean +- std over seeds, rows =
         handlers, column groups = (method x instance). Best mean per
         column in bold; cells with n < 20 seeds marked with^{\\dagger}.
      2. Same-GA (nsga2 family) significance summary: Holm-significant
         win/loss/tie counts per handler per scenario, feasible HV.
      3. GA-effect study: per same-formulation pair and scenario, the
         number of instances x methods where SMS-EMOA / NSGA-II is
         Holm-significantly better (feasible HV).
    """
    os.makedirs(tables_dir, exist_ok=True)
    lines = []

    inst_order = sorted(metrics_df[['suite', 'pid']].drop_duplicates().itertuples(index=False),
                        key=lambda t: (t.suite, t.pid))

    # ── Table blocks 1a/1b: per-scenario value tables (HV, best error) ─────
    # Transposed orientation: handlers as COLUMNS, one row per
    # (method, instance); random (method-independent of the handler axis)
    # is the last column. table* spans both columns of a two-column layout.
    def _scenario_value_tables(value_col, caption_body, label_suffix, best_is_max):
        if value_col not in metrics_df.columns:
            lines.append(f'% ---- {value_col} absent from metrics CSV; table skipped ----')
            lines.append('')
            return
        for scenario in sorted(metrics_df.scenario.unique()):
            sub = metrics_df[metrics_df.scenario == scenario]
            insts = [(s, p) for s, p in inst_order
                     if not sub[(sub.suite == s) & (sub.pid == p)].empty]
            mode  = SCENARIOS[scenario]['mode']
            ctype = SCENARIOS[scenario]['constr']
            methods  = [m for m in ('samos', 'samos-cheap') if (sub.method == m).any()]
            handlers = [h for h in HANDLER_ORDER if (sub.handler == h).any()]
            has_random = (sub.method == 'random').any()
            colspec = 'll' + 'r' * len(handlers) + ('r' if has_random else '')
            lines.extend([
                f'% ---- {SCENARIO_LABEL.get(scenario, scenario)}: {caption_body} '
                f'(mean +- std over seeds) ----',
                r'\begin{table*}[t]', r'\centering',
                rf'\caption{{{_scen_long(scenario).capitalize()}: {caption_body}, '
                rf'mean $\pm$ std over seeds. Best handler per row in bold; '
                rf'$\dagger$: fewer than 20 seeds. The random column is '
                rf'handler-independent (scenario-default handler).}}',
                rf'\label{{tab:c2-{SCENARIO_SLUG.get(scenario, scenario)}-{label_suffix}}}',
                r'\resizebox{\textwidth}{!}{%',
                rf'\begin{{tabular}}{{{colspec}}}', r'\toprule'])
            head = ['method', 'instance'] + [_tex_escape(h) for h in handlers] \
                + (['random'] if has_random else [])
            lines.append(' & '.join(head) + r' \\')
            lines.append(r'\midrule')

            for mi, m in enumerate(methods):
                if mi:
                    lines.append(r'\midrule')
                for ii, i in enumerate(insts):
                    d_mi = sub[(sub.method == m) & (sub.suite == i[0]) & (sub.pid == i[1])]
                    means = d_mi.groupby('handler')[value_col].mean()
                    best = (means.max() if best_is_max else means.min()) if not means.empty else np.nan
                    row = [_tex_escape(m) if ii == 0 else '',
                           INSTANCE_SHORT.get(i, f'{i[0]}/{i[1]}')]
                    for h in handlers:
                        cell = d_mi[d_mi.handler == h][value_col]
                        if cell.empty:
                            row.append('--')
                            continue
                        mu, sd, n = cell.mean(), cell.std(ddof=0), len(cell)
                        dag = r'$^{\dagger}$' if n < 20 else ''
                        txt = f'{mu:.3f}$\\pm${sd:.3f}{dag}'
                        if abs(mu - best) < 1e-12:
                            txt = rf'\textbf{{{txt}}}'
                        row.append(txt)
                    if has_random:
                        cell = sub[(sub.method == 'random') & (sub.suite == i[0])
                                   & (sub.pid == i[1])][value_col]
                        row.append(f'{cell.mean():.3f}$\\pm${cell.std(ddof=0):.3f}'
                                   if not cell.empty else '--')
                    lines.append(' & '.join(row) + r' \\')
            lines.extend([r'\bottomrule', r'\end{tabular}}', r'\end{table*}', ''])

    _scenario_value_tables('m7_hv_run', 'final feasible hypervolume', 'hv', best_is_max=True)
    _scenario_value_tables('m8_best_err',
                           'best feasible error (min.\\ true error over final feasible solutions; '
                           'lower is better)', 'best-err', best_is_max=False)

    if stats_df.empty:
        # No testable handler pair (e.g. a single-seed smoke tree): the
        # value tables above still stand, the significance blocks cannot.
        lines.append('% ---- significance tables skipped: no handler pair had enough '
                     'common seeds to test ----')
        out_tex = os.path.join(tables_dir, 'constraint2_tables.tex')
        with open(out_tex, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        return out_tex

    # ── Table block 2: same-GA significance summary (scenarios as rows) ────
    scen_order = sorted(stats_df.scenario.unique())
    ns = stats_df[(stats_df.family == 'nsga2') & (stats_df.metric == 'm7_hv_run')
                  & (stats_df.method != 'random')]
    wlt_handlers = [h for h in HANDLER_ORDER if h in set(ns.handler_a) | set(ns.handler_b)]
    lines += ['% ---- same-GA (NSGA-II) significance summary: W/L/T per handler ----',
              r'\begin{table*}[t]', r'\centering',
              r'\caption{Same-GA comparison (NSGA-II inner): Holm-significant '
              r'win/loss/tie counts per handler on final feasible HV, over all '
              r'pairwise tests (instances $\times$ methods), per scenario.}',
              r'\label{tab:c2-nsga2-wlt}',
              r'\resizebox{\textwidth}{!}{%',
              rf"\begin{{tabular}}{{l{'c' * len(wlt_handlers)}}}", r'\toprule',
              'scenario & ' + ' & '.join(_tex_escape(h) for h in wlt_handlers) + r' \\',
              r'\midrule']
    for sc in scen_order:
        row = [SCENARIO_LABEL.get(sc, sc)]
        for h in wlt_handlers:
            d = ns[(ns.scenario == sc) & ((ns.handler_a == h) | (ns.handler_b == h))]
            w = int((d.significant & (((d.handler_a == h) & (d.median_a > d.median_b))
                                      | ((d.handler_b == h) & (d.median_b > d.median_a)))).sum())
            l = int((d.significant & (((d.handler_a == h) & (d.median_a < d.median_b))
                                      | ((d.handler_b == h) & (d.median_b < d.median_a)))).sum())
            row.append(f'{w}/{l}/{len(d) - w - l}')
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}', '']

    # ── Table block 3: GA-effect study (scenarios as rows) ─────────────────
    ga = stats_df[(stats_df.family == 'ga-effect') & (stats_df.metric == 'm7_hv_run')]
    ga_pairs = [('h4-cdp: NSGA-II vs SMS', ('h4-cdp', 'h4-cdp-sms')),
                ('b0: NSGA-II vs SMS', ('b0-nsga2', 'b0-sms'))]
    lines += ['% ---- GA-effect study: same formulation, NSGA-II vs SMS-EMOA ----',
              r'\begin{table*}[t]', r'\centering',
              r'\caption{Inner-GA effect (same formulation, seed-paired Wilcoxon on '
              r'final feasible HV): counts of Holm-significant outcomes over '
              r'instances $\times$ methods per scenario.}',
              r'\label{tab:c2-ga-effect}',
              rf"\begin{{tabular}}{{l{'c' * len(ga_pairs)}}}", r'\toprule',
              'scenario & ' + ' & '.join(_tex_escape(p) for p, _ in ga_pairs) + r' \\',
              r'\midrule']
    for sc in scen_order:
        row = [SCENARIO_LABEL.get(sc, sc)]
        for _label, (ha, hb) in ga_pairs:
            d = ga[(ga.scenario == sc)
                   & (((ga.handler_a == ha) & (ga.handler_b == hb))
                      | ((ga.handler_a == hb) & (ga.handler_b == ha)))]
            nsga_w = int((d.significant & (((d.handler_a == ha) & (d.median_a > d.median_b))
                                           | ((d.handler_b == ha) & (d.median_b > d.median_a)))).sum())
            sms_w = int((d.significant).sum()) - nsga_w
            row.append(f'{nsga_w}/{sms_w}/{len(d) - nsga_w - sms_w}')
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '',
              '% cell format: NSGA-II wins / SMS wins / ties  (block 3),',
              '%              wins / losses / ties            (block 2)']

    out_tex = os.path.join(tables_dir, 'constraint2_tables.tex')
    with open(out_tex, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out_tex


# ─── shared rendering entry point (full run and --plots_only) ───────────────

def render_outputs(output_dir, trajectories, plot_cache, metrics_df, stats_df):
    """All figures + LaTeX tables from finished analysis artifacts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[analyse] matplotlib not available; skipping plots.')
        return
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for out in plot_trajectory_grids(trajectories, plot_cache, plot_dir, plt):
        print(f'[analyse] wrote {out}')

    for out in plot_waste_bars(metrics_df, plot_dir, plt):
        print(f'[analyse] wrote {out}')
    if 'm8_best_err' in metrics_df.columns:
        for out in plot_best_err_bars(metrics_df, plot_dir, plt):
            print(f'[analyse] wrote {out}')
    else:
        print('[analyse] metrics CSV predates m8_best_err; best-error bars skipped '
              '(re-run the full analysis to add them).')

    for scenario, by_inst in sorted(plot_cache['fronts'].items()):
        for inst_key, fronts_by_mh in sorted(by_inst.items()):
            ref = plot_cache['refs'].get(scenario, {}).get(inst_key)
            inst_cache = plot_cache['instances'].get(inst_key)
            if ref is None or inst_cache is None:
                continue
            suite, pid_s = inst_key.split('/pid')
            out_png = os.path.join(plot_dir, f'{scenario}_{suite}_pid{pid_s}_attainment.png')
            if plot_attainment_grid(scenario, inst_key, inst_cache, ref,
                                    fronts_by_mh, out_png, plt):
                print(f'[analyse] wrote {out_png}')

    # Constraint-vs-error boundary diagnostics -- separate folder. Absent
    # from plot caches written before final_front_c existed (skip + note).
    fronts_c = plot_cache.get('fronts_c') or {}
    if fronts_c:
        constr_dir = os.path.join(plot_dir, 'constr_vs_err')
        os.makedirs(constr_dir, exist_ok=True)
        fronts_ic = plot_cache.get('fronts_ic') or {}
        for scenario, by_inst in sorted(fronts_c.items()):
            for inst_key, fronts_c_by_mh in sorted(by_inst.items()):
                ref = plot_cache['refs'].get(scenario, {}).get(inst_key)
                inst_cache = plot_cache['instances'].get(inst_key)
                if ref is None or inst_cache is None:
                    continue
                suite, pid_s = inst_key.split('/pid')
                out_png = os.path.join(constr_dir, f'{scenario}_{suite}_pid{pid_s}_constr_vs_err.png')
                if plot_constr_attainment_grid(scenario, inst_key, inst_cache, ref,
                                               fronts_c_by_mh,
                                               fronts_ic.get(scenario, {}).get(inst_key),
                                               out_png, plt):
                    print(f'[analyse] wrote {out_png}')
        for out in plot_constr_mode_pairs(plot_cache, metrics_df, plot_dir, plt):
            print(f'[analyse] wrote {out}')
    else:
        print('[analyse] plot cache predates final_front_c; constraint-vs-error plots '
              'skipped (re-run the full analysis to add them).')

    out_tex = write_latex_tables(metrics_df, stats_df, os.path.join(output_dir, 'tables'))
    print(f'[analyse] wrote {out_tex}')


# ─── reference (true feasible front / HV) per config signature ─────────────

def _build_ref(scenario, inst_key, config_kind, resolved, F_cloud, cloud_label, enumerable, suite, pid):
    """True feasible front / HV reference for one config signature, built
    once and reused by every run that shares it (see main()). Returns None
    (with a warning) if *resolved*'s objectives/constraint don't resolve to
    benchmark columns for this instance."""
    constr_names, thresholds = _constraint_lists(resolved)
    try:
        obj_indices    = [metric_index(suite, pid, m) for m in resolved['obj_metrics']]
        constr_indices = [metric_index(suite, pid, m) for m in constr_names]
    except KeyError:
        print(f"  [analyse] WARN: {scenario}/{inst_key} ({config_kind}): objectives "
              f"{resolved['obj_metrics']} / constraint {resolved['constr_metric']!r} not "
              f"resolvable for this instance; skipping its rows.")
        return None

    threshold = resolved['threshold']
    frac, cloud_front, cloud_front_c = true_feasible_front(
        F_cloud, obj_indices, constr_indices, thresholds)
    if enumerable:
        # Sanity asserts (enumerable = exact ground truth only). The
        # enumerated fraction may differ slightly from the design fraction
        # (thresholds are quantiles of a 10k random SAMPLE of the space).
        assert len(cloud_front) > 0, (
            f'{scenario}/{inst_key} ({config_kind}): true feasible front is empty -- '
            f'threshold/constraint wiring is broken.')
        if len(constr_names) == 1:
            design = DESIGN_FEASIBLE_FRACTION.get((suite, pid), {}).get(constr_names[0])
        else:
            design = JOINT_DESIGN_FEASIBLE_FRACTION.get(
                (SCENARIOS.get(scenario, {}).get('constr'), (suite, pid)))
        assert design is not None and abs(frac - design) < 0.05, (
            f'{scenario}/{inst_key} ({config_kind}): enumerated feasible fraction {frac:.4f} far '
            f'from the design fraction {design} (T={threshold}, constr={resolved["constr_metric"]}) '
            f'-- check THRESHOLDS.md provenance.')
        true_front   = cloud_front
        true_front_c = cloud_front_c
        hv_true      = _hv(true_front, REF_POINT)
        igd_ind      = IGDPlus(true_front)
        # True best feasible error (column 0 = the error objective).
        best_err_true = float(np.min(true_front[:, 0]))
        print(f'[analyse] {scenario}/{inst_key} ({config_kind}): true feasible fraction = {frac:.4f} '
              f'({int(round(frac * len(F_cloud)))}/{len(F_cloud)}), true feasible ND-front size = '
              f'{len(true_front)}, HV(true front) = {hv_true:.4f}')
    else:
        true_front, true_front_c, hv_true, igd_ind = None, None, float('nan'), None
        best_err_true = float('nan')
        print(f'[analyse] {scenario}/{inst_key} ({config_kind}): sampled feasible fraction = {frac:.4f} '
              f'(info only, {cloud_label}); M7 = n/a (non-enumerable).')

    # constr_index/threshold stay scalar aliases (first constraint) for the
    # single-constraint plot paths; constr_indices/thresholds are the
    # authoritative lists.
    return dict(cfg=resolved, obj_indices=obj_indices,
                constr_indices=constr_indices, thresholds=list(thresholds),
                constr_index=constr_indices[0], threshold=thresholds[0],
                frac=frac, true_front=true_front,
                true_front_c=true_front_c, hv_true=hv_true, igd_ind=igd_ind,
                best_err_true=best_err_true)


# ─── handler significance testing ──────────────────────────────────────────

def _paired_wilcoxon(a_vals, b_vals):
    """Two-sided Wilcoxon signed-rank on seed-paired samples. An
    all-zero difference vector (byte-identical runs, e.g. the replicated
    random rows) is a p = 1.0 tie by definition -- scipy's ``wilcoxon``
    raises on it instead. zero_method='wilcox' drops the remaining zero
    differences (classic treatment); method='auto' picks the exact null
    distribution when the (nonzero) sample is small and tie-free."""
    d = np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float)
    if np.all(d == 0.0):
        return 0.0, 1.0
    stat, p = wilcoxon(a_vals, b_vals, alternative='two-sided',
                       zero_method='wilcox', method='auto')
    return float(stat), float(p)


def _holm(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the same order as *pvals* (each clipped to [running max, 1])."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def _stat_families(handlers, ga_of):
    """{family: [(a, b), ...]} comparison families over *handlers*:

      'nsga2'     -- every pair with NSGA-II inners on BOTH sides (the
                     priority comparison: the shared handler row plus
                     b0-nsga2, all on the same inner GA).
      'sms'       -- every pair with SMS-EMOA inners on both sides
                     (b0-sms vs h4-cdp-sms: formulation under SMS).
      'ga-effect' -- the separate inner-GA study: same-FORMULATION pairs
                     across the GA axis (h4-cdp vs h4-cdp-sms, b0-nsga2 vs
                     b0-sms).

    Cross-GA, cross-formulation pairs (e.g. h2-penalty vs b0-sms) are
    deliberately NOT tested: they differ in two factors at once, and each
    factor already has its own clean comparison above. Each family is its
    own Holm-correction unit."""
    fams = {'nsga2': [], 'sms': [], 'ga-effect': []}
    for a, b in itertools.combinations(handlers, 2):
        ga_a, ga_b = ga_of(a), ga_of(b)
        if ga_a == ga_b == 'nsga2':
            fams['nsga2'].append((a, b))
        elif ga_a == ga_b == 'sms':
            fams['sms'].append((a, b))
        elif FORMULATION.get(a) is not None and FORMULATION.get(a) == FORMULATION.get(b):
            fams['ga-effect'].append((a, b))
    return {name: pairs for name, pairs in fams.items() if pairs}


def compute_handler_stats(stat_records, alpha=STATS_ALPHA, min_seeds=STATS_MIN_SEEDS):
    """Pairwise two-sided Wilcoxon SIGNED-RANK, paired by seed (seed-matched
    runs share RNG streams and DOE, see module docstring),
    between handler pairs sharing the same (scenario, instance, config
    signature, method), on each of STATS_METRICS. Pairs are grouped into
    comparison FAMILIES (see _stat_families): same-inner-GA comparisons
    ('nsga2', 'sms') and the separate GA-effect study ('ga-effect');
    Holm-corrected within each (scenario, instance, config signature,
    method, metric, family). Pairs with fewer than *min_seeds* common
    finite seeds are skipped (noted, excluded from the family's correction)
    rather than tested -- guards against a partial/still-downloading
    instance.

    Returns (rows, matrices):
      rows      -- list of dict rows for handler_stats.csv.
      matrices  -- {(scenario, inst_key): [block, ...]} where each block is
                   a dict(config_label, method, metric, family, handlers,
                   cell) for the stdout win/tie/loss printer; cell[(a, b)]
                   is 'W'/'L'/'T' from handler a's perspective against b
                   ('.' printed for untested pairs).
    """
    groups = {}
    for r in stat_records:
        key = (r['scenario'], r['inst_key'], r['sig'], r['method'])
        g = groups.setdefault(key, dict(kinds=set(), suite=r['suite'], pid=r['pid'],
                                         enumerable=r['enumerable'], by_handler={}))
        g['kinds'].add(r['config_kind'])
        g['by_handler'].setdefault(r['handler'], []).append(r)

    rows = []
    matrices = {}
    for (scenario, inst_key, sig, method), g in groups.items():
        handlers = sorted(g['by_handler'], key=_handler_sort_key)
        if len(handlers) < 2:
            continue
        config_label = '+'.join(sorted(k.replace(':', '-') for k in g['kinds']))

        def ga_of(h):
            ga = g['by_handler'][h][0].get('inner_ga')
            if ga in ('sms', 'nsga2'):
                return ga
            return 'sms' if h in _SMS_HANDLERS else 'nsga2'

        families = _stat_families(handlers, ga_of)
        for metric in STATS_METRICS:
            if metric == 'm7_hv_ratio' and not g['enumerable']:
                continue    # all-NaN for this instance; nothing to test
            # Seed-keyed values for pairing: one value per (handler,
            # seed), non-finite dropped. A duplicate (handler, seed) row
            # cannot happen by construction (one pkl per seed per row).
            values = {}
            for h in handlers:
                values[h] = {rec['seed']: float(rec[metric])
                             for rec in g['by_handler'][h]
                             if np.isfinite(rec[metric])}

            for family, pairs in families.items():
                pair_results = []
                for a, b in pairs:
                    va, vb = values[a], values[b]
                    common = sorted(set(va) & set(vb))
                    if len(common) < min_seeds:
                        print(f'  [analyse] handler-stats: {scenario}/{inst_key} config={config_label} '
                              f'method={method} metric={metric} family={family}: {a} (n={len(va)}) vs '
                              f'{b} (n={len(vb)}) -- fewer than {min_seeds} common seeds '
                              f'({len(common)}); skipped (not tested).')
                        continue
                    a_vals = np.array([va[s] for s in common], dtype=float)
                    b_vals = np.array([vb[s] for s in common], dtype=float)
                    stat, p = _paired_wilcoxon(a_vals, b_vals)
                    pair_results.append(dict(a=a, b=b, stat=stat, p_raw=p,
                                              n_a=len(common), n_b=len(common),
                                              med_a=float(np.median(a_vals)), med_b=float(np.median(b_vals))))
                if not pair_results:
                    continue

                p_holm = _holm([pr['p_raw'] for pr in pair_results])
                cell = {}
                fam_handlers = sorted({h for pr in pair_results for h in (pr['a'], pr['b'])},
                                      key=_handler_sort_key)
                direction = STATS_METRIC_DIRECTION.get(metric, 1)
                for pr, p_adj in zip(pair_results, p_holm):
                    significant = bool(p_adj < alpha)
                    better_a = (pr['med_a'] > pr['med_b']) if direction > 0 \
                        else (pr['med_a'] < pr['med_b'])
                    outcome_ab = ('W' if better_a else 'L') if significant else 'T'
                    cell[(pr['a'], pr['b'])] = outcome_ab
                    cell[(pr['b'], pr['a'])] = {'W': 'L', 'L': 'W', 'T': 'T'}[outcome_ab]
                    # method_ga is handler-dependent (see _method_ga), so this
                    # row gets one column per side rather than a single
                    # 'method_ga' -- they differ exactly on the 'ga-effect'
                    # family's pairs.
                    rows.append(dict(
                        scenario=scenario, suite=g['suite'], pid=g['pid'], config=config_label,
                        method=method, metric=metric, family=family,
                        handler_a=pr['a'], handler_b=pr['b'],
                        method_ga_a=_method_ga(method, pr['a'], ga_of(pr['a'])),
                        method_ga_b=_method_ga(method, pr['b'], ga_of(pr['b'])),
                        n_a=pr['n_a'], n_b=pr['n_b'], median_a=pr['med_a'], median_b=pr['med_b'],
                        statistic=pr['stat'], p_raw=pr['p_raw'], p_holm=float(p_adj),
                        significant=significant))
                matrices.setdefault((scenario, inst_key), []).append(dict(
                    config_label=config_label, method=method, metric=metric,
                    family=family, handlers=fam_handlers, cell=cell))
    return rows, matrices


def print_winloss_matrices(matrices, instances, scenarios):
    """Compact per (scenario, instance) win/tie/loss blocks -- one N x N
    handler grid per (config signature, method, metric) present, cell(a, b)
    read from row handler a's perspective against column handler b."""
    print('\n' + '=' * 118)
    print(f'Handler pairwise significance (Wilcoxon signed-rank, paired by seed, Holm-corrected, alpha={STATS_ALPHA})')
    print("W = row handler beats column (Holm-significant & better median); L = loses; T = tie / not significant")
    print('=' * 118)
    any_block = False
    for scenario in scenarios:
        for suite, pid in instances:
            inst_key = f'{suite}/pid{pid}'
            blocks = matrices.get((scenario, inst_key))
            if not blocks:
                continue
            for blk in blocks:
                any_block = True
                handlers = blk['handlers']
                row_w = max(20, max(len(h) for h in handlers) + 2)
                col_w = max(16, max(len(h) for h in handlers) + 2)
                print(f'\n{scenario} @ {inst_key}  [config={blk["config_label"]}]  method={blk["method"]}  '
                      f'metric={STATS_METRIC_LABEL[blk["metric"]]}  family={blk["family"]}')
                print('  ' + ' ' * row_w + ''.join(f'{h:>{col_w}s}' for h in handlers))
                for a in handlers:
                    row = ''.join(
                        f'{"-":>{col_w}s}' if a == b else f'{blk["cell"].get((a, b), "."):>{col_w}s}'
                        for b in handlers)
                    print(f'  {a:>{row_w}s}{row}')
    if not any_block:
        print('\n(no handler pair had enough seeds on both sides for any family; nothing to report)')
    print('=' * 118)


# ─── main ───────────────────────────────────────────────────────────────────

def plots_only(args):
    """Regenerate figures + LaTeX tables from the saved analysis artifacts
    (plot_data.pkl, constraint_trajectories.json, the two CSVs) -- no pkl
    scanning, no true-eval work. Seconds instead of the full analysis."""
    import pandas as pd
    cache_path = os.path.join(args.output_dir, 'plot_data.pkl')
    json_path  = os.path.join(args.output_dir, 'constraint_trajectories.json')
    csv_path   = os.path.join(args.output_dir, 'constraint_metrics.csv')
    stats_path = os.path.join(args.output_dir, 'handler_stats.csv')
    missing = [p for p in (cache_path, json_path, csv_path, stats_path)
               if not os.path.exists(p)]
    if missing:
        print('[analyse] --plots_only: missing artifacts (run the full analysis first):')
        for p in missing:
            print(f'  {p}')
        return 1
    with open(cache_path, 'rb') as fh:
        plot_cache = pickle.load(fh)
    with open(json_path) as fh:
        trajectories = json.load(fh)
    render_outputs(args.output_dir, trajectories, plot_cache,
                   pd.read_csv(csv_path), pd.read_csv(stats_path))
    return 0


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    instances = discover_instances(args.results_root, args.scenarios)
    if not instances:
        print(f'[analyse] no (suite, pid) instances with seed pkls under {args.results_root}; nothing to do.')
        return 0
    print(f'[analyse] discovered instances: {", ".join(f"{s}/pid{p}" for s, p in instances)}')

    csv_rows = []
    stat_records = []   # per-run rows feeding compute_handler_stats()
    trajectories = {'instances': {}, 'true_feasible_fraction': {}, 'runs': {}}
    # Everything the plot/table layer needs, persisted to plot_data.pkl so
    # --plots_only can rerun rendering without any true-eval work.
    plot_cache = {'instances': {}, 'refs': {}, 'fronts': {}, 'fronts_c': {}, 'fronts_ic': {}}
    warned_misplaced = set()   # one misplaced-pkl warning per directory

    for suite, pid in instances:
        inst_key = f'{suite}/pid{pid}'
        if (suite, pid) not in THRESHOLDS:
            print(f'[analyse] {inst_key}: no THRESHOLDS entry in run_constraint.py; skipping instance.')
            continue

        benchmark = get_benchmark(suite, pid)
        obj_names = get_obj_names(suite, pid)
        n_space   = space_size(benchmark)
        enumerable = n_space <= args.enum_limit

        cache = {}
        if enumerable:
            print(f'[analyse] {inst_key}: enumerating full space '
                  f'(lb={list(benchmark.search_space.lb)}, ub={list(benchmark.search_space.ub)}, '
                  f'|space|={n_space}) ...')
            X_cloud, F_cloud = enumerate_full_space(benchmark)
            cloud_label = f'enumerated ({len(X_cloud)})'
            print(f'[analyse] {inst_key}: enumerated {len(X_cloud)} finite architectures.')
        else:
            print(f'[analyse] {inst_key}: |space|={float(n_space):.3e} > enum_limit={args.enum_limit} '
                  f'-- NON-ENUMERABLE: M7 IGD+/HV-ratio n/a; attainable cloud from a '
                  f'fixed-seed {CLOUD_SAMPLE_N} random sample (approximation).')
            X_cloud, F_cloud = sample_cloud(benchmark)
            cloud_label = f'sampled ({CLOUD_SAMPLE_N})'
        for x, f in zip(X_cloud, F_cloud):        # seed the shared re-eval cache
            cache[tuple(int(v) for v in x)] = np.asarray(f, dtype=float)

        trajectories['instances'][inst_key] = dict(
            enumerable=enumerable, space_size=float(n_space),
            n_cloud=len(X_cloud), cloud_label=cloud_label)
        plot_cache['instances'][inst_key] = dict(
            enumerable=enumerable, cloud_label=cloud_label,
            obj_names=list(obj_names), F_cloud=np.asarray(F_cloud, dtype=float))

        # ── per-scenario discovery + analysis ──────────────────────────────────
        for scenario in args.scenarios:
            runs_by_method = {}
            for method in args.methods:
                runs_by_method[method] = discover_runs(
                    args.results_root, scenario, suite, pid, method, args.seeds)
            if not any(runs_by_method.values()):
                print(f'[analyse] {scenario}/{inst_key}: no seed pkls found; skipping.')
                continue

            # Reference data (true feasible front / HV) built lazily, once
            # per distinct config SIGNATURE actually present -- exactly one
            # in this single-config campaign (see module docstring). None
            # marks a signature whose objectives/constraint could not be
            # resolved for this instance (skip its rows, warn once).
            ref_by_sig, kinds_by_sig = {}, {}
            fronts_by_sig, fronts_c_by_sig, fronts_ic_by_sig = {}, {}, {}

            for method, runs in runs_by_method.items():
                if not runs:
                    print(f'[analyse] {scenario}/{inst_key}/{method}: no seed pkls found; skipping.')
                    continue

                for seed, handler, path, config_kind in runs:
                    try:
                        with open(path, 'rb') as fh:
                            data = pickle.load(fh)
                    except Exception as exc:
                        print(f'  [analyse] WARN: failed to load {path} '
                              f'(partial/mid-write?): {exc}; skipping run.')
                        continue

                    # A pkl whose own record names a different instance or
                    # scenario than the directory it sits in is a misplaced
                    # copy (e.g. a download rsync mishap): analysing it here
                    # would mix another instance's data into this group.
                    # Skip it -- the authoritative copy lives at the path its
                    # meta describes.
                    _meta = data.get('meta')
                    if _meta is not None and (
                            (_meta.get('suite'), _meta.get('pid')) != (suite, pid)
                            or _meta.get('scenario') != scenario):
                        _dir = os.path.dirname(path)
                        if _dir not in warned_misplaced:
                            warned_misplaced.add(_dir)
                            print(f"  [analyse] WARN: misplaced pkl(s) under {_dir}: meta says "
                                  f"{_meta.get('scenario')}/{_meta.get('suite')}/pid{_meta.get('pid')} "
                                  f"-- skipped (authoritative copies live at their own path).")
                        continue

                    resolved = _run_config(scenario, suite, pid, data.get('meta'))
                    if resolved is None:
                        print(f'  [analyse] WARN: {scenario}/{inst_key} ({config_kind}): no resolvable '
                              f'config for {path}; skipping run.')
                        continue
                    sig = _config_signature(resolved)
                    kinds_by_sig.setdefault(sig, set()).add(config_kind)

                    if sig not in ref_by_sig:
                        ref_by_sig[sig] = _build_ref(scenario, inst_key, config_kind, resolved,
                                                      F_cloud, cloud_label, enumerable, suite, pid)
                    ref = ref_by_sig[sig]
                    if ref is None:
                        continue   # objectives/constraint not resolvable -- already warned
                    obj_indices = ref['obj_indices']
                    constr_indices, thresholds = ref['constr_indices'], ref['thresholds']
                    true_front, hv_true, igd_ind = ref['true_front'], ref['hv_true'], ref['igd_ind']

                    m = analyse_run(data, obj_indices, constr_indices, thresholds, benchmark, cache)

                    hv_run = _hv(m['final_front'], REF_POINT)
                    if enumerable and hv_true > 0:
                        hv_ratio = hv_run / hv_true
                        igd_plus = (float(igd_ind(np.asarray(m['final_front'], dtype=float)))
                                    if len(m['final_front']) > 0 else float('nan'))
                    else:
                        hv_ratio = float('nan')
                        igd_plus = float('nan')

                    label = f'{method}:{handler}'
                    inner_ga = (data.get('meta') or {}).get('inner_ga')
                    method_ga = _method_ga(method, handler, inner_ga)

                    csv_rows.append(dict(
                        scenario=scenario, suite=suite, pid=pid, method=method,
                        handler=handler, config=config_kind, seed=seed,
                        n_gen=m['n_gen'],
                        m2_evals_to_first_feasible=m['m2_evals_to_first_feasible'],
                        m2_censored=m['m2_censored'], m2_n_evaluated=m['m2_n_evaluated'],
                        m3_final_infeasible_waste=m['m3_final_cumulative'],
                        exact_waste=m['exact_waste'],
                        m4_final_front_size=m['m4_final'],
                        m6_min_slack=m['m6_min_slack'], m6_median_slack=m['m6_median_slack'],
                        m7_igd_plus=igd_plus, m7_hv_ratio=hv_ratio,
                        m7_hv_run=hv_run, m7_hv_true=hv_true,
                        m8_best_err=m['m8_best_err'],
                        m8_best_err_slack=m['m8_best_err_slack'],
                        m8_err_true=ref['best_err_true'],
                        soft_hv=m['soft_hv'],
                        m7_enumerable=enumerable,
                        method_ga=method_ga,
                    ))
                    stat_records.append(dict(
                        scenario=scenario, inst_key=inst_key, suite=suite, pid=pid,
                        sig=sig, config_kind=config_kind, method=method, handler=handler,
                        seed=seed, enumerable=enumerable, inner_ga=inner_ga,
                        m7_hv_run=hv_run, m7_hv_ratio=hv_ratio, soft_hv=m['soft_hv'],
                        m8_best_err=m['m8_best_err'],
                    ))
                    trajectories['runs'] \
                        .setdefault(scenario, {}).setdefault(inst_key, {}) \
                        .setdefault((sig, label), {})[str(seed)] = dict(
                            m1_ratio=m['m1_ratio'], m3_cumulative=m['m3_cumulative'],
                            m3_per_gen=m['m3_per_gen'], m4_front_size=m['m4_front_size'],
                            m5_mean_violation=m['m5_mean_violation'],
                            hv_traj=m['hv_traj'],
                            exact_waste_traj=m['exact_waste_traj'],
                            m7_hv_ratio_traj=[
                                (hv / hv_true if enumerable and hv_true > 0 else float('nan'))
                                for hv in m['hv_traj']],
                        )
                    fronts_by_sig.setdefault(sig, {}).setdefault((method, handler), []) \
                        .append(m['final_front'])
                    fronts_c_by_sig.setdefault(sig, {}).setdefault((method, handler), []) \
                        .append(m['final_front_c'])
                    fronts_ic_by_sig.setdefault(sig, {}).setdefault((method, handler), []) \
                        .append(m['final_infeas_c'])

            n_sigs_present = len({s for s, r in ref_by_sig.items() if r is not None and fronts_by_sig.get(s)})
            multi_config = n_sigs_present > 1
            # Re-key trajectories['runs'] labels now that multi_config is known
            # (suffix only when more than one config's rows actually coexist
            # here, so the common single-config case keeps its original,
            # un-suffixed label -- see module docstring).
            by_inst = trajectories['runs'].get(scenario, {}).get(inst_key, {})
            for (sig, label), seed_dict in list(by_inst.items()):
                del by_inst[(sig, label)]
                kinds = kinds_by_sig.get(sig, set())
                final_label = label if not multi_config else \
                    f'{label} [{"+".join(sorted(k.replace(":", "-") for k in kinds))}]'
                by_inst[final_label] = seed_dict
            if scenario in trajectories['runs'] and inst_key in trajectories['runs'][scenario]:
                trajectories['runs'][scenario][inst_key] = by_inst

            for sig, ref in ref_by_sig.items():
                if ref is None:
                    continue
                for kind in kinds_by_sig.get(sig, ()):
                    trajectories['true_feasible_fraction'].setdefault(scenario, {}) \
                        .setdefault(inst_key, {})[kind] = ref['frac']

            # Plot/table inputs (single-config campaign: the first resolvable
            # ref is THE ref; fronts merge across sigs, of which there is one).
            ref_main = next((r for r in ref_by_sig.values() if r is not None), None)
            if ref_main is not None:
                plot_cache['refs'].setdefault(scenario, {})[inst_key] = dict(
                    obj_indices=ref_main['obj_indices'], constr_index=ref_main['constr_index'],
                    threshold=ref_main['threshold'],
                    constr_indices=ref_main['constr_indices'],
                    thresholds=ref_main['thresholds'], cfg=ref_main['cfg'],
                    true_front=ref_main['true_front'], true_front_c=ref_main['true_front_c'],
                    frac=ref_main['frac'])
                for cache_key, by_sig in (('fronts', fronts_by_sig),
                                          ('fronts_c', fronts_c_by_sig),
                                          ('fronts_ic', fronts_ic_by_sig)):
                    merged = {}
                    for by_mh in by_sig.values():
                        for mh, fr in by_mh.items():
                            merged.setdefault(mh, []).extend(fr)
                    if merged:
                        plot_cache[cache_key].setdefault(scenario, {})[inst_key] = merged

    # ── write CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'constraint_metrics.csv')
    fieldnames = ['scenario', 'suite', 'pid', 'method', 'handler', 'config', 'seed', 'n_gen',
                  'm2_evals_to_first_feasible', 'm2_censored', 'm2_n_evaluated',
                  'm3_final_infeasible_waste', 'exact_waste', 'm4_final_front_size',
                  'm6_min_slack', 'm6_median_slack',
                  'm7_igd_plus', 'm7_hv_ratio', 'm7_hv_run', 'm7_hv_true',
                  'm8_best_err', 'm8_best_err_slack', 'm8_err_true', 'soft_hv',
                  'm7_enumerable', 'method_ga']
    import csv
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'\n[analyse] wrote {len(csv_rows)} rows -> {csv_path}')

    # ── handler significance tests + win/tie/loss ──────────────────────────
    stats_rows, winloss_matrices = compute_handler_stats(stat_records)
    stats_path = os.path.join(args.output_dir, 'handler_stats.csv')
    stats_fieldnames = ['scenario', 'suite', 'pid', 'config', 'method', 'metric', 'family',
                         'handler_a', 'handler_b', 'method_ga_a', 'method_ga_b',
                         'n_a', 'n_b', 'median_a', 'median_b',
                         'statistic', 'p_raw', 'p_holm', 'significant']
    with open(stats_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=stats_fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f'[analyse] wrote {len(stats_rows)} handler comparison rows -> {stats_path}')

    # ── write JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, 'constraint_trajectories.json')
    with open(json_path, 'w') as fh:
        json.dump(trajectories, fh, indent=2)
    print(f'[analyse] wrote per-generation trajectories -> {json_path}')

    # ── plot cache (for --plots_only) + plots + LaTeX tables ───────────────
    cache_path = os.path.join(args.output_dir, 'plot_data.pkl')
    with open(cache_path, 'wb') as fh:
        pickle.dump(plot_cache, fh)
    print(f'[analyse] wrote plot cache -> {cache_path}')

    import pandas as pd
    render_outputs(args.output_dir, trajectories, plot_cache,
                   pd.DataFrame(csv_rows), pd.DataFrame(stats_rows))

    # ── stdout summary (mean over seeds) ────────────────────────────────────
    # One block per (scenario, instance, config) actually present -- config
    # is always 'c2' in this campaign; see module docstring.
    print('\n' + '=' * 118)
    print('Feasibility-region metrics summary (mean over seeds where n_seeds > 1)')
    print('=' * 118)
    for scenario in args.scenarios:
        for suite, pid in instances:
            inst_key = f'{suite}/pid{pid}'
            rows_inst = [r for r in csv_rows if r['scenario'] == scenario
                         and r['suite'] == suite and r['pid'] == pid]
            if not rows_inst:
                continue
            enumerable = trajectories['instances'][inst_key]['enumerable']
            for config_kind in sorted({r['config'] for r in rows_inst}):
                cfg = _path_config(scenario, suite, pid)
                rows_cfg = [r for r in rows_inst if r['config'] == config_kind]
                if cfg is None or not rows_cfg:
                    continue
                frac = trajectories['true_feasible_fraction'].get(scenario, {}) \
                    .get(inst_key, {}).get(config_kind, float('nan'))
                frac_kind = 'true (enumerated)' if enumerable else 'sampled (info only)'
                m7_note = '' if enumerable else '   [M7: n/a (non-enumerable)]'
                print(f'\nScenario {scenario} @ {inst_key}  [config={config_kind}]  (mode={cfg["mode"]}, '
                      f'objectives={"/".join(cfg["obj_metrics"])}, constr={cfg["constr_metric"]})  '
                      f'feasible fraction={frac:.4f} [{frac_kind}]{m7_note}')
                default_handler = 'h4-cdp' if cfg['mode'] == 'hard' else 'h2-penalty'
                handlers_present = sorted({r['handler'] for r in rows_cfg}, key=_handler_sort_key)
                if len(handlers_present) > 1:
                    print(f'  (default handler: {default_handler}; random runs only its '
                          f'default and h1-rejection -- selection-free otherwise)')
                print(f'  {"method:handler":>28s}  {"n":>2s}  {"M2(evals->feas)":>16s}  {"M3-final(waste)":>16s}  '
                      f'{"M4-final(front)":>16s}  {"M6 min/med slack":>18s}  {"M7 IGD+":>12s}  {"M7 HV-ratio":>12s}  '
                      f'{"M8 bestErr/slack":>18s}  {"soft-HV":>10s}')
                for method, handler in ((m, h) for m in args.methods for h in handlers_present):
                    rows = [r for r in rows_cfg if r['method'] == method and r['handler'] == handler]
                    if not rows:
                        continue
                    name = method if handler == default_handler else f'{method}:{handler}'
                    n = len(rows)
                    m2_vals = [r['m2_evals_to_first_feasible'] for r in rows]
                    m2_finite = [v for v in m2_vals if np.isfinite(v)]
                    n_censored = sum(1 for r in rows if r['m2_censored'])
                    if m2_finite:
                        m2_str = f'{np.mean(m2_finite):.1f}' + (f' ({n_censored}/{n} cens.)' if n_censored else '')
                    else:
                        m2_str = f'inf ({n_censored}/{n} cens.)'
                    m3_mean = np.nanmean([r['m3_final_infeasible_waste'] for r in rows])
                    m4_mean = np.nanmean([r['m4_final_front_size'] for r in rows])
                    m6_min_mean = np.nanmean([r['m6_min_slack'] for r in rows])
                    m6_med_mean = np.nanmean([r['m6_median_slack'] for r in rows])
                    if enumerable:
                        m7_igd_str = f'{np.nanmean([r["m7_igd_plus"] for r in rows]):>12.4f}'
                        m7_hvr_str = f'{np.nanmean([r["m7_hv_ratio"] for r in rows]):>12.4f}'
                    else:
                        m7_igd_str = f'{"n/a":>12s}'
                        m7_hvr_str = f'{"n/a":>12s}'
                    soft_hv_mean = np.nanmean([r['soft_hv'] for r in rows])
                    m8_mean  = np.nanmean([r['m8_best_err'] for r in rows])
                    m8s_mean = np.nanmean([r['m8_best_err_slack'] for r in rows])
                    print(f'  {name:>28s}  {n:>2d}  {m2_str:>16s}  {m3_mean:>16.4f}  '
                          f'{m4_mean:>16.1f}  {m6_min_mean:>8.4f}/{m6_med_mean:<8.4f}  '
                          f'{m7_igd_str}  {m7_hvr_str}  {m8_mean:>9.4f}/{m8s_mean:<7.3f}  '
                          f'{soft_hv_mean:>10.4f}')
    print('=' * 118)

    print_winloss_matrices(winloss_matrices, instances, args.scenarios)

    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_root', default=os.path.join('results', 'constraint2'),
                    help='Root written by run_constraint.py. Point at a dedicated '
                         'smoke-test folder when testing -- never analyse test '
                         'data written into results/ (see CLAUDE.md).')
    p.add_argument('--scenarios', nargs='+', default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='Seed subset (default: discover every seed_*.pkl present).')
    p.add_argument('--output_dir', default=None,
                    help='Default: {results_root}/analysis.')
    p.add_argument('--enum_limit', type=int, default=200_000,
                    help='Max search-space cardinality for exhaustive M7 enumeration; '
                         'larger spaces get M7 = n/a and a sampled attainable cloud.')
    p.add_argument('--self_test', action='store_true',
                    help='Run the attainment-surface unit checks and exit.')
    p.add_argument('--plots_only', action='store_true',
                    help='Regenerate figures + LaTeX tables from the saved '
                         'analysis artifacts in --output_dir (plot_data.pkl, '
                         'trajectories JSON, CSVs) without re-analysing pkls.')
    args = p.parse_args()
    if args.self_test:
        _self_test_attainment()
        sys.exit(0)
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_root, 'analysis')
    if args.plots_only:
        sys.exit(plots_only(args))
    sys.exit(main(args))
