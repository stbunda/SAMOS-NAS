"""experiments/constraint2/analyse_constraint.py --- Post-hoc feasibility-region
metrics for the S1-S4 constraint-scenario campaign (no re-runs needed:
everything is recomputed from the saved archives).

ADAPTATION PENDING for the constraint2 campaign: this is a verbatim copy of
experiments/constraint_deprecated/analyse_constraint.py and still assumes
the OLD result layout (objective-tag path level, legacy config maps, the
s4-as-view-of-s2 handling, per-scenario objective sets, the old THRESHOLDS
import). The constraint2 layout is
{results_root}/{scenario}/{suite}/pid{pid}/{budget}/{method}/{handler}/
seed_N.pkl with self-describing meta (config='c2', mode/gate,
design_feasible_fraction) and s4 as a physical run set. Do not run this
against results/constraint2 until the discovery/config layer is rewritten;
the metric computations, soft-HV, waste columns, and the seed-paired
Wilcoxon signed-rank machinery all carry over unchanged.

Consumes the per-seed pickles written by ``run_constraint.py``, discovering
BOTH output layouts it has ever used:

  tagged (current)  {results_root}/{scenario}/{suite}/pid{pid}/{objtag}/{budget}/{method}/{handler}/seed_*.pkl
  untagged (legacy) {results_root}/{scenario}/{suite}/pid{pid}/{budget}/{method}/{handler}/seed_*.pkl

Each is a ``FeasibilityAwareEvoxBenchCallback.data`` dict with keys
``var_pop``, ``obj_pop``, ``var_archive``, ``obj_archive``,
``test_obj_archive``, ``indicators``, ``n_feasible``, ``n_total``, ``time``,
plus (tagged pkls only) a ``meta`` dict self-describing the run's exact
config (see ``_run_config()``). Seven feasibility-region metrics are computed
entirely offline -- no runner changes, no re-runs. Every (suite, pid)
instance found under ``results_root`` is analysed; instances without a
resolvable threshold, or with missing/partial data, are skipped with a
message rather than failing.

Objectives are per-scenario, not a single global pair: the tagged layout's
config comes from the pkl's own ``meta`` dict when present (cross-checked
against the path-derived objtag/scenario, warning on any mismatch) or,
absent ``meta``, from ``run_constraint.SCENARIOS`` (current definitions);
the untagged layout's config comes from ``LEGACY_CONFIG`` below (what those
paths were written under before objectives became per-scenario). A run's
config SIGNATURE (objectives + constraint metric + mode) decides whether it
shares its reference data / attainment plot with other runs of the same
scenario+instance: s1's legacy and current definitions coincide (their rows
merge, e.g. into one attainment plot), s2/s3/s4's legacy definitions use
different objectives and are always kept in their own group, distinguished
by a ``config`` column ('r3' tagged / 'legacy' untagged / 'view:s2' -- see
below) in the CSV and by a label suffix in trajectories/plots, so nothing is
ever averaged or plotted across incompatible objective spaces silently.

s4 has no physical pkls of its own (run_constraint.py refuses to write
them): it is an analysis VIEW of s2's tagged runs (identical objectives,
constraint, threshold, handler set -- only the hard/soft framing differs),
so this script additionally reads s2's tagged pkls when asked for s4,
labeling them ``config='view:s2'``. Legacy s4 pkls, if any exist from
before that decision, are read too and kept in their own group as usual.

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
    8**5 = 32768 and c10mop/pid3 = 8**5 = 32768 qualify, everything else does
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
between every pair of handlers, on final feasible HV (``m7_hv_run``),
HV-ratio (enumerable instances only), and ``soft_hv``. Runs sharing a seed
share their RNG streams and (for the same method) their evaluated DOE
(run_constraint.py seeds np.random and minimize identically per seed), so
seeds are matched blocks: the paired test removes between-seed variance
(e.g. a lucky DOE lifting every handler that seed) that an unpaired
rank-sum would pool into noise. Only seeds present with finite values
on BOTH sides pair up; a pair with fewer than 5 common seeds is skipped
(noted, excluded from that family's Holm correction) rather than tested.
All-zero difference vectors (byte-identical runs, e.g. replicated random
rows) are p = 1.0 ties by definition (scipy's wilcoxon rejects them). Raw
p-values are Holm-corrected within each (scenario, instance, config
signature, method, metric) family. A comparison counts as a win for the
handler with the better (higher) median over the COMMON seeds when its
Holm-adjusted p < 0.05, otherwise a tie; only handler pairs sharing the
same method and config signature are ever compared (never across objective
spaces).

Attainment plots
----------------
Per (scenario x instance x config), ``{scenario}_{suite}_pid{pid}[_{config
suffix}]_attainment.png`` in ``{output_dir}/plots`` (the suffix, e.g.
'_legacy' or '_view-s2', only appears when more than one config's rows exist
for that scenario+instance -- the common single-config case keeps its
original, un-suffixed filename): the attainable objective cloud in that
config's own 2-objective true-eval plane, split feasible/infeasible (from
the exhaustive enumeration where enumerable, else a fixed-seed
[np.random.seed(0)] 10k uniform random sample, labeled 'sampled (10k)' since
it is an approximation); the true feasible Pareto front as a reference
staircase (enumerable only); and each method's empirical attainment surface
of FEASIBLE solutions as a minimization staircase -- with 1 seed that is the
final feasible ND front, with n > 1 seeds the median (ceil(n/2)-out-of-n)
attainment surface with a shaded best/worst band. ``--self_test`` runs a
unit-style check of the multi-seed attainment computation on synthetic 2-D
points and exits.

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
                                          Mann-Whitney-U comparison (raw p, Holm
                                          p, medians, significance flag)
  {output_dir}/plots/{scenario}_{suite}_pid{pid}_feasibility_ratio.png
  {output_dir}/plots/{scenario}_{suite}_pid{pid}_hv_ratio.png   (enumerable) /
                     ..._feasible_hv.png                        (non-enumerable)
  {output_dir}/plots/{scenario}_{suite}_pid{pid}[_{config suffix}]_attainment.png
  stdout: plain-text summary table (mean over seeds where n_seeds > 1), one
          block per (scenario, instance, config) actually present, followed
          by a per (scenario, instance) win/tie/loss matrix block from the
          significance tests above

Examples
--------
  python experiments/constraint/analyse_constraint.py
  python experiments/constraint/analyse_constraint.py --self_test
  python experiments/constraint/analyse_constraint.py --results_root /tmp/smoke --scenarios s1 s2
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
# so numbers are guaranteed comparable. Objectives are per-config now (see
# LEGACY_CONFIG / _run_config below), never a single global index pair.
from run_constraint import METHODS, SCENARIOS, THRESHOLDS

# Same ref-point convention as FeasibilityAwareEvoxBenchCallback: every
# scenario config (current or legacy) picks exactly 2 objectives.
REF_POINT = np.ones(2) * 1.05

# Legacy (pre-instance-axis) scenario definitions -- what the untagged
# {scenario}/{suite}/pid{pid}/{budget}/... paths were written under, before
# objectives became per-scenario config. s1's legacy definition is IDENTICAL
# to its current SCENARIOS entry (same objectives/constraint/mode), so those
# rows share one reference/plot group automatically; s2, s3, s4's legacy
# definitions used {Err, FLOPs} objectives where the current ones use
# {Err, #Params}, so they resolve to a DIFFERENT config signature and are
# always kept in their own group -- never averaged or plotted against
# current-config rows (see _config_signature).
LEGACY_CONFIG = {
    's1': dict(obj_metrics=('Err.', 'FLOPs'), constr_metric='#Params', mode='hard'),
    's2': dict(obj_metrics=('Err.', 'FLOPs'), constr_metric='Latency', mode='hard'),
    's3': dict(obj_metrics=('Err.', 'FLOPs'), constr_metric='#Params', mode='soft'),
    's4': dict(obj_metrics=('Err.', 'FLOPs'), constr_metric='Latency', mode='soft'),
}


def _cfg_for_kind(scenario, config_kind):
    """cfg dict (obj_metrics/constr_metric/mode) for one discovered run,
    given which layout/source it came from. 'r3' and 'view:s2' both read the
    current SCENARIOS definition (a view is, by construction, the same
    config as what it is a view of); 'legacy' reads LEGACY_CONFIG."""
    if config_kind == 'legacy':
        return LEGACY_CONFIG.get(scenario)
    return SCENARIOS.get(scenario)


def _config_signature(cfg):
    """Identity used to decide whether two configs are the SAME experiment
    (their rows/plots merge) or DIFFERENT ones (kept apart)."""
    return (tuple(cfg['obj_metrics']), cfg['constr_metric'], cfg['mode'])


def _run_config(scenario, suite, pid, config_kind, meta):
    """Full config for one discovered run -- obj_metrics/constr_metric/mode/
    threshold -- preferring its own ``meta`` dict (new-phase
    pkls are self-describing, so a run's numbers never drift from a later
    change to SCENARIOS/THRESHOLDS) over the path-derived lookup, with a
    mismatch warning if both are available and disagree. Falls back to
    path + LEGACY_CONFIG/SCENARIOS + a THRESHOLDS lookup when ``meta`` is
    absent (legacy pkls, written before it existed) or its constraint metric
    has no threshold for this instance. None if nothing usable resolves.

    'view:s2' is a deliberate exception to the mismatch check: that pkl's own
    meta was written under scenario s2 (mode='hard'), while the s4 VIEW
    reinterprets the identical physical run (same objectives/constraint/
    threshold, see run_constraint.py's --scenario s4 refusal message) under
    s4's soft framing -- so 'mode' alone is overridden from SCENARIOS['s4']
    rather than compared against meta."""
    path_cfg = _cfg_for_kind(scenario, config_kind)
    if meta is not None:
        meta_cfg = dict(obj_metrics=tuple(meta['obj_metrics']), constr_metric=meta['constr_metric'],
                         mode=meta['mode'], threshold=float(meta['threshold']))
        if config_kind == 'view:s2':
            if path_cfg is not None:
                meta_cfg['mode'] = path_cfg['mode']
            return meta_cfg
        if path_cfg is not None and _config_signature(meta_cfg) != _config_signature(path_cfg):
            print(f"  [analyse] WARN: {scenario} ({config_kind}): pkl meta config "
                  f"{_config_signature(meta_cfg)} disagrees with the path-derived config "
                  f"{_config_signature(path_cfg)}; trusting meta (the pkl's own record).")
        return meta_cfg
    if path_cfg is None:
        return None
    if path_cfg['constr_metric'] not in THRESHOLDS.get((suite, pid), {}):
        return None
    return dict(path_cfg, threshold=THRESHOLDS[(suite, pid)][path_cfg['constr_metric']])


def _method_ga(method, handler, inner_ga=None):
    """Method+inner-GA label: 'random' for method random; otherwise
    f'{method}-sms' when the inner GA is SMS-EMOA (handler in ('b0-as-obj',
    'h4-cdp-sms')), else f'{method}-nsga2' (NSGA-II, the samos default;
    handler 'b0-nsga2' is also NSGA-II). *inner_ga* ('sms'/'nsga2'), when a
    pkl's own meta carries it, overrides the handler rule -- present on
    newer pkls only, so legacy pkls always fall back to the handler rule."""
    if method == 'random':
        return 'random'
    if inner_ga in ('sms', 'nsga2'):
        return f'{method}-{inner_ga}'
    return f'{method}-sms' if handler in ('b0-as-obj', 'h4-cdp-sms') else f'{method}-nsga2'


# Okabe-Ito hues (colorblind-safe), fixed assignment per method -- never cycled.
METHOD_COLOURS = {
    'random':      '#0072B2',
    'samos':       '#D55E00',
    'samos-cheap': '#009E73',
}
CLOUD_FEASIBLE_COLOUR   = '#b8d4ea'   # light cool
CLOUD_INFEASIBLE_COLOUR = '#f4c7b8'   # light warm

CLOUD_SAMPLE_N    = 10_000
CLOUD_SAMPLE_SEED = 0

# ─── significance-testing constants ────────────────────────────────────────
STATS_ALPHA     = 0.05
STATS_MIN_SEEDS = 5    # per side; families below this are skipped, not tested
STATS_METRICS   = ('m7_hv_run', 'm7_hv_ratio', 'soft_hv')
STATS_METRIC_LABEL = {'m7_hv_run': 'feasible HV', 'm7_hv_ratio': 'HV-ratio', 'soft_hv': 'soft-HV'}


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


def true_feasible_front(F_all, obj_indices, constr_index, threshold):
    """(feasible_fraction, feasible ND front in {obj_indices} space)."""
    feasible_mask = F_all[:, constr_index] <= threshold
    frac = float(feasible_mask.mean())
    F_feas = F_all[feasible_mask][:, obj_indices]
    if len(F_feas) == 0:
        return frac, np.empty((0, len(obj_indices)))
    nd_idx = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)
    return frac, F_feas[nd_idx]


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

def analyse_run(data: dict, obj_indices, constr_index, threshold, benchmark, cache: dict):
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
        metric = F_fin[:, constr_index]
        infeasible = metric > threshold
        if not infeasible.any():
            m5_mean_violation.append(float('nan'))
        else:
            m5_mean_violation.append(float(np.mean((metric[infeasible] - threshold) / threshold)))

    # M2: 1-based index into the FINAL var_archive (evaluation-order proxy,
    # see module docstring), among finite entries only (matches n_total's own
    # finite-row convention).
    if n_gen > 0:
        F_last, finite_last = full_evals_per_gen[-1]
        F_last_fin = F_last[finite_last]
        feasible_seq = F_last_fin[:, constr_index] <= threshold
        if feasible_seq.any():
            m2_evals_to_first_feasible = int(np.argmax(feasible_seq)) + 1
            m2_censored = False
        else:
            m2_evals_to_first_feasible = float('inf')
            m2_censored = True
        m2_n_evaluated = int(len(F_last_fin))
    else:
        F_last_fin = np.empty((0, benchmark.evaluator.n_objs))
        m2_evals_to_first_feasible = float('inf')
        m2_censored = True
        m2_n_evaluated = 0

    # M6: boundary slack over the FINAL feasible ND front.
    final_front = np.asarray(test_obj_archive[-1]) if n_gen > 0 and len(test_obj_archive[-1]) > 0 \
        else np.empty((0, len(obj_indices)))
    feasible_mask_last = (F_last_fin[:, constr_index] <= threshold) if len(F_last_fin) else \
        np.zeros(0, dtype=bool)
    if len(final_front) > 0 and feasible_mask_last.any():
        F_feas_last = F_last_fin[feasible_mask_last]
        obj_feas_last = F_feas_last[:, obj_indices]
        nd_idx = NonDominatedSorting().do(obj_feas_last, only_non_dominated_front=True)
        slack = (threshold - F_feas_last[nd_idx, constr_index]) / threshold
        m6_min_slack, m6_median_slack = float(np.min(slack)), float(np.median(slack))
    else:
        # No re-evaluated feasible point in the final archive (a run may end
        # all-infeasible, or the callback-time and re-evaluated feasibility
        # can disagree at the boundary).
        m6_min_slack = m6_median_slack = float('nan')

    # Soft-HV: violation-graded HV of the FULL final archive (feasible
    # points unchanged, infeasible points pushed away from the origin in
    # BOTH objectives by their normalized constraint violation) -- defined
    # for every run regardless of scenario mode (see module docstring).
    if len(F_last_fin) > 0:
        metric_last = F_last_fin[:, constr_index]
        violation = np.maximum(0.0, (metric_last - threshold) / threshold)
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
    else:
        exact_waste = float('nan')

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
        soft_hv=soft_hv,
        exact_waste=exact_waste,
        final_front=final_front,
        n_gen=n_gen,
    )


# ─── discovery / IO ─────────────────────────────────────────────────────────

def discover_instances(results_root, scenarios):
    """Every (suite, pid) with at least one seed pkl (tagged or legacy
    layout) under any of *scenarios*. s2 is always scanned in addition when
    s4 is requested, since s4's physical data lives under s2's tagged runs
    (see discover_s4_view_runs) -- otherwise --scenarios s4 alone would find
    nothing even though its view has data."""
    scan = set(scenarios)
    if 's4' in scan:
        scan.add('s2')
    found = set()
    for scenario in scan:
        for pid_dir in glob.glob(os.path.join(results_root, scenario, '*', 'pid*')):
            if not os.path.isdir(pid_dir):
                continue
            suite = os.path.basename(os.path.dirname(pid_dir))
            try:
                pid = int(os.path.basename(pid_dir)[len('pid'):])
            except ValueError:
                continue
            # Tagged: pid_dir/{objtag}/{budget}/{method}/{handler}/seed_*.pkl
            # (4 directory levels). Legacy: pid_dir/{budget}/{method}/
            # {handler}/seed_*.pkl (3 levels) -- objtag dirs are named
            # 'obj-*', budget dirs 'B<n>_P<n>', so a fixed wildcard depth
            # distinguishes the two layouts without ever cross-matching.
            has_tagged = bool(glob.glob(os.path.join(pid_dir, 'obj-*', '*', '*', '*', 'seed_*.pkl')))
            has_legacy = bool(glob.glob(os.path.join(pid_dir, '*', '*', '*', 'seed_*.pkl')))
            if has_tagged or has_legacy:
                found.add((suite, pid))
    return sorted(found)


def _parse_seed(fname):
    try:
        return int(fname[len('seed_'):-len('.pkl')])
    except ValueError:
        return None


def discover_runs(results_root, scenario, suite, pid, method, seeds):
    """(seed, handler, path, config_kind) tuples for one (scenario, suite,
    pid, method), across both output layouts run_constraint.py has ever
    used:
      config_kind='r3'     -- tagged layout, pid_dir/{objtag}/{budget}/{method}/{handler}/seed_N.pkl
      config_kind='legacy' -- untagged layout, pid_dir/{budget}/{method}/{handler}/seed_N.pkl
                               (pre-objective-tag paths, read-only history,
                               never written again -- see run_constraint.py).
    """
    pid_dir = os.path.join(results_root, scenario, suite, f'pid{pid}')
    out = []

    for path in sorted(glob.glob(os.path.join(pid_dir, 'obj-*', '*', method, '*', 'seed_*.pkl'))):
        rel = os.path.relpath(path, pid_dir).split(os.sep)
        if len(rel) != 5:
            continue
        _objtag, _budget, _method, handler, fname = rel
        seed = _parse_seed(fname)
        if seed is None or (seeds is not None and seed not in seeds):
            continue
        out.append((seed, handler, path, 'r3'))

    for path in sorted(glob.glob(os.path.join(pid_dir, '*', method, '*', 'seed_*.pkl'))):
        rel = os.path.relpath(path, pid_dir).split(os.sep)
        if len(rel) != 4:
            continue
        _budget, _method, handler, fname = rel
        seed = _parse_seed(fname)
        if seed is None or (seeds is not None and seed not in seeds):
            continue
        out.append((seed, handler, path, 'legacy'))

    return out


def discover_s4_view_runs(results_root, suite, pid, method, seeds):
    """s4's physical data is s2's tagged runs (identical objectives,
    constraint, threshold and handler set; only the hard/soft framing
    differs, and the shared feasibility indicator treats the threshold as a
    hard cutoff either way -- see run_constraint.py's --scenario s4
    refusal). Only the tagged layout qualifies; any legacy s4 pkls are
    discover_runs('s4', ...)'s own job, not this one."""
    return [(seed, handler, path, 'view:s2')
            for seed, handler, path, kind in discover_runs(results_root, 's2', suite, pid, method, seeds)
            if kind == 'r3']


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

def _base_method(label):
    """Strip a trajectory/CSV label back to its plain method name (labels can
    carry a ':{handler}' and/or ' [{config_kind}]' suffix) so color stays
    consistent per method regardless of those suffixes."""
    return label.split(':')[0].split(' [')[0]


def plot_trajectory(by_label, metric_key, ylabel, title, out_png, plt):
    fig, ax = plt.subplots(figsize=(6, 4))
    any_line = False
    for label, seeds_dict in by_label.items():
        series = [v[metric_key] for v in seeds_dict.values() if v.get(metric_key)]
        if not series:
            continue
        mean_series = _pad_mean(series)
        ax.plot(range(1, len(mean_series) + 1), mean_series,
                color=METHOD_COLOURS.get(_base_method(label)), label=label)
        any_line = True
    if not any_line:
        plt.close(fig)
        return False
    ax.set_xlabel('generation')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def plot_attainment(cloud_F, cloud_label, obj_indices, constr_index, threshold, true_front,
                    fronts_by_method, obj_names, title, out_png, plt):
    """One axes: feasible/infeasible attainable cloud, true feasible front
    staircase (if given), per-method feasible attainment staircases (median
    surface + best/worst band when n > 1 seeds). obj_indices are this
    config's own 2 benchmark columns (varies by scenario/config -- see
    module docstring)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    obj  = cloud_F[:, obj_indices]
    feas = cloud_F[:, constr_index] <= threshold
    ax.scatter(obj[~feas, 0], obj[~feas, 1], s=3, c=CLOUD_INFEASIBLE_COLOUR,
               alpha=0.5, linewidths=0, rasterized=True,
               label=f'infeasible ({cloud_label})')
    ax.scatter(obj[feas, 0], obj[feas, 1], s=3, c=CLOUD_FEASIBLE_COLOUR,
               alpha=0.5, linewidths=0, rasterized=True,
               label=f'feasible ({cloud_label})')

    if true_front is not None and len(true_front) > 0:
        tf = np.asarray(true_front, dtype=float)
        tf = tf[np.argsort(tf[:, 0], kind='stable')]
        ax.step(tf[:, 0], np.minimum.accumulate(tf[:, 1]), where='post',
                color='black', lw=1.5, label='true feasible front')

    for method in METHODS:                       # fixed order = fixed colors
        fronts = [np.asarray(f, dtype=float) for f in fronts_by_method.get(method, [])
                  if f is not None and len(f) > 0]
        if not fronts:
            continue
        colour = METHOD_COLOURS.get(method)
        if len(fronts) == 1:
            f = fronts[0][np.argsort(fronts[0][:, 0], kind='stable')]
            ax.step(f[:, 0], np.minimum.accumulate(f[:, 1]), where='post',
                    color=colour, lw=1.8, label=f'{method} (final feasible front)')
        else:
            n = len(fronts)
            k_med = math.ceil(n / 2)
            xs, surf = attainment_surfaces(fronts, ks=[1, k_med, n])
            to_nan = lambda a: np.where(np.isfinite(a), a, np.nan)
            ax.step(xs, to_nan(surf[k_med]), where='post', color=colour, lw=1.8,
                    label=f'{method} (median attainment, n={n})')
            ax.fill_between(xs, to_nan(surf[1]), to_nan(surf[n]), step='post',
                            color=colour, alpha=0.18, linewidth=0,
                            label=f'{method} (best-worst band)')

    ax.set_xlabel(f'{obj_names[obj_indices[0]]} (true-eval, norm.)')
    ax.set_ylabel(f'{obj_names[obj_indices[1]]} (true-eval, norm.)')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ─── reference (true feasible front / HV) per config signature ─────────────

def _build_ref(scenario, inst_key, config_kind, resolved, F_cloud, cloud_label, enumerable, suite, pid):
    """True feasible front / HV reference for one config signature, built
    once and reused by every run that shares it (see main()). Returns None
    (with a warning) if *resolved*'s objectives/constraint don't resolve to
    benchmark columns for this instance."""
    try:
        obj_indices  = [metric_index(suite, pid, m) for m in resolved['obj_metrics']]
        constr_index = metric_index(suite, pid, resolved['constr_metric'])
    except KeyError:
        print(f"  [analyse] WARN: {scenario}/{inst_key} ({config_kind}): objectives "
              f"{resolved['obj_metrics']} / constraint {resolved['constr_metric']!r} not "
              f"resolvable for this instance; skipping its rows.")
        return None

    threshold = resolved['threshold']
    frac, cloud_front = true_feasible_front(F_cloud, obj_indices, constr_index, threshold)
    if enumerable:
        # Sanity asserts (enumerable = exact ground truth only).
        assert len(cloud_front) > 0, (
            f'{scenario}/{inst_key} ({config_kind}): true feasible front is empty -- '
            f'threshold/constraint wiring is broken.')
        assert 0.3 < frac < 0.7, (
            f'{scenario}/{inst_key} ({config_kind}): enumerated feasible fraction {frac:.4f} far '
            f'from the ~50% design intent (T={threshold}, constr={resolved["constr_metric"]}) '
            f'-- check THRESHOLDS.md provenance.')
        true_front = cloud_front
        hv_true    = _hv(true_front, REF_POINT)
        igd_ind    = IGDPlus(true_front)
        print(f'[analyse] {scenario}/{inst_key} ({config_kind}): true feasible fraction = {frac:.4f} '
              f'({int(round(frac * len(F_cloud)))}/{len(F_cloud)}), true feasible ND-front size = '
              f'{len(true_front)}, HV(true front) = {hv_true:.4f}')
    else:
        true_front, hv_true, igd_ind = None, float('nan'), None
        print(f'[analyse] {scenario}/{inst_key} ({config_kind}): sampled feasible fraction = {frac:.4f} '
              f'(info only, {cloud_label}); M7 = n/a (non-enumerable).')

    return dict(cfg=resolved, obj_indices=obj_indices, constr_index=constr_index,
                threshold=threshold, frac=frac, true_front=true_front,
                hv_true=hv_true, igd_ind=igd_ind)


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


def compute_handler_stats(stat_records, alpha=STATS_ALPHA, min_seeds=STATS_MIN_SEEDS):
    """Pairwise two-sided Wilcoxon SIGNED-RANK, paired by seed (seed-matched
    runs share RNG streams and DOE, see module docstring),
    between every pair of handlers sharing the same (scenario, instance,
    config signature, method), on each of STATS_METRICS. Holm-corrected
    within each (scenario, instance, config signature, method, metric)
    family. Pairs with fewer than *min_seeds* common finite seeds are
    skipped (noted, excluded from the family's correction) rather than
    tested -- guards against a partial/still-downloading instance.

    Returns (rows, matrices):
      rows      -- list of dict rows for handler_stats.csv.
      matrices  -- {(scenario, inst_key): [block, ...]} where each block is
                   a dict(config_label, method, metric, handlers, cell) for
                   the stdout win/tie/loss printer; cell[(a, b)] is 'W'/'L'/
                   'T' from handler a's perspective against b.
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
        handlers = sorted(g['by_handler'])
        if len(handlers) < 2:
            continue
        config_label = '+'.join(sorted(k.replace(':', '-') for k in g['kinds']))
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

            pair_results = []
            for a, b in itertools.combinations(handlers, 2):
                va, vb = values[a], values[b]
                common = sorted(set(va) & set(vb))
                if len(common) < min_seeds:
                    print(f'  [analyse] handler-stats: {scenario}/{inst_key} config={config_label} '
                          f'method={method} metric={metric}: {a} (n={len(va)}) vs {b} (n={len(vb)}) '
                          f'-- fewer than {min_seeds} common seeds ({len(common)}); skipped (not tested).')
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
            for pr, p_adj in zip(pair_results, p_holm):
                significant = bool(p_adj < alpha)
                outcome_ab = ('W' if pr['med_a'] > pr['med_b'] else 'L') if significant else 'T'
                cell[(pr['a'], pr['b'])] = outcome_ab
                cell[(pr['b'], pr['a'])] = {'W': 'L', 'L': 'W', 'T': 'T'}[outcome_ab]
                # method_ga is handler-dependent (see _method_ga), and a and b
                # are two different handlers by construction here, so this
                # row gets one column per side rather than a single
                # 'method_ga' -- they differ exactly when the pair is an
                # inner-GA comparison (e.g. h4-cdp vs h4-cdp-sms).
                ga_a = g['by_handler'][pr['a']][0].get('inner_ga')
                ga_b = g['by_handler'][pr['b']][0].get('inner_ga')
                rows.append(dict(
                    scenario=scenario, suite=g['suite'], pid=g['pid'], config=config_label,
                    method=method, metric=metric, handler_a=pr['a'], handler_b=pr['b'],
                    method_ga_a=_method_ga(method, pr['a'], ga_a),
                    method_ga_b=_method_ga(method, pr['b'], ga_b),
                    n_a=pr['n_a'], n_b=pr['n_b'], median_a=pr['med_a'], median_b=pr['med_b'],
                    statistic=pr['stat'], p_raw=pr['p_raw'], p_holm=float(p_adj),
                    significant=significant))
            matrices.setdefault((scenario, inst_key), []).append(dict(
                config_label=config_label, method=method, metric=metric,
                handlers=handlers, cell=cell))
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
                      f'metric={STATS_METRIC_LABEL[blk["metric"]]}')
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
    attainment_jobs = []   # deferred until matplotlib import is known good
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

        # ── per-scenario discovery + analysis ──────────────────────────────────
        for scenario in args.scenarios:
            runs_by_method = {}
            for method in args.methods:
                runs = discover_runs(args.results_root, scenario, suite, pid, method, args.seeds)
                if scenario == 's4':   # s4's physical data is s2's tagged runs
                    runs = runs + discover_s4_view_runs(args.results_root, suite, pid, method, args.seeds)
                runs_by_method[method] = runs
            if not any(runs_by_method.values()):
                print(f'[analyse] {scenario}/{inst_key}: no seed pkls found; skipping.')
                continue

            # Reference data (true feasible front / HV) built lazily, once
            # per distinct config SIGNATURE actually present -- usually one;
            # s1 merges its legacy and current definitions into it since they
            # coincide, s2/s3/s4 keep a legacy signature apart from the
            # current one when the objectives differ (see module docstring).
            # None marks a signature whose objectives/constraint could not be
            # resolved for this instance (skip its rows, warn once).
            ref_by_sig, kinds_by_sig, fronts_by_sig = {}, {}, {}

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
                    # meta describes. (s4's view of s2 is the one documented
                    # scenario mismatch and is exempt.)
                    _meta = data.get('meta')
                    if _meta is not None and (
                            (_meta.get('suite'), _meta.get('pid')) != (suite, pid)
                            or (_meta.get('scenario') != scenario and config_kind != 'view:s2')):
                        _dir = os.path.dirname(path)
                        if _dir not in warned_misplaced:
                            warned_misplaced.add(_dir)
                            print(f"  [analyse] WARN: misplaced pkl(s) under {_dir}: meta says "
                                  f"{_meta.get('scenario')}/{_meta.get('suite')}/pid{_meta.get('pid')} "
                                  f"-- skipped (authoritative copies live at their own path).")
                        continue

                    resolved = _run_config(scenario, suite, pid, config_kind, data.get('meta'))
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
                    obj_indices, constr_index, threshold = ref['obj_indices'], ref['constr_index'], ref['threshold']
                    true_front, hv_true, igd_ind = ref['true_front'], ref['hv_true'], ref['igd_ind']

                    m = analyse_run(data, obj_indices, constr_index, threshold, benchmark, cache)

                    hv_run = _hv(m['final_front'], REF_POINT)
                    if enumerable and hv_true > 0:
                        hv_ratio = hv_run / hv_true
                        igd_plus = (float(igd_ind(np.asarray(m['final_front'], dtype=float)))
                                    if len(m['final_front']) > 0 else float('nan'))
                    else:
                        hv_ratio = float('nan')
                        igd_plus = float('nan')

                    default_handler = 'h4-cdp' if resolved['mode'] == 'hard' else 'h2-penalty'
                    label = method if handler == default_handler else f'{method}:{handler}'
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
                        soft_hv=m['soft_hv'],
                        m7_enumerable=enumerable,
                        method_ga=method_ga,
                    ))
                    stat_records.append(dict(
                        scenario=scenario, inst_key=inst_key, suite=suite, pid=pid,
                        sig=sig, config_kind=config_kind, method=method, handler=handler,
                        seed=seed, enumerable=enumerable, inner_ga=inner_ga,
                        m7_hv_run=hv_run, m7_hv_ratio=hv_ratio, soft_hv=m['soft_hv'],
                    ))
                    trajectories['runs'] \
                        .setdefault(scenario, {}).setdefault(inst_key, {}) \
                        .setdefault((sig, label), {})[str(seed)] = dict(
                            m1_ratio=m['m1_ratio'], m3_cumulative=m['m3_cumulative'],
                            m3_per_gen=m['m3_per_gen'], m4_front_size=m['m4_front_size'],
                            m5_mean_violation=m['m5_mean_violation'],
                            hv_traj=m['hv_traj'],
                            m7_hv_ratio_traj=[
                                (hv / hv_true if enumerable and hv_true > 0 else float('nan'))
                                for hv in m['hv_traj']],
                        )
                    # Attainment plots compare methods at their scenario-default
                    # handler; the handler axis is compared via the trajectory
                    # plots and the summary tables instead, keeping one readable
                    # staircase per method here.
                    if handler == default_handler:
                        fronts_by_sig.setdefault(sig, {}).setdefault(method, []).append(m['final_front'])

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

                fronts_by_method = fronts_by_sig.get(sig)
                if not fronts_by_method:
                    continue
                kinds = kinds_by_sig.get(sig, set())
                suffix = '' if not multi_config else \
                    '_' + '+'.join(sorted(k.replace(':', '-') for k in kinds))
                attainment_jobs.append(dict(
                    scenario=scenario, suite=suite, pid=pid, stem_suffix=suffix,
                    cloud_F=F_cloud, cloud_label=cloud_label,
                    obj_indices=ref['obj_indices'], constr_index=ref['constr_index'],
                    threshold=ref['threshold'], true_front=ref['true_front'],
                    fronts_by_method=fronts_by_method, obj_names=obj_names,
                    enumerable=enumerable, cfg=ref['cfg'], config_kinds=kinds))

    # ── write CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'constraint_metrics.csv')
    fieldnames = ['scenario', 'suite', 'pid', 'method', 'handler', 'config', 'seed', 'n_gen',
                  'm2_evals_to_first_feasible', 'm2_censored', 'm2_n_evaluated',
                  'm3_final_infeasible_waste', 'exact_waste', 'm4_final_front_size',
                  'm6_min_slack', 'm6_median_slack',
                  'm7_igd_plus', 'm7_hv_ratio', 'm7_hv_run', 'm7_hv_true', 'soft_hv',
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
    stats_fieldnames = ['scenario', 'suite', 'pid', 'config', 'method', 'metric',
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

    # ── plots ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot_dir = os.path.join(args.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Trajectory plots per (scenario, instance).
        for scenario, by_inst in trajectories['runs'].items():
            for inst_key, by_method in by_inst.items():
                suite, pid_s = inst_key.split('/pid')
                stem = f'{scenario}_{suite}_pid{pid_s}'
                enumerable = trajectories['instances'][inst_key]['enumerable']
                out = os.path.join(plot_dir, f'{stem}_feasibility_ratio.png')
                if plot_trajectory(by_method, 'm1_ratio',
                                    'feasibility ratio (n_feasible / n_total)',
                                    f'{scenario} {inst_key}: feasibility ratio', out, plt):
                    print(f'[analyse] wrote {out}')
                if enumerable:
                    out = os.path.join(plot_dir, f'{stem}_hv_ratio.png')
                    if plot_trajectory(by_method, 'm7_hv_ratio_traj',
                                        'HV-ratio (run / true feasible front)',
                                        f'{scenario} {inst_key}: HV-ratio', out, plt):
                        print(f'[analyse] wrote {out}')
                else:
                    out = os.path.join(plot_dir, f'{stem}_feasible_hv.png')
                    if plot_trajectory(by_method, 'hv_traj',
                                        'feasible HV (raw, ref=1.05 per obj)',
                                        f'{scenario} {inst_key}: feasible HV (no true front)', out, plt):
                        print(f'[analyse] wrote {out}')

        # Attainment plots (one per (scenario, instance, config signature);
        # stem_suffix is empty in the common single-config case, so those
        # keep their original, un-suffixed filename).
        for job in attainment_jobs:
            stem = f'{job["scenario"]}_{job["suite"]}_pid{job["pid"]}{job["stem_suffix"]}'
            out_png = os.path.join(plot_dir, f'{stem}_attainment.png')
            cfg = job['cfg']
            kinds_note = '' if not job['stem_suffix'] else f'  [{"/".join(sorted(job["config_kinds"]))}]'
            title = (f'{job["scenario"]} {job["suite"]}/pid{job["pid"]} '
                     f'({cfg["mode"]}, {"/".join(cfg["obj_metrics"])}, '
                     f'{cfg["constr_metric"]} <= {job["threshold"]:.4f}){kinds_note}'
                     + ('' if job['enumerable'] else '\n[cloud sampled (10k) -- approximation, not the true region]'))
            plot_attainment(job['cloud_F'], job['cloud_label'], job['obj_indices'], job['constr_index'],
                            job['threshold'], job['true_front'], job['fronts_by_method'],
                            job['obj_names'], title, out_png, plt)
            print(f'[analyse] wrote {out_png}')
    except ImportError:
        print('[analyse] matplotlib not available; skipping plots.')

    # ── stdout summary (mean over seeds) ────────────────────────────────────
    # One block per (scenario, instance, config) actually present -- config
    # is 'r3' (current), 'legacy' (pre-instance-axis), or 'view:s2' (s4
    # reading s2's physical runs); see module docstring.
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
                cfg = _cfg_for_kind(scenario, config_kind)
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
                handlers_present = sorted({r['handler'] for r in rows_cfg})
                if len(handlers_present) > 1:
                    print(f'  (default handler: {default_handler}; random runs only its '
                          f'default and h1-rejection -- selection-free otherwise)')
                print(f'  {"method:handler":>28s}  {"n":>2s}  {"M2(evals->feas)":>16s}  {"M3-final(waste)":>16s}  '
                      f'{"M4-final(front)":>16s}  {"M6 min/med slack":>18s}  {"M7 IGD+":>12s}  {"M7 HV-ratio":>12s}  '
                      f'{"soft-HV":>10s}')
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
                    print(f'  {name:>28s}  {n:>2d}  {m2_str:>16s}  {m3_mean:>16.4f}  '
                          f'{m4_mean:>16.1f}  {m6_min_mean:>8.4f}/{m6_med_mean:<8.4f}  '
                          f'{m7_igd_str}  {m7_hvr_str}  {soft_hv_mean:>10.4f}')
    print('=' * 118)

    print_winloss_matrices(winloss_matrices, instances, args.scenarios)

    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_root', default=os.path.join('results', 'constraint'),
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
    args = p.parse_args()
    if args.self_test:
        _self_test_attainment()
        sys.exit(0)
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_root, 'analysis')
    sys.exit(main(args))
