"""experiments2/scenario_run/analyse_scenarios.py -- post-hoc analysis for the
scenario_run S1-S6 campaign (scenarios.py). Consumes the per-seed pkls written
by run_scenario.py:

  {results_root}/{sid}/{mode}/{method}/{handler}/seed_{N}.pkl

Every pkl is self-describing via its ``meta`` dict (see run_scenario.run_single).

Grid, vs. the constraint2 campaign
-----------------------------------
constraint2's comparison axis was HANDLER (methods samos/samos-cheap, one
instance per scenario). Here the grid is scenario x mode x method x handler:
  scenarios : S1..S6 (scenarios.SCENARIOS), one fixed (space, suite, pid,
              objectives, constraint) per sid -- no per-instance discovery
              needed, unlike constraint2's multi-instance-per-scenario tree.
  modes     : hard / soft, SAME tau -- they differ only in TREATMENT (gate
              vs. archive-and-penalise), not in what counts as feasible.
  methods   : nsga2, samos are the full-row methods (GRID_METHODS) and carry
              the handler axis -- theirs is the comparison the user cares
              about. random, ctaea and ssansga2 (REFERENCE_METHODS) exist ONLY
              for their scenarios.fixed_handlers slots -- the mode default for
              random, h4-cdp for ctaea (whose constraint handling is
              intrinsic), and h4-cdp + b1-unconstrained for ssansga2 (whose
              only seam is whether its surrogate models the constraint) -- so
              the grid is RAGGED and handled explicitly everywhere below
              (never assumed present for every handler).
  handlers  : the 7 in scenarios.HANDLERS, run under BOTH modes (mode does
              not gate which handlers are legal, only the default), plus
              scenarios.EXTRA_HANDLER_SLOTS ('b1-unconstrained'), which is
              deliberately NOT part of that row but IS run by all three of
              nsga2, samos and ssansga2 -- it is the unconstrained-search
              floor of the grid, so it appears on the handler axis of every
              per-handler figure and statistic alongside the seven.

_metrics.py (local module)
---------------------------
true_feasible_front, sample_cloud, enumerate_full_space, space_size,
attainment_surfaces, _hv, _paired_wilcoxon, _holm, _pad_mean, _full_reeval
are self-contained ports of the single-constraint subset of the constraint2
campaign's feasibility-region metrics -- no coupling to that campaign's own
scenario/threshold config.

analyse_run_scenario/_build_ref below are NOT ports of that campaign's
analyse_run/_build_ref, for two reasons:
  - _build_ref there cross-checks that campaign's own DESIGN_FEASIBLE_FRACTION
    dicts; here scenarios.py's own ``feasible_fraction`` is the right
    reference.
  - that campaign's analyse_run hardcodes REF_POINT = np.ones(2) * 1.05 as a
    MODULE GLOBAL for both its per-generation hv_traj and its soft_hv,
    but the ref point belongs to the SCENARIO (derive_tau.py computes it as
    max(1.05, p95) per objective), not to the module: all six scenarios
    happen to land on (1.05, 1.05) today, but a space whose normalized
    objectives run well past 1.05 would silently score zero HV over most of
    its archive. analyse_run_scenario below is a local, single-constraint,
    ref_point-parameterised port of its M1-M8/soft-HV logic instead,
    reusing _full_reeval/_hv/NonDominatedSorting for the actual numeric work.
  - true_feasible_front's violation formula is ceiling-only. Used as-is for
    S1-S5; for S6 (a floor: feasible <=> metric >= tau)
    _true_feasible_front_signed below reflects the constrained column around
    tau (metric' = 2*tau - metric) before calling it and reflects the
    returned column back afterwards -- an algebraic involution (feasible <=>
    metric' <= tau <=> metric >= tau), so the ceiling-only helper is reused
    exactly, not forked.
  - analyse_run_scenario takes an explicit ``sense`` instead: the local
    equivalent has no such coupling to fix, so the sign is threaded straight
    into its own violation formula (sense * (metric - tau) / tau).

Output
------
  {output_dir}/scenario_metrics.csv     one row per (sid, mode, method,
                                        handler, seed); includes scenarios.csv's
                                        hard@10%/soft@10%/rho joined in.
  {output_dir}/scenario_trajectories.json   per-generation hv_traj / n_total
                                        (the evaluation axis) / m1_ratio per run.
  {output_dir}/handler_winloss.csv      handler-pair Wilcoxon (Holm-corrected,
                                        paired by seed), grouped within
                                        (sid, mode, method) -- "which handler
                                        wins" answered separately per method.
  {output_dir}/method_effect.csv        nsga2 vs samos, same handler, grouped
                                        within (sid, mode).
  {output_dir}/baseline_effect.csv      random / ctaea vs nsga2 / samos at the
                                        baseline's own handler slot, grouped
                                        within (sid, mode).
  {output_dir}/hard_vs_soft.csv         hard vs soft, same (method, handler),
                                        grouped within sid.
  {output_dir}/plot_data.pkl            everything the plot/table layer needs
                                        (--plots_only reruns rendering only).
  {output_dir}/eval_cache/{sid}.pkl     every genotype ever true-evaled for
                                        this scenario (cloud + per-run
                                        re-evals), reused by later analyses.
  {output_dir}/plots/traj_{mode}_{method}.png    feasible HV vs cumulative
                                        high-fidelity evaluations, scenario
                                        subplots, colour=handler, with the
                                        random reference and (enumerable
                                        scenarios) the exact HV ceiling.
  {output_dir}/plots/{sid}_{mode}_attainment.png   subplot per handler: cloud,
                                        true/sampled feasible front, per-method
                                        attainment surfaces.
  {output_dir}/plots/{sid}_{hv,besterr}_bars.png   hard/soft panels on a shared
                                        y-axis, x=handler, mean +- std per
                                        method, final feasible HV / best
                                        feasible error.
  {output_dir}/tables/handler_winloss_{method}.tex   W/L/T per handler, per
                                        (scenario, mode) row.
  {output_dir}/tables/method_effect.tex   nsga2 vs samos outcome per handler,
                                        per (scenario, mode) row.
  {output_dir}/critical_difference.csv  Friedman mean ranks over the same six
                                        methods, blocks = (scenario, mode),
                                        one row per (scope, metric, method).
  {output_dir}/plots/cd_{both,hard,soft}.png   Demsar critical-difference
                                        diagrams, one panel per metric.
  {output_dir}/tables/critical_difference.tex  mean-rank table with CD and the
                                        Friedman p per scope.
  {output_dir}/tables/method_comparison.tex   headline ten-row slice
                                        (random, nsga2/samos under h4-cdp,
                                        b0-as-obj and b1-unconstrained, ctaea,
                                        ssansga2 under h4-cdp and
                                        b1-unconstrained): final and
                                        anytime feasible HV, best feasible
                                        accuracy, waste.
  stdout summary table + win/tie/loss blocks.

Not re-doing work
-----------------
Nearly all the cost is true-evaluating genotypes through evoxbench's surrogate
(the sampled/enumerated cloud, plus every var_archive point re-evaled per
generation) -- for MoSegNAS ~0.2 s per unseen genotype. Two caches avoid it:
  - {output_dir}/eval_cache/{sid}.pkl : genotype -> objectives, keyed by the
    scenario definition + cloud settings. Evaluation is deterministic, so a
    key match makes every cached point valid; a rerun re-evaluates only
    genotypes it has never seen (i.e. only newly added seeds).
  - plot_data.pkl : when it is newer than every seed_*.pkl there is nothing
    new to analyse, so the run falls through to --plots_only. --force
    overrides; --no_cache disables the eval cache.

Examples
--------
  python experiments2/scenario_run/analyse_scenarios.py
  python experiments2/scenario_run/analyse_scenarios.py --scenarios S1 S6 --seeds 0 1 2
  python experiments2/scenario_run/analyse_scenarios.py --results_root smoke_tests/scenario_run \\
      --output_dir smoke_tests/scenario_run/analysis
"""

import argparse
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import pickle
import sys

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import spearmanr

from _metrics import (true_feasible_front, sample_cloud, enumerate_full_space,
                      space_size, attainment_surfaces, _hv, _paired_wilcoxon,
                      _holm, _pad_mean, _full_reeval)
from problem.evoxbench.utils import get_benchmark, bounds_with_override
import scenarios as SC

_DEFAULT_RESULTS_ROOT = os.path.join('results2', 'scenario_run')
_SCENARIOS_CSV = os.path.join('results2', 'constraint_analysis', 'paper', 'scenarios.csv')

STATS_ALPHA     = 0.05
STATS_MIN_SEEDS = 3    # scenario_run campaigns are smaller-N than constraint2's
STATS_METRICS   = ('hv_run', 'best_err')
STATS_METRIC_DIRECTION = {'hv_run': 1, 'best_err': -1}   # +1: higher better; -1: lower (error) better
STATS_METRIC_LABEL = {'hv_run': 'feasible HV', 'best_err': 'best feasible Err'}

CLOUD_SAMPLE_N    = 10_000
CLOUD_SAMPLE_SEED = 0
_CLOUD_PLOT_MAX   = 6000

# Bump when the genotype->objective mapping changes (e.g. the evoxbench decode
# patch in problem/evoxbench/utils.py), which invalidates every cached point.
_EVAL_CACHE_VERSION = 1

HANDLER_COLOURS = {
    'h1-rejection':      '#1f77b4',
    'h2-static_penalty': '#ff7f0e',
    'h3-adaptive':       '#2ca02c',
    'h4-cdp':            '#d62728',
    'h5-epsilon':        '#9467bd',
    'h6-DSR':            '#8c564b',
    'b0-as-obj':         '#7f7f7f',
    'b1-unconstrained':  '#bcbd22',
}
METHOD_LINESTYLE = {'samos': '-', 'nsga2': '--', 'ctaea': '-.', 'random': ':',
                    'ssansga2': '-.'}
METHOD_COLOURS   = {'nsga2': '#0072B2', 'samos': '#D55E00', 'ctaea': '#CC79A7',
                    'random': '#009E73', 'ssansga2': '#8c6bb1'}
# Per-SLOT style for reference methods holding MORE than one fixed slot: with
# one style per method, ssansga2's constrained and unconstrained curves would
# be drawn identically in every reference legend and the pair -- the reason the
# method is in the campaign at all -- would be unreadable.
REF_SLOT_STYLE = {
    ('ssansga2', 'h4-cdp'):           ('#8c6bb1', (0, (4, 1, 1, 1))),
    ('ssansga2', 'b1-unconstrained'): ('#d4a6c8', (0, (1, 1))),
}
# Full-row methods: the handler axis is theirs, so they carry the handler
# colouring and the handler-pair / method-effect statistics. REFERENCE_METHODS
# (random, ctaea, ssansga2) hold their scenarios.fixed_handlers slots instead
# and are drawn as reference curves/lines in every per-handler figure.
GRID_METHODS      = [m for m in SC.METHODS if m not in SC.FIXED_HANDLER_METHODS]
REFERENCE_METHODS = [m for m in SC.METHODS if m in SC.FIXED_HANDLER_METHODS]
CLOUD_FEASIBLE_COLOUR   = '#b8d4ea'
CLOUD_INFEASIBLE_COLOUR = '#f4c7b8'


def _ref_style(method, handler):
    """(colour, linestyle) for one reference-method curve at one slot."""
    return REF_SLOT_STYLE.get((method, handler),
                              (METHOD_COLOURS[method], METHOD_LINESTYLE[method]))


def _ref_sort_key(method_handler):
    method, handler = method_handler
    return (REFERENCE_METHODS.index(method), _handler_sort_key(handler))


def _handler_sort_key(handler):
    try:
        return (SC.ALL_HANDLERS.index(handler), handler)
    except ValueError:
        return (len(SC.ALL_HANDLERS), handler)


# ─── scenarios.csv join (optional, per the analysis brief) ─────────────────

def _load_scenario_suite_table(path):
    """{sid: {'hard_at_10', 'soft_at_10', 'rho'}} from the published scenario
    suite CSV, or {} if the file is absent (join is optional)."""
    if not os.path.exists(path):
        print(f'[analyse] {path} not found; scenario-suite columns will be blank.')
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    out = {}
    for row in df.itertuples():
        out[row.sid] = dict(hard_at_10=float(getattr(row, '_9')),   # 'hard@10%'
                             soft_at_10=float(getattr(row, '_10')),  # 'soft@10%'
                             rho=float(row.rho))
    return out


# ─── bounds-override wrapper (MoSegNAS x0>=1; see scenarios.SEARCH_SPACE_LB_OVERRIDE) ──

class _BoundsOverrideBenchmark:
    """Thin proxy over a benchmark that reports overridden search-space
    bounds and delegates everything else -- used only when building the
    attainable cloud (sample_cloud / enumerate_full_space / space_size) for
    scenarios whose space needs bounds_with_override (MoSegNAS: x0 == 0 is an
    invalid genotype). Per-run re-evaluation never needs this: the run's own
    problem already enforced the override while searching, so var_archive
    points already respect it."""

    def __init__(self, benchmark, lb_override):
        self._bm = benchmark
        xl, xu = bounds_with_override(benchmark, lb_override)
        self.search_space = type('_SS', (), {'lb': xl, 'ub': xu})()

    def __getattr__(self, name):
        return getattr(self._bm, name)


# ─── local, sense-aware true-feasible-front (reuses true_feasible_front) ───

def _true_feasible_front_signed(F_all, obj_indices, constr_idx, tau, sense):
    """true_feasible_front's violation formula is ceiling-only ((metric-T)/T
    <= 0). For a floor constraint (sense=-1, S6: feasible <=> metric >= tau)
    reflect the constrained column around tau -- metric' = 2*tau - metric --
    before calling it (feasible <=> metric' <= tau <=> metric >= tau, an
    algebraic involution), then reflect the returned constraint column back
    to the real metric so callers never see the reflected value."""
    if sense == -1:
        F_signed = np.array(F_all, dtype=float, copy=True)
        F_signed[:, constr_idx] = 2.0 * tau - F_signed[:, constr_idx]
    else:
        F_signed = F_all
    frac, front, front_c = true_feasible_front(F_signed, obj_indices, constr_idx, tau)
    if sense == -1 and len(front_c) > 0:
        front_c = front_c.copy()
        front_c[:, -1] = 2.0 * tau - front_c[:, -1]
    return frac, front, front_c


# ─── persistent genotype->objective cache ─────────────────────────────────

def _eval_cache_key(sid, enum_limit):
    """Identity of a scenario's evaluated point set: the scenario definition
    plus the cloud settings. Evaluation is deterministic, so a matching key
    means every cached (genotype -> F) pair is still valid."""
    payload = repr((_EVAL_CACHE_VERSION, sid, sorted(SC.SCENARIOS[sid].items(), key=str),
                    enum_limit, CLOUD_SAMPLE_N, CLOUD_SAMPLE_SEED))
    return hashlib.sha1(payload.encode()).hexdigest()


def _load_eval_cache(cache_dir, sid, key):
    """(X_cloud, F_cloud, cache) from a previous analysis, or None when the
    file is absent or its key no longer matches."""
    path = os.path.join(cache_dir, f'{sid}.pkl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as fh:
            blob = pickle.load(fh)
    except Exception as exc:
        print(f'[analyse] {sid}: unreadable eval cache ({exc}); rebuilding.')
        return None
    if blob.get('key') != key:
        print(f'[analyse] {sid}: eval cache stale (scenario or cloud settings changed); rebuilding.')
        return None
    cache = {tuple(int(v) for v in x): f for x, f in zip(blob['cache_X'], blob['cache_F'])}
    return blob['X_cloud'], blob['F_cloud'], cache


def _save_eval_cache(cache_dir, sid, key, ref):
    """Persist the cloud and every point evaluated while analysing this
    scenario, so the next analysis re-evaluates nothing."""
    keys = list(ref['cache'].keys())
    if not keys:
        return
    os.makedirs(cache_dir, exist_ok=True)
    blob = dict(key=key, X_cloud=ref['X_cloud'], F_cloud=ref['F_cloud'],
                cache_X=np.asarray(keys, dtype=np.int32),
                cache_F=np.asarray([ref['cache'][k] for k in keys], dtype=float))
    path = os.path.join(cache_dir, f'{sid}.pkl')
    with open(path, 'wb') as fh:
        pickle.dump(blob, fh, protocol=4)
    print(f'[analyse] {sid}: cached {len(keys)} evaluated points -> {path}')


# ─── per-scenario reference (local _build_ref; see module docstring) ───────

def _build_ref(sid, enum_limit, cache_dir=None):
    """True/sampled feasible front, HV reference and re-eval cache for one
    scenario -- shared across BOTH modes and every method/handler (tau,
    ref_point and objectives are the same for hard and soft; only the
    runtime TREATMENT of infeasibility differs, not what analysis needs)."""
    scenario  = SC.SCENARIOS[sid]
    benchmark = get_benchmark(scenario['suite'], scenario['pid'])
    obj_idx    = list(SC.obj_indices(sid))
    constr_idx = SC.constr_index(sid)
    tau        = scenario['tau']
    sense      = SC.constr_sense(sid)
    ref_point  = np.asarray(scenario['ref_point'], dtype=float)

    lb_override = SC.SEARCH_SPACE_LB_OVERRIDE.get(scenario['space'])
    cloud_bm = _BoundsOverrideBenchmark(benchmark, lb_override) if lb_override else benchmark

    n_space = space_size(cloud_bm)
    enumerable = n_space <= enum_limit
    hit = _load_eval_cache(cache_dir, sid, _eval_cache_key(sid, enum_limit)) if cache_dir else None
    if hit is not None:
        X_cloud, F_cloud, cache = hit
        cloud_label = f'{"enumerated" if enumerable else "sampled"} ({len(X_cloud)}) [cached]'
    elif enumerable:
        X_cloud, F_cloud = enumerate_full_space(cloud_bm)
        cloud_label = f'enumerated ({len(X_cloud)})'
    else:
        X_cloud, F_cloud = sample_cloud(cloud_bm, n=CLOUD_SAMPLE_N, seed=CLOUD_SAMPLE_SEED)
        cloud_label = f'sampled ({len(X_cloud)})'
    print(f'[analyse] {sid} ({scenario["space"]} {scenario["suite"]}/pid{scenario["pid"]}): '
          f'|space|={float(n_space):.3e} -- {"ENUMERABLE" if enumerable else "non-enumerable"}, '
          f'cloud={cloud_label}')

    frac, true_front, true_front_c = _true_feasible_front_signed(
        F_cloud, obj_idx, constr_idx, tau, sense)
    design_frac = scenario['feasible_fraction']
    if enumerable:
        assert len(true_front) > 0, f'{sid}: true feasible front is empty -- tau/sense wiring is broken.'
        assert abs(frac - design_frac) < 0.05, (
            f'{sid}: enumerated feasible fraction {frac:.4f} far from the design fraction '
            f'{design_frac:.4f} (tau={tau}, constr={scenario["constr_metric"]}) -- check TAU.md.')
        hv_true = _hv(true_front, ref_point)
        best_err_true = float(np.min(true_front[:, 0]))
        print(f'[analyse] {sid}: true feasible fraction={frac:.4f}, front size={len(true_front)}, '
              f'HV(true front)={hv_true:.4f}')
    else:
        hv_true, best_err_true = float('nan'), float('nan')
        print(f'[analyse] {sid}: sampled feasible fraction={frac:.4f} (info only); '
              f'true front n/a (non-enumerable).')

    if hit is None:
        cache = {tuple(int(v) for v in x): np.asarray(f, dtype=float) for x, f in zip(X_cloud, F_cloud)}

    # Scalars, not the benchmark handle: refs is stripped of `benchmark` before
    # being pickled into plot_data.pkl, and these have to survive --plots_only.
    err_utopian = float(benchmark.utopian_point[obj_idx[0]])
    err_nadir   = float(benchmark.nadir_point[obj_idx[0]])
    # NaN for non-enumerable scenarios (best_err_true is), which is what keeps
    # the comparison table's dagger off a sampled, non-exact optimum.
    best_acc_true = _accuracy_pct(best_err_true, err_utopian, err_nadir)

    return dict(sid=sid, scenario=scenario, benchmark=benchmark, obj_idx=obj_idx,
                constr_idx=constr_idx, tau=tau, sense=sense, ref_point=ref_point,
                eval_kind=SC.eval_kind(sid),
                enumerable=enumerable, n_space=n_space, F_cloud=F_cloud, X_cloud=X_cloud,
                cloud_label=cloud_label, true_front=true_front, true_front_c=true_front_c,
                hv_true=hv_true, best_err_true=best_err_true,
                err_utopian=err_utopian, err_nadir=err_nadir, best_acc_true=best_acc_true,
                frac=frac, cache=cache)


# numpy renamed trapz -> trapezoid in 2.0 and dropped the old spelling; this
# env is on 1.26, which has only the old one.
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def _hv_auc(hv_traj, n_eval_traj):
    """Anytime quality: mean feasible HV over the WHOLE evaluation budget --
    the area under the HV-vs-evaluations curve divided by the budget spent.

    Anchored at (0 evals, 0 HV) rather than integrated between a run's own
    first and last recorded points, for two reasons: the HV really is zero
    before anything has been evaluated, and every run then shares the same
    [0, budget] window. Integrating over each run's own window would hand a
    silent advantage to whichever method starts recording latest (a large DOE,
    or h1-rejection burning evaluations on infeasible probes), because it
    would simply skip the early low-HV stretch instead of being charged for
    it."""
    n = min(len(hv_traj or []), len(n_eval_traj or []))
    if n < 1:
        return float('nan')
    x = np.concatenate([[0.0], np.asarray(n_eval_traj[:n], dtype=float)])
    y = np.concatenate([[0.0], np.asarray(hv_traj[:n], dtype=float)])
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or x[ok][-1] <= 0:
        return float('nan')
    return float(_trapz(y[ok], x[ok]) / x[ok][-1])


def _accuracy_pct(err_norm, utopian, nadir):
    """Benchmark-normalized first-objective (error) value -> test accuracy in
    percent. Inverts Benchmark.normalize -- an affine map onto the
    utopian/nadir box, so raw = utopian + err_norm * (nadir - utopian) -- and
    complements the result.

    The raw unit is NOT uniform across the suite: c10mop reports Err. as a
    percentage (S1's nadir is 20.21, S4-S6's is 90.0) while in1kmop reports a
    fraction (S2's nadir is 0.31, S3's is 0.30). A nadir above 1.0 cannot be a
    fraction and one below 1.0 cannot be a percentage, so the nadir picks the
    unit unambiguously for every scenario in this suite. NaN in (a
    non-enumerable scenario's absent true optimum) gives NaN out."""
    raw = utopian + float(err_norm) * (nadir - utopian)
    return 100.0 - (raw if nadir > 1.0 else 100.0 * raw)


# ─── per-run metrics (local analyse_run; see module docstring) ─────────────

def analyse_run_scenario(data, obj_idx, constr_idx, tau, sense, ref_point, benchmark, cache):
    """Per-run metrics: feasibility trajectories, feasible-HV trajectory
    (ref_point-aware), M2/M6/M8-style boundary metrics and soft-HV, single
    constraint, sense-aware (re-eval every generation's var_archive via
    _full_reeval, distinguish feasible/infeasible via the signed violation,
    feasible HV against the scenario's own ref_point) -- see module
    docstring for why this is a local, ref_point-parameterised port rather
    than a shared implementation."""
    var_archive      = data.get('var_archive', [])
    test_obj_archive = data.get('test_obj_archive', [])
    n_feasible = np.asarray(data.get('n_feasible', []), dtype=float)
    n_total    = np.asarray(data.get('n_total', []), dtype=float)
    n_gen = len(var_archive)

    m1_ratio = np.where(n_total > 0, n_feasible / np.where(n_total > 0, n_total, 1), np.nan)
    m3_cumulative = np.where(n_total > 0, (n_total - n_feasible) / np.where(n_total > 0, n_total, 1), np.nan)
    delta_total    = np.diff(n_total, prepend=0.0)
    delta_feasible = np.diff(n_feasible, prepend=0.0)
    m3_per_gen = np.where(delta_total > 0,
                         (delta_total - delta_feasible) / np.where(delta_total > 0, delta_total, 1),
                         np.nan)
    m4_front_size = [len(g) for g in test_obj_archive]

    def viol(col):
        return sense * (col - tau) / tau

    m5_mean_violation, full_evals_per_gen = [], []
    for g in range(n_gen):
        F_full, finite_mask = _full_reeval(benchmark, var_archive[g], cache)
        full_evals_per_gen.append((F_full, finite_mask))
        F_fin = F_full[finite_mask]
        if len(F_fin) == 0:
            m5_mean_violation.append(float('nan'))
            continue
        v = viol(F_fin[:, constr_idx])
        infeasible = v > 0
        m5_mean_violation.append(float(np.mean(v[infeasible])) if infeasible.any() else float('nan'))

    if n_gen > 0:
        F_last, finite_last = full_evals_per_gen[-1]
        F_last_fin = F_last[finite_last]
        v_last = viol(F_last_fin[:, constr_idx]) if len(F_last_fin) else np.zeros(0)
        feasible_seq = v_last <= 0
        if feasible_seq.any():
            m2_evals_to_first_feasible, m2_censored = int(np.argmax(feasible_seq)) + 1, False
        else:
            m2_evals_to_first_feasible, m2_censored = float('inf'), True
        m2_n_evaluated = int(len(F_last_fin))
    else:
        F_last_fin = np.empty((0, benchmark.evaluator.n_objs))
        v_last = np.zeros(0)
        m2_evals_to_first_feasible, m2_censored, m2_n_evaluated = float('inf'), True, 0

    final_front = (np.asarray(test_obj_archive[-1]) if n_gen > 0 and len(test_obj_archive[-1]) > 0
                   else np.empty((0, len(obj_idx))))
    feasible_mask_last = v_last <= 0 if len(F_last_fin) else np.zeros(0, dtype=bool)

    if len(F_last_fin) and (~feasible_mask_last).any():
        F_inf = F_last_fin[~feasible_mask_last]
        final_infeas_c = np.column_stack([F_inf[:, obj_idx], F_inf[:, constr_idx]])
    else:
        final_infeas_c = np.empty((0, len(obj_idx) + 1))

    if len(final_front) > 0 and feasible_mask_last.any():
        F_feas_last = F_last_fin[feasible_mask_last]
        v_feas_last = v_last[feasible_mask_last]
        obj_feas_last = F_feas_last[:, obj_idx]
        nd_idx = NonDominatedSorting().do(obj_feas_last, only_non_dominated_front=True)
        slack = -v_feas_last[nd_idx]
        m6_min_slack, m6_median_slack = float(np.min(slack)), float(np.median(slack))
        final_front_c = np.column_stack([obj_feas_last[nd_idx], F_feas_last[nd_idx, constr_idx]])
        best_i = int(np.argmin(F_feas_last[:, obj_idx[0]]))
        m8_best_err = float(F_feas_last[best_i, obj_idx[0]])
        m8_best_err_slack = float(-v_feas_last[best_i])
    else:
        m6_min_slack = m6_median_slack = float('nan')
        final_front_c = np.empty((0, len(obj_idx) + 1))
        m8_best_err = m8_best_err_slack = float('nan')

    if len(F_last_fin) > 0:
        # hv_all: the SAME final archive as soft_hv, but unshifted and
        # unfiltered -- feasibility ignored entirely, so it measures raw
        # multi-objective search quality and hv_all - hv_run is what
        # satisfying the constraint costs. Under the hard gate the two nearly
        # coincide by construction: infeasible F is inf there, so those
        # members are dominated out of the callback's archive before this ever
        # sees them. That is the premise of hard mode, not a defect of the
        # metric -- the column is informative for soft and b0 rows.
        obj_all = F_last_fin[:, obj_idx]
        nd_idx_all = NonDominatedSorting().do(obj_all, only_non_dominated_front=True)
        hv_all = _hv(obj_all[nd_idx_all], ref_point)
        shifted = F_last_fin[:, obj_idx] + np.maximum(0.0, v_last)[:, None]
        nd_idx_soft = NonDominatedSorting().do(shifted, only_non_dominated_front=True)
        soft_hv = _hv(shifted[nd_idx_soft], ref_point)
    else:
        hv_all = soft_hv = 0.0

    hv_traj = [_hv(g, ref_point) for g in test_obj_archive]

    n_evaluated_raw = data.get('n_evaluated')
    n_feasible_evaluated_raw = data.get('n_feasible_evaluated')
    if n_evaluated_raw and n_feasible_evaluated_raw and n_evaluated_raw[-1]:
        exact_waste = 1.0 - n_feasible_evaluated_raw[-1] / n_evaluated_raw[-1]
        ne = np.asarray(n_evaluated_raw, dtype=float)
        nf = np.asarray(n_feasible_evaluated_raw, dtype=float)
        exact_waste_traj = np.where(ne > 0, 1.0 - nf / np.where(ne > 0, ne, 1), np.nan).tolist()
    else:
        exact_waste, exact_waste_traj = float('nan'), []

    return dict(
        m1_ratio=m1_ratio.tolist(), m3_cumulative=m3_cumulative.tolist(), m3_per_gen=m3_per_gen.tolist(),
        m4_front_size=m4_front_size, m5_mean_violation=m5_mean_violation, hv_traj=hv_traj,
        n_total_traj=n_total.tolist(),
        # n_total is the ARCHIVE SIZE (non-monotone: members drop out when
        # dominated), never the budget spent. n_evaluated is the algorithm's
        # own cumulative high-fidelity evaluation count -- the axis this
        # evaluation-budgeted campaign is actually run against.
        n_eval_traj=[float(v) for v in (n_evaluated_raw or [])],
        m2_evals_to_first_feasible=m2_evals_to_first_feasible, m2_censored=m2_censored,
        m2_n_evaluated=m2_n_evaluated,
        m3_final_cumulative=float(m3_cumulative[-1]) if n_gen > 0 else float('nan'),
        m4_final=m4_front_size[-1] if m4_front_size else 0,
        m6_min_slack=m6_min_slack, m6_median_slack=m6_median_slack,
        m8_best_err=m8_best_err, m8_best_err_slack=m8_best_err_slack,
        hv_all=hv_all, soft_hv=soft_hv,
        hv_auc=_hv_auc(hv_traj, n_evaluated_raw),
        exact_waste=exact_waste, exact_waste_traj=exact_waste_traj,
        final_front=final_front, final_front_c=final_front_c, final_infeas_c=final_infeas_c,
        n_gen=n_gen,
    )


# ─── discovery ──────────────────────────────────────────────────────────────

def _discover_seeds(run_dir):
    if not os.path.isdir(run_dir):
        return []
    seeds = []
    for fn in os.listdir(run_dir):
        if fn.startswith('seed_') and fn.endswith('.pkl'):
            try:
                seeds.append(int(fn[len('seed_'):-len('.pkl')]))
            except ValueError:
                pass
    return sorted(seeds)


def _pairs_for_mode(mode, methods, handlers):
    """(method, handler) work list for one mode: random, ctaea and ssansga2
    exist only for their scenarios.fixed_handlers slots -- a ragged grid,
    mirroring run_scenario.main()'s own pairing rule exactly."""
    pairs = []
    for method in methods:
        fixed = SC.fixed_handlers(method, mode)
        if fixed is not None:
            pairs.extend((method, h) for h in fixed if h in handlers)
            continue
        for handler in handlers:
            pairs.append((method, handler))
    return pairs


# ─── significance testing (local; families differ from constraint2's GA-axis ones) ─

def compute_handler_winloss(stat_records, alpha=STATS_ALPHA, min_seeds=STATS_MIN_SEEDS):
    """Handler-pair Wilcoxon signed-rank, paired by seed, Holm-corrected
    within each (sid, mode, method) group -- "which handler wins" answered
    SEPARATELY per method (nsga2, samos), never mixing methods or modes into
    one correction unit."""
    groups = {}
    for r in stat_records:
        if r['method'] not in GRID_METHODS:
            continue                      # one handler slot each -- no pair to test
        key = (r['sid'], r['mode'], r['method'])
        groups.setdefault(key, {}).setdefault(r['handler'], []).append(r)

    rows = []
    for (sid, mode, method), by_handler in groups.items():
        handlers = sorted(by_handler, key=_handler_sort_key)
        if len(handlers) < 2:
            continue
        for metric in STATS_METRICS:
            values = {h: {rec['seed']: rec[metric] for rec in by_handler[h] if np.isfinite(rec[metric])}
                      for h in handlers}
            pair_results = []
            for a, b in itertools.combinations(handlers, 2):
                common = sorted(set(values[a]) & set(values[b]))
                if len(common) < min_seeds:
                    continue
                av = np.array([values[a][s] for s in common], dtype=float)
                bv = np.array([values[b][s] for s in common], dtype=float)
                stat, p = _paired_wilcoxon(av, bv)
                pair_results.append(dict(a=a, b=b, stat=stat, p_raw=p, n=len(common),
                                         med_a=float(np.median(av)), med_b=float(np.median(bv))))
            if not pair_results:
                continue
            p_holm = _holm([pr['p_raw'] for pr in pair_results])
            direction = STATS_METRIC_DIRECTION[metric]
            for pr, p_adj in zip(pair_results, p_holm):
                significant = bool(p_adj < alpha)
                better_a = (pr['med_a'] > pr['med_b']) if direction > 0 else (pr['med_a'] < pr['med_b'])
                outcome_ab = ('W' if better_a else 'L') if significant else 'T'
                rows.append(dict(sid=sid, mode=mode, method=method, metric=metric,
                                 handler_a=pr['a'], handler_b=pr['b'], n=pr['n'],
                                 median_a=pr['med_a'], median_b=pr['med_b'],
                                 statistic=pr['stat'], p_raw=pr['p_raw'], p_holm=float(p_adj),
                                 significant=significant, outcome_ab=outcome_ab))
    return rows


def compute_method_effect(stat_records, alpha=STATS_ALPHA, min_seeds=STATS_MIN_SEEDS):
    """nsga2 vs samos, SAME handler, seed-paired, Holm-corrected within each
    (sid, mode) group across every handler present under both methods."""
    groups = {}
    for r in stat_records:
        if r['method'] not in ('nsga2', 'samos'):
            continue
        key = (r['sid'], r['mode'])
        groups.setdefault(key, {}).setdefault(r['handler'], {}).setdefault(r['method'], []).append(r)

    rows = []
    for (sid, mode), by_handler in groups.items():
        handlers = sorted((h for h, by_m in by_handler.items() if 'nsga2' in by_m and 'samos' in by_m),
                          key=_handler_sort_key)
        if not handlers:
            continue
        for metric in STATS_METRICS:
            pair_results = []
            for h in handlers:
                va = {rec['seed']: rec[metric] for rec in by_handler[h]['nsga2'] if np.isfinite(rec[metric])}
                vb = {rec['seed']: rec[metric] for rec in by_handler[h]['samos'] if np.isfinite(rec[metric])}
                common = sorted(set(va) & set(vb))
                if len(common) < min_seeds:
                    continue
                av = np.array([va[s] for s in common], dtype=float)
                bv = np.array([vb[s] for s in common], dtype=float)
                stat, p = _paired_wilcoxon(av, bv)
                pair_results.append(dict(handler=h, stat=stat, p_raw=p, n=len(common),
                                         med_nsga2=float(np.median(av)), med_samos=float(np.median(bv))))
            if not pair_results:
                continue
            p_holm = _holm([pr['p_raw'] for pr in pair_results])
            direction = STATS_METRIC_DIRECTION[metric]
            for pr, p_adj in zip(pair_results, p_holm):
                significant = bool(p_adj < alpha)
                nsga2_better = ((pr['med_nsga2'] > pr['med_samos']) if direction > 0
                               else (pr['med_nsga2'] < pr['med_samos']))
                outcome = ('nsga2' if nsga2_better else 'samos') if significant else 'tie'
                rows.append(dict(sid=sid, mode=mode, metric=metric, handler=pr['handler'], n=pr['n'],
                                 median_nsga2=pr['med_nsga2'], median_samos=pr['med_samos'],
                                 statistic=pr['stat'], p_raw=pr['p_raw'], p_holm=float(p_adj),
                                 significant=significant, better=outcome))
    return rows


def compute_baseline_effect(stat_records, alpha=STATS_ALPHA, min_seeds=STATS_MIN_SEEDS):
    """Each REFERENCE_METHODS baseline (random, ctaea, ssansga2) vs every
    full-row method AT EACH OF THE BASELINE'S OWN HANDLER SLOTS, seed-paired,
    Holm-corrected within each (sid, mode) group.

    Slot-matched rather than baseline-vs-all-49-cells: ctaea's slot is h4-cdp
    because constraint domination is the mechanism it implements, so
    ctaea-vs-{nsga2, samos} at h4-cdp isolates the ALGORITHM (CA/DA archives +
    restricted mating, or a surrogate) with the handling mechanism held fixed.
    Testing it against all seven handlers instead would answer a different
    question with 7x the family size and correspondingly less power; the
    per-handler numbers are all in scenario_metrics.csv for anyone who wants
    them.

    ssansga2 contributes BOTH of its slots here on the same grounds: nsga2 and
    samos run b1-unconstrained too, so that slot is genuinely slot-matched --
    the test isolates the algorithm with the constraint hidden from all three,
    the unconstrained counterpart of the h4-cdp comparison directly above it."""
    by_cell = {}
    for r in stat_records:
        by_cell.setdefault((r['sid'], r['mode']), {}) \
               .setdefault((r['method'], r['handler']), {})[r['seed']] = r

    rows = []
    for (sid, mode), cells in by_cell.items():
        for metric in STATS_METRICS:
            pair_results = []
            for baseline in REFERENCE_METHODS:
                for slot in SC.fixed_handlers(baseline, mode):
                    base_cell = cells.get((baseline, slot))
                    if not base_cell:
                        continue
                    for method in GRID_METHODS:
                        other = cells.get((method, slot))
                        if not other:
                            continue
                        common = sorted(s for s in set(base_cell) & set(other)
                                        if np.isfinite(base_cell[s][metric])
                                        and np.isfinite(other[s][metric]))
                        if len(common) < min_seeds:
                            continue
                        av = np.array([base_cell[s][metric] for s in common], dtype=float)
                        bv = np.array([other[s][metric] for s in common], dtype=float)
                        stat, p = _paired_wilcoxon(av, bv)
                        pair_results.append(dict(baseline=baseline, method=method, handler=slot,
                                                 stat=stat, p_raw=p, n=len(common),
                                                 med_base=float(np.median(av)),
                                                 med_method=float(np.median(bv))))
            if not pair_results:
                continue
            p_holm = _holm([pr['p_raw'] for pr in pair_results])
            direction = STATS_METRIC_DIRECTION[metric]
            for pr, p_adj in zip(pair_results, p_holm):
                significant = bool(p_adj < alpha)
                base_better = ((pr['med_base'] > pr['med_method']) if direction > 0
                               else (pr['med_base'] < pr['med_method']))
                outcome = ((pr['baseline'] if base_better else pr['method'])
                           if significant else 'tie')
                rows.append(dict(sid=sid, mode=mode, metric=metric, baseline=pr['baseline'],
                                 method=pr['method'], handler=pr['handler'], n=pr['n'],
                                 median_baseline=pr['med_base'], median_method=pr['med_method'],
                                 statistic=pr['stat'], p_raw=pr['p_raw'], p_holm=float(p_adj),
                                 significant=significant, better=outcome))
    return rows


def compute_hard_vs_soft(stat_records, alpha=STATS_ALPHA, min_seeds=STATS_MIN_SEEDS):
    """Hard vs soft, SAME (method, handler), seed-paired, Holm-corrected
    within each sid across every (method, handler) present under both modes."""
    groups = {}
    for r in stat_records:
        key = r['sid']
        groups.setdefault(key, {}).setdefault((r['method'], r['handler']), {}) \
            .setdefault(r['mode'], []).append(r)

    rows = []
    for sid, by_mh in groups.items():
        mh_present = sorted((mh for mh, by_mode in by_mh.items() if 'hard' in by_mode and 'soft' in by_mode),
                            key=lambda mh: (mh[0], _handler_sort_key(mh[1])))
        if not mh_present:
            continue
        for metric in STATS_METRICS:
            pair_results = []
            for method, handler in mh_present:
                va = {rec['seed']: rec[metric] for rec in by_mh[(method, handler)]['hard']
                      if np.isfinite(rec[metric])}
                vb = {rec['seed']: rec[metric] for rec in by_mh[(method, handler)]['soft']
                      if np.isfinite(rec[metric])}
                common = sorted(set(va) & set(vb))
                if len(common) < min_seeds:
                    continue
                av = np.array([va[s] for s in common], dtype=float)
                bv = np.array([vb[s] for s in common], dtype=float)
                stat, p = _paired_wilcoxon(av, bv)
                pair_results.append(dict(method=method, handler=handler, stat=stat, p_raw=p, n=len(common),
                                         med_hard=float(np.median(av)), med_soft=float(np.median(bv))))
            if not pair_results:
                continue
            p_holm = _holm([pr['p_raw'] for pr in pair_results])
            direction = STATS_METRIC_DIRECTION[metric]
            for pr, p_adj in zip(pair_results, p_holm):
                significant = bool(p_adj < alpha)
                hard_better = (pr['med_hard'] > pr['med_soft']) if direction > 0 else (pr['med_hard'] < pr['med_soft'])
                outcome = ('hard' if hard_better else 'soft') if significant else 'tie'
                rows.append(dict(sid=sid, method=pr['method'], handler=pr['handler'], metric=metric,
                                 n=pr['n'], median_hard=pr['med_hard'], median_soft=pr['med_soft'],
                                 statistic=pr['stat'], p_raw=pr['p_raw'], p_holm=float(p_adj),
                                 significant=significant, better=outcome))
    return rows


def print_winloss_matrices(winloss_rows):
    """Compact per-(sid, mode, method) handler win/tie/loss grid on stdout."""
    print('\n' + '=' * 110)
    print(f'Handler pairwise significance (Wilcoxon signed-rank, paired by seed, Holm-corrected, alpha={STATS_ALPHA})')
    print("W = row handler beats column; L = loses; T = tie / not significant")
    print('=' * 110)
    groups = {}
    for r in winloss_rows:
        groups.setdefault((r['sid'], r['mode'], r['method'], r['metric']), []).append(r)
    if not groups:
        print('\n(no handler pair had enough seeds on both sides; nothing to report)')
        print('=' * 110)
        return
    for (sid, mode, method, metric), rows in sorted(groups.items()):
        handlers = sorted({h for r in rows for h in (r['handler_a'], r['handler_b'])}, key=_handler_sort_key)
        cell = {}
        for r in rows:
            cell[(r['handler_a'], r['handler_b'])] = r['outcome_ab']
            cell[(r['handler_b'], r['handler_a'])] = {'W': 'L', 'L': 'W', 'T': 'T'}[r['outcome_ab']]
        col_w = max(16, max(len(h) for h in handlers) + 2)
        print(f'\n{sid} / {mode} / {method}  metric={STATS_METRIC_LABEL[metric]}')
        print('  ' + ' ' * col_w + ''.join(f'{h:>{col_w}s}' for h in handlers))
        for a in handlers:
            row = ''.join(f'{"-":>{col_w}s}' if a == b else f'{cell.get((a, b), "."):>{col_w}s}'
                         for b in handlers)
            print(f'  {a:>{col_w}s}{row}')
    print('=' * 110)


def print_method_effect_summary(method_effect_rows, metric='hv_run'):
    """nsga2-vs-samos outcome per (sid, mode, handler) on stdout."""
    rows = [r for r in method_effect_rows if r['metric'] == metric]
    print('\n' + '=' * 100)
    print(f'Method effect (nsga2 vs samos, same handler, {STATS_METRIC_LABEL[metric]})')
    print('=' * 100)
    if not rows:
        print('(no handler had enough common seeds under both methods; nothing to report)')
        print('=' * 100)
        return
    for r in sorted(rows, key=lambda r: (int(r['sid'][1:]), r['mode'], _handler_sort_key(r['handler']))):
        print(f"  {r['sid']}/{r['mode']:<4s} {r['handler']:<19s} n={r['n']:<2d} "
             f"nsga2={r['median_nsga2']:.4f}  samos={r['median_samos']:.4f}  "
             f"p_holm={r['p_holm']:.4f}  winner={r['better']}")
    print('=' * 100)


def print_baseline_effect_summary(baseline_rows, metric='hv_run'):
    """Reference-method (random, ctaea) vs full-row method at the baseline's
    own handler slot, on stdout."""
    rows = [r for r in baseline_rows if r['metric'] == metric]
    print('\n' + '=' * 100)
    print(f'Baseline effect (random / ctaea vs nsga2 / samos at the baseline handler slot, '
          f'{STATS_METRIC_LABEL[metric]})')
    print('=' * 100)
    if not rows:
        print('(no baseline cell had enough common seeds against a full-row method; nothing to report)')
        print('=' * 100)
        return
    for r in sorted(rows, key=lambda r: (int(r['sid'][1:]), r['mode'], r['baseline'], r['method'])):
        print(f"  {r['sid']}/{r['mode']:<4s} {r['baseline']:>6s} vs {r['method']:<6s} "
             f"@{r['handler']:<19s} n={r['n']:<2d} "
             f"{r['median_baseline']:.4f} / {r['median_method']:.4f}  "
             f"p_holm={r['p_holm']:.4f}  winner={r['better']}")
    print('=' * 100)


def print_hard_vs_soft_summary(hard_soft_rows, metric='hv_run'):
    """hard-vs-soft outcome per (sid, method, handler) on stdout."""
    rows = [r for r in hard_soft_rows if r['metric'] == metric]
    print('\n' + '=' * 100)
    print(f'Hard vs soft (same method/handler, {STATS_METRIC_LABEL[metric]})')
    print('=' * 100)
    if not rows:
        print('(no method/handler had enough common seeds under both modes; nothing to report)')
        print('=' * 100)
        return
    for r in sorted(rows, key=lambda r: (int(r['sid'][1:]), r['method'], _handler_sort_key(r['handler']))):
        print(f"  {r['sid']}/{r['method']:<6s} {r['handler']:<19s} n={r['n']:<2d} "
             f"hard={r['median_hard']:.4f}  soft={r['median_soft']:.4f}  "
             f"p_holm={r['p_holm']:.4f}  winner={r['better']}")
    print('=' * 100)


# ─── plotting ───────────────────────────────────────────────────────────────

TRAJ_YLIM_FROM = 0.25   # ignore the first 25% of the budget when setting ylim


def _traj_mean_curve(seeds_dict):
    """(x, y) mean-over-seeds trajectory for one (method, handler) cell, or
    None when the cell has no usable series."""
    x_series = [v['n_eval_traj'] for v in seeds_dict.values() if v.get('n_eval_traj')]
    y_series = [v['hv_traj'] for v in seeds_dict.values() if v.get('hv_traj')]
    if not x_series or not y_series:
        return None
    return _pad_mean(x_series), _pad_mean(y_series)


def _traj_ylim_bottom(curves):
    """Lower y-limit that keeps the discriminating part of the plot legible:
    the worst mean HV any curve still has after TRAJ_YLIM_FROM of the budget.
    Ramp-up from an empty archive spans most of the y-range and squashes the
    part where the handlers actually differ."""
    lows = []
    for x, y in curves:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        cut = np.searchsorted(x, TRAJ_YLIM_FROM * x[-1]) if len(x) else 0
        tail = y[cut:][np.isfinite(y[cut:])]
        if len(tail):
            lows.append(float(np.min(tail)))
    return min(lows) if lows else None


def plot_trajectory_grids(trajectories, refs, plot_dir, plt):
    """One figure per (mode, method): scenario subplots, x = CUMULATIVE
    HIGH-FIDELITY EVALUATIONS (n_evaluated -- the budgeted axis; n_total is
    the archive size and is not monotone), y = feasible HV (mean over seeds),
    colour = handler. Split per full-row method and given the REFERENCE_METHODS
    curves (random, ctaea, ssansga2 -- their fixed handler slots) plus, on
    enumerable scenarios, the exact feasible-HV ceiling: 13 curves in one axes
    made the combined figure unreadable."""
    from matplotlib.lines import Line2D
    written = []
    for mode, by_sid in sorted(trajectories.items()):
        sids = sorted(by_sid, key=lambda s: int(s[1:]))
        methods = [m for m in GRID_METHODS
                   if any(mh[0] == m for by_mh in by_sid.values() for mh in by_mh)]
        for method in methods:
            ncols = 3
            nrows = math.ceil((len(sids) + 1) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
            axes = axes.ravel()
            seen_handlers, seen_extra = set(), set()
            for ax, sid in zip(axes, sids):
                curves = []
                for (m, handler), seeds_dict in sorted(by_sid[sid].items()):
                    if m != method and m not in REFERENCE_METHODS:
                        continue
                    curve = _traj_mean_curve(seeds_dict)
                    if curve is None:
                        continue
                    if m in REFERENCE_METHODS:
                        colour, ls = _ref_style(m, handler)
                        ax.plot(*curve, color=colour, ls=ls, lw=1.2, zorder=1)
                        seen_extra.add((m, handler))
                    else:
                        ax.plot(*curve, color=HANDLER_COLOURS.get(handler, '#333333'), lw=1.3)
                        seen_handlers.add(handler)
                    curves.append(curve)
                ref = refs.get(sid, {})
                hv_true = ref.get('hv_true', float('nan'))
                if np.isfinite(hv_true):
                    ax.axhline(hv_true, color='black', ls='--', lw=0.9, zorder=0)
                    seen_extra.add('true')
                bottom = _traj_ylim_bottom(curves)
                if bottom is not None:
                    tops = [float(np.nanmax(c[1])) for c in curves if len(c[1])]
                    if np.isfinite(hv_true):
                        tops.append(float(hv_true))
                    pad = 0.03 * (max(tops) - bottom) + 1e-9
                    ax.set_ylim(bottom - pad, max(tops) + pad)
                kind = SC.eval_kind(sid)
                ax.set_title(f"{sid}  [{kind}]", fontsize=9)
                ax.set_xlabel('high-fidelity evaluations', fontsize=8)
                ax.set_ylabel('feasible HV', fontsize=8)
                ax.tick_params(labelsize=8)
            leg_ax = axes[len(sids)]
            leg_ax.axis('off')
            handles = [Line2D([0], [0], color=HANDLER_COLOURS.get(h, '#333333'), lw=2, label=h)
                      for h in SC.ALL_HANDLERS if h in seen_handlers]
            for m, h in sorted((x for x in seen_extra if isinstance(x, tuple)),
                               key=_ref_sort_key):
                colour, ls = _ref_style(m, h)
                handles.append(Line2D([0], [0], color=colour, ls=ls, lw=1.2,
                                      label=f'{m} ({h})'))
            if 'true' in seen_extra:
                handles.append(Line2D([0], [0], color='black', ls='--', lw=0.9,
                                      label='HV of true feasible front'))
            leg_ax.legend(handles=handles, loc='center left', fontsize=7.5, frameon=False,
                         title='handler (colour)', title_fontsize=8)
            for ax in axes[len(sids) + 1:]:
                ax.axis('off')
            fig.suptitle(f'Feasible HV vs. evaluations -- {method}, {mode} mode', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out_png = os.path.join(plot_dir, f'traj_{mode}_{method}.png')
            fig.savefig(out_png, dpi=130)
            plt.close(fig)
            written.append(out_png)
    return written


def _extend_staircase(xs, ys, xmax, ymax):
    """(xs, ys) padded so a where='post' step spans the whole view: a riser to
    *ymax* above the left-most point and a run to *xmax* at the right-most.

    An unextended staircase stops at its right-most point, so a front that
    merely reaches further right (a cheaper, worse-error solution the other
    front never sampled) is drawn OUTSIDE a reference front it does not in
    fact dominate anywhere -- e.g. S5, where the samos surface appeared to
    beat the exact feasible front it is provably bounded by."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    finite = np.isfinite(ys)
    if not finite.any():
        return xs, ys
    i0 = int(np.argmax(finite))
    i1 = len(ys) - 1 - int(np.argmax(finite[::-1]))
    xs_ext = np.concatenate([[xs[i0]], xs[i0:i1 + 1], [xmax]])
    ys_ext = np.concatenate([[ymax], ys[i0:i1 + 1], [ys[i1]]])
    return xs_ext, ys_ext


def _draw_front_set(ax, fronts, colour, label, view, lw=1.6, band=True):
    fronts = [np.asarray(f, dtype=float) for f in fronts if f is not None and len(f) > 0]
    if not fronts:
        return
    xmax, ymax = view
    if len(fronts) == 1:
        f = fronts[0][np.argsort(fronts[0][:, 0], kind='stable')]
        xs, ys = _extend_staircase(f[:, 0], np.minimum.accumulate(f[:, 1]), xmax, ymax)
        ax.step(xs, ys, where='post', color=colour, lw=lw, label=label)
        return
    n = len(fronts)
    k_med = math.ceil(n / 2)
    xs, surf = attainment_surfaces(fronts, ks=[1, k_med, n])
    to_nan = lambda a: np.where(np.isfinite(a), a, np.nan)
    xs_med, ys_med = _extend_staircase(xs, to_nan(surf[k_med]), xmax, ymax)
    ax.step(xs_med, ys_med, where='post', color=colour, lw=lw, label=f'{label} (median, n={n})')
    if band:
        ax.fill_between(xs, to_nan(surf[1]), to_nan(surf[n]), step='post', color=colour, alpha=0.15, linewidth=0)


def plot_attainment_grid(sid, mode, ref, fronts_by_mh, out_png, plt):
    """One figure per (sid, mode): subplot per handler, feasible/infeasible
    attainable cloud, true/sampled feasible front, per-method (nsga2, samos)
    attainment staircases, with the REFERENCE_METHODS (random, ctaea,
    ssansga2) repeated in every subplot as a fixed comparison -- one curve per
    (method, slot), so ssansga2's constrained and unconstrained runs stay
    separate rather than being pooled into one front."""
    handlers = [h for h in SC.ALL_HANDLERS
                if any(mh[1] == h and mh[0] in GRID_METHODS for mh in fronts_by_mh)]
    if not handlers:
        return False
    ref_fronts = {mh: fr for mh, fr in fronts_by_mh.items()
                  if mh[0] in REFERENCE_METHODS and fr}
    # random is the floor of the figure, so it keeps a neutral grey; every
    # other reference slot uses its own _ref_style colour.
    ref_colour = lambda m, h: '#999999' if m == 'random' else _ref_style(m, h)[0]

    F_cloud, obj_idx, constr_idx = ref['F_cloud'], ref['obj_idx'], ref['constr_idx']
    tau, sense = ref['tau'], ref['sense']
    obj = F_cloud[:, obj_idx]
    feas = (sense * (F_cloud[:, constr_idx] - tau)) <= 0
    sub = (np.random.RandomState(0).choice(len(F_cloud), _CLOUD_PLOT_MAX, replace=False)
          if len(F_cloud) > _CLOUD_PLOT_MAX else np.arange(len(F_cloud)))
    tf = None
    if ref['true_front'] is not None and len(ref['true_front']) > 0:
        tf = np.asarray(ref['true_front'], dtype=float)
        tf = tf[np.argsort(tf[:, 0], kind='stable')]

    # View box: the fronts, not the cloud. The cloud spans the whole normalized
    # range while every front lives in a corner of it, so an auto-scaled axes
    # squashes the comparison into a few pixels.
    box = [f for f in ([tf] if tf is not None else []) +
          [np.asarray(f, dtype=float) for (m, _h), fr in fronts_by_mh.items() if m != 'random'
           for f in fr] if len(f) > 0]   # random excluded: its front is the widest and would blow up the view
    pts = np.vstack(box)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.06 * np.maximum(hi - lo, 1e-9)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])
    view = (xlim[1], ylim[1])

    ncols = 4
    nrows = math.ceil((len(handlers) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.1 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, handler in zip(axes, handlers):
        s_feas, s_inf = feas[sub], ~feas[sub]
        ax.scatter(obj[sub][s_inf, 0], obj[sub][s_inf, 1], s=2, c=CLOUD_INFEASIBLE_COLOUR,
                  alpha=0.4, linewidths=0, rasterized=True)
        ax.scatter(obj[sub][s_feas, 0], obj[sub][s_feas, 1], s=2, c=CLOUD_FEASIBLE_COLOUR,
                  alpha=0.4, linewidths=0, rasterized=True)
        if tf is not None:
            xs, ys = _extend_staircase(tf[:, 0], np.minimum.accumulate(tf[:, 1]), *view)
            ax.step(xs, ys, where='post', color='black', lw=1.0)
            ax.plot(tf[:, 0], tf[:, 1], '.', color='black', ms=3.5)
        for (m, h), fr in ref_fronts.items():
            _draw_front_set(ax, fr, ref_colour(m, h), m, view, lw=1.0, band=False)
        for method in GRID_METHODS:
            _draw_front_set(ax, fronts_by_mh.get((method, handler), []), METHOD_COLOURS[method],
                           method, view)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(handler, fontsize=9)
        ax.tick_params(labelsize=7)

    from matplotlib.patches import Patch
    leg_ax = axes[len(handlers)]
    leg_ax.axis('off')
    handles = [Patch(color=CLOUD_FEASIBLE_COLOUR, label='feasible cloud'),
              Patch(color=CLOUD_INFEASIBLE_COLOUR, label='infeasible cloud')]
    if tf is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='black', marker='.', lw=1.0,
                              label=('true feasible front' if ref['enumerable']
                                     else 'sampled feasible front')))
    from matplotlib.lines import Line2D
    handles += [Line2D([0], [0], color=ref_colour(m, h), lw=1.0,
                       label=f'{m} / {h} (median attainment)')
               for m, h in sorted(ref_fronts, key=_ref_sort_key)]
    handles += [Line2D([0], [0], color=METHOD_COLOURS[m], lw=1.6, label=f'{m} (median attainment)')
               for m in GRID_METHODS]
    leg_ax.legend(handles=handles, loc='center left', fontsize=8, frameon=False)
    for ax in axes[len(handlers) + 1:]:
        ax.axis('off')

    cloud_note = ('' if ref['enumerable']
                  else f"   [cloud {ref['cloud_label']} -- front is an approximation runs may exceed]")
    scenario = ref['scenario']
    fig.suptitle(f"{sid} / {mode} [{SC.eval_kind(sid)}]: "
                f"{scenario['space']} {scenario['suite']}/pid{scenario['pid']} -- "
                f"{'/'.join(scenario['obj_metrics'])} objectives, "
                f"{scenario['constr_metric']} {'>=' if sense == -1 else '<='} {tau:.4f}{cloud_note}",
                fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def plot_final_bars(metrics_rows, plot_dir, plt):
    """Per sid: hard/soft panels on a SHARED y-axis, x = handler, mean +- std
    over seeds per method, one figure for feasible HV and one for best
    feasible error.

    Points-with-error-bars rather than bars: the panels must share a y-axis
    (the hard-vs-soft gap is the point of the figure, and independently scaled
    panels hide it), and a shared axis is only legible once it can be clipped
    to the data range -- which a bar, anchored at zero, cannot honestly be."""
    written = []
    sids = sorted({r['sid'] for r in metrics_rows}, key=lambda s: int(s[1:]))
    for value_col, ylabel, stem in (('hv_run', 'final feasible HV', 'hv'),
                                    ('best_err', 'best feasible Err.', 'besterr')):
        for sid in sids:
            rows_sid = [r for r in metrics_rows if r['sid'] == sid]
            modes_present = [m for m in SC.MODES if any(r['mode'] == m for r in rows_sid)]
            if not modes_present:
                continue
            fig, axes = plt.subplots(1, len(modes_present), figsize=(5.5 * len(modes_present), 4),
                                     squeeze=False, sharey=True)
            axes = axes[0]
            for ax, mode in zip(axes, modes_present):
                rows_m = [r for r in rows_sid if r['mode'] == mode]
                handlers = sorted({r['handler'] for r in rows_m if r['method'] in GRID_METHODS},
                                  key=_handler_sort_key)
                methods = [m for m in GRID_METHODS if any(r['method'] == m for r in rows_m)]
                offsets = np.linspace(-0.16, 0.16, max(len(methods), 1))
                x = np.arange(len(handlers))
                for mi, method in enumerate(methods):
                    means, stds = [], []
                    for h in handlers:
                        vals = [r[value_col] for r in rows_m if r['method'] == method and r['handler'] == h
                               and np.isfinite(r[value_col])]
                        means.append(np.mean(vals) if vals else np.nan)
                        stds.append(np.std(vals) if vals else np.nan)
                    ax.errorbar(x + offsets[mi], means, yerr=stds, fmt='o', ms=5, capsize=3,
                               lw=1.2, label=method, color=METHOD_COLOURS.get(method, '#333333'))
                # Reference methods do not span the handler axis, so each of
                # their slots is a horizontal line rather than a series.
                for m in REFERENCE_METHODS:
                    for h in SC.fixed_handlers(m, mode):
                        vals = [r[value_col] for r in rows_m if r['method'] == m
                               and r['handler'] == h and np.isfinite(r[value_col])]
                        if not vals:
                            continue
                        colour, ls = _ref_style(m, h)
                        ax.axhline(np.mean(vals), color=colour, ls=ls, lw=1.2,
                                  label=f'{m} ({h})')
                ax.set_xticks(x)
                ax.set_xticklabels(handlers, rotation=35, ha='right', fontsize=7)
                ax.set_xlim(-0.6, len(handlers) - 0.4)
                ax.grid(axis='y', color='0.9', lw=0.6)
                ax.set_axisbelow(True)
                ax.set_title(mode, fontsize=9)
            axes[0].legend(fontsize=7)
            axes[0].set_ylabel(ylabel, fontsize=8)
            kind = SC.eval_kind(sid)
            fig.suptitle(f'{sid} [{kind}]: {ylabel} (mean $\\pm$ std over seeds, shared y-axis)',
                        fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            out_png = os.path.join(plot_dir, f'{sid}_{stem}_bars.png')
            fig.savefig(out_png, dpi=130)
            plt.close(fig)
            written.append(out_png)
    return written


# ─── LaTeX tables ───────────────────────────────────────────────────────────

_EVAL_KIND_TEX = {'tabular': 'tab', 'surrogate': 'sur'}


def _tex_escape(s):
    return str(s).replace('_', r'\_')


def write_handler_winloss_table(winloss_rows, tables_dir, metric='hv_run'):
    """One .tex per method (nsga2, samos): rows = (scenario, mode), columns
    = handlers, cell = aggregated W-L-T count over that handler's pairwise
    tests within the (sid, mode, method) group."""
    written = []
    os.makedirs(tables_dir, exist_ok=True)
    for method in GRID_METHODS:
        rows = [r for r in winloss_rows if r['method'] == method and r['metric'] == metric]
        out_path = os.path.join(tables_dir, f'handler_winloss_{method}.tex')
        if not rows:
            with open(out_path, 'w') as fh:
                fh.write(f'% ---- handler win/loss ({method}, {metric}) skipped: no testable pair ----\n')
            written.append(out_path)
            continue
        handlers = sorted({h for r in rows for h in (r['handler_a'], r['handler_b'])}, key=_handler_sort_key)
        sid_modes = sorted({(r['sid'], r['mode']) for r in rows},
                           key=lambda sm: (int(sm[0][1:]), sm[1]))
        lines = [
            f'% ---- Handler win/loss/tie ({method}, {STATS_METRIC_LABEL[metric]}) ----',
            r'\begin{table*}[t]', r'\centering',
            rf'\caption{{Handler win/loss/tie counts ({method}, {STATS_METRIC_LABEL[metric]}): '
            r'Holm-corrected paired Wilcoxon signed-rank over seeds, one cell per handler '
            r'aggregating its pairwise tests against every other handler present in that '
            r"(scenario, mode). Most-wins handler per row in bold. ``eval'' is how the "
            r'benchmark produces its metrics: \emph{tab}ular lookup (exact, enumerable '
            r'search space) or a \emph{sur}rogate predictor.}',
            rf'\label{{tab:scenario-handler-winloss-{method}}}',
            r'\resizebox{\textwidth}{!}{%',
            rf"\begin{{tabular}}{{lll{'c' * len(handlers)}}}", r'\toprule',
            'scenario & eval & mode & ' + ' & '.join(_tex_escape(h) for h in handlers) + r' \\',
            r'\midrule']
        for sid, mode in sid_modes:
            d = [r for r in rows if r['sid'] == sid and r['mode'] == mode]
            counts = {}
            for h in handlers:
                w = sum(1 for r in d if r['significant'] and (
                    (r['handler_a'] == h and r['outcome_ab'] == 'W') or
                    (r['handler_b'] == h and r['outcome_ab'] == 'L')))
                l = sum(1 for r in d if r['significant'] and (
                    (r['handler_a'] == h and r['outcome_ab'] == 'L') or
                    (r['handler_b'] == h and r['outcome_ab'] == 'W')))
                n_pairs = sum(1 for r in d if h in (r['handler_a'], r['handler_b']))
                counts[h] = (w, l, n_pairs - w - l)
            max_w = max((c[0] for c in counts.values()), default=0)
            row = [sid, _EVAL_KIND_TEX[SC.eval_kind(sid)], mode]
            for h in handlers:
                w, l, t = counts[h]
                txt = f'{w}-{l}-{t}'
                if w == max_w and w > 0:
                    txt = rf'\textbf{{{txt}}}'
                row.append(txt)
            lines.append(' & '.join(row) + r' \\')
        lines.extend([r'\bottomrule', r'\end{tabular}}', r'\end{table*}', ''])
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        written.append(out_path)
    return written


# ─── headline method comparison table ──────────────────────────────────────

# Rows of write_method_comparison_table, in display order. handler=None means
# "whatever slot scenarios.fixed_handlers gives that method in this mode" (the
# mode default for random; methods with more than one slot always name theirs
# explicitly). The constrained/unconstrained row pairs are what the table
# exists for -- same algorithm, constraint enforced by feasibility-first
# domination vs not enforced -- and there are now three flavours of
# "unconstrained" to keep distinct: b0-as-obj DEMOTES the constrained metric
# to an ordinary objective (the algorithm still sees it), while
# b1-unconstrained HIDES it entirely.
COMPARISON_ROWS = [
    ('random',   None,               'random'),
    ('nsga2',    'h4-cdp',           r'NSGA-II (CDP)'),
    ('samos',    'h4-cdp',           r'SAMOS (CDP)'),
    ('ctaea',    None,               r'C-TAEA'),
    ('ssansga2', 'h4-cdp',           r'SSA-NSGA-II (CDP)'),
    ('nsga2',    'b0-as-obj',        r'NSGA-II (b0)'),
    ('samos',    'b0-as-obj',        r'SAMOS (b0)'),
    ('nsga2',    'b1-unconstrained', r'NSGA-II (unc.)'),
    ('samos',    'b1-unconstrained', r'SAMOS (unc.)'),
    ('ssansga2', 'b1-unconstrained', r'SSA-NSGA-II (unc.)'),
]

# Stable colour per comparison row, keyed by LABEL. scikit-posthocs otherwise
# assigns from its palette by RANK POSITION, so a method is drawn in a
# different colour in every CD panel and the colour identifies nothing. Each
# row keeps its method's METHOD_COLOURS hue; the unconstrained rows take a
# lighter same-family tone of their constrained counterpart, so hue reads as
# the algorithm and shade as the constraint treatment -- and with two
# unconstrained flavours per full-row method, b0 takes the mid tone and the
# strictly-unconstrained b1 the lightest.
COMPARISON_COLOURS = {
    'random':              METHOD_COLOURS['random'],
    'NSGA-II (CDP)':       METHOD_COLOURS['nsga2'],
    'SAMOS (CDP)':         METHOD_COLOURS['samos'],
    'C-TAEA':              METHOD_COLOURS['ctaea'],
    'SSA-NSGA-II (CDP)':   REF_SLOT_STYLE[('ssansga2', 'h4-cdp')][0],
    'NSGA-II (b0)':        '#56B4E9',
    'SAMOS (b0)':          '#E69F00',
    'NSGA-II (unc.)':      '#A6DAF0',
    'SAMOS (unc.)':        '#F5D08C',
    'SSA-NSGA-II (unc.)':  REF_SLOT_STYLE[('ssansga2', 'b1-unconstrained')][0],
}

# (metrics_rows key, column header, +1 = higher is better, decimals, scale)
#
# hv_auc, not hv_all. Every other column is a final-budget snapshot, so nothing
# here tested how FAST a method gets to a good feasible front -- the claim a
# surrogate method actually makes. hv_all was the weaker candidate to drop:
# under the hard gate infeasible F is inf, so those members never reach the
# archive and hv_all is bit-identical to hv_run in 83% of hard runs, leaving
# half the pooled Friedman blocks near-degenerate. It stays in
# scenario_metrics.csv, where it still separates the soft rows.
COMPARISON_METRICS = [
    ('hv_run',      r'HV$_{\mathrm{feas}}$',   +1, 3, 1.0),
    ('hv_auc',      r'HV$_{\mathrm{any}}$',    +1, 3, 1.0),
    ('best_acc',    r'Acc$_{\max}$ (\%)',      +1, 2, 1.0),
    ('exact_waste', r'waste (\%)',             -1, 1, 100.0),
]

# Accuracy cells that reach the scenario's best attainable feasible accuracy
# are daggered. The reference is exact only where the space was enumerated
# (best_acc_true is NaN otherwise), so the marker can never fire off a sampled
# optimum a run could legitimately beat. Comparing the CELL MEAN means a dagger
# says every seed found it, not just the luckiest one.
_ACC_OPTIMAL_ATOL = 1e-6


def _comparison_cells(metrics_rows, sid, mode):
    """{row_index: {metric_key: (mean, std, n)}} for one (sid, mode) block,
    over COMPARISON_ROWS. Cells with no seeds are simply absent."""
    out = {}
    for i, (method, handler, _label) in enumerate(COMPARISON_ROWS):
        slot = handler if handler is not None else SC.fixed_handlers(method, mode)[0]
        rows_c = [r for r in metrics_rows if r['sid'] == sid and r['mode'] == mode
                  and r['method'] == method and r['handler'] == slot]
        if not rows_c:
            continue
        cell = {}
        for key, _hdr, _dirn, _dec, scale in COMPARISON_METRICS:
            vals = [r[key] * scale for r in rows_c
                    if np.isfinite(r.get(key, float('nan')))]
            if vals:
                cell[key] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        if cell:
            out[i] = cell
    return out


def write_method_comparison_table(metrics_rows, tables_dir):
    """Headline method comparison: rows = (scenario, method), columns =
    COMPARISON_METRICS under each mode, cell = mean $\\pm$ std over seeds, best
    per column within a scenario block in bold.

    The row set is a fixed ten-way slice of the campaign grid, not every cell:
    random as the floor, nsga2/samos under h4-cdp (constrained), b0-as-obj
    (constraint demoted to an ordinary objective) and b1-unconstrained
    (constraint hidden outright), ctaea as the dedicated constrained MOEA, and
    ssansga2 under both of its slots -- h4-cdp (constraint surrogated, inner
    CDP) and b1-unconstrained. All three b1 rows together are the campaign's
    unconstrained floor, at fixed algorithm and fixed budget against their own
    h4-cdp row. The five remaining handlers answer a different question (which
    handler, given a method) and already have their own win/loss table.

    The two b0 rows do NOT respond to mode the same way, which the caption
    states rather than leaving a reader to infer a null result: build_problem
    returns a B0ObjectiveProblem with no G at all for b0-as-obj, so nsga2's b0
    row is literally the same run under both modes (its _HardGateNSGA2B0 only
    counts, never drops). SAMOS2's b0 row still differs, because SAMOS2 gates
    its own archive off the constrained metric's F column (``gate_g_fn``,
    algorithms.py) whenever mode is hard."""
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, 'method_comparison.tex')
    sids = sorted({r['sid'] for r in metrics_rows}, key=lambda s: int(s[1:]))
    modes = [m for m in SC.MODES if any(r['mode'] == m for r in metrics_rows)]
    blocks = {(sid, mode): _comparison_cells(metrics_rows, sid, mode)
              for sid in sids for mode in modes}
    if not any(blocks.values()):
        with open(out_path, 'w') as fh:
            fh.write('% ---- method comparison skipped: no matching (method, handler) cell ----\n')
        return out_path

    n_seen = [c[2] for blk in blocks.values() for cell in blk.values() for c in cell.values()]
    n_note = (f'{min(n_seen)}' if min(n_seen) == max(n_seen) else f'{min(n_seen)}-{max(n_seen)}')
    n_num = len(COMPARISON_METRICS) * len(modes)

    lines = [
        '% ---- Headline method comparison ----',
        '%',
        f'% Cells:   mean $\\pm$ std over seeds (n = {n_note} per cell); best per column',
        '%          within a scenario block in bold.',
        '% HV_feas: hypervolume of the FEASIBLE non-dominated final archive, against',
        "%          the scenario's own reference point (scenarios.py ref_point).",
        '% HV_any:  ANYTIME quality -- mean feasible HV over the whole evaluation',
        '%          budget (area under the HV-vs-evaluations curve, anchored at zero',
        '%          HV at zero evaluations, divided by the budget spent). HV_feas',
        '%          says where a method finished; this says how fast it got there,',
        '%          and a method can win one and lose the other.',
        '% Acc_max: highest FEASIBLE test accuracy in the final archive, as a true',
        "%          percentage -- the benchmark's normalised error column inverted",
        '%          through its utopian/nadir box and complemented, NOT the normalised',
        '%          value itself (which would read 0 at the best architecture in the',
        '%          space and invite it being mistaken for zero error).',
        '% Dagger:  the cell mean equals the best feasible accuracy attainable anywhere',
        '%          in the space, i.e. EVERY seed found the optimum. Only fires on the',
        '%          enumerable (tabular) scenarios, where that reference is exact; the',
        "%          surrogate scenarios' optimum is a 10k sample a run could beat.",
        '%          S6 daggers because its arithmetic-intensity floor does not bind on',
        '%          the accuracy axis -- both globally most accurate NB201',
        '%          architectures clear it, so the constrained optimum is the',
        '%          unconstrained one.',
        '% waste:   share of high-fidelity evaluations that landed infeasible,',
        '%          from the algorithms own evaluated/feasible counters (not the',
        '%          dominance-filtered archive, which understates it).',
        '% b0 rows: the b0 problem defines no constraint at all, so NSGA-II (b0) is the',
        '%          same run under both modes and repeats across both halves. SAMOS (b0)',
        '%          is not: SAMOS2 still gates its own archive off the constrained',
        '%          objective column under hard, so its two halves genuinely differ.',
        '% random:  runs the scenario/mode default handler slot, h4-cdp under hard and',
        '%          h2-static_penalty under soft.',
        '% eval:    tab = tabular lookup (exact, enumerable), sur = surrogate predictor.',
        '%',
        r'\begin{table*}[t]', r'\centering',
        r'\caption{Constrained vs.\ unconstrained search across the scenario suite: '
        r'final and anytime feasible hypervolume, best feasible accuracy, and the share '
        r'of the evaluation budget spent on infeasible architectures. Hypervolume is measured '
        r"in the benchmark's normalised objective space; accuracy is a true percentage. "
        r'Mean $\pm$ std over '
        r'seeds; best per column within each scenario in bold. The b0 rows demote the '
        r'constraint to an ordinary objective, so NSGA-II~(b0) sees no hard/soft '
        r'distinction at all and repeats across both halves of the table.}',
        r'\label{tab:scenario-method-comparison}',
        r'\resizebox{\textwidth}{!}{%',
        rf"\begin{{tabular}}{{lll{'r' * n_num}}}", r'\toprule',
        r'& & & ' + ' & '.join(rf'\multicolumn{{{len(COMPARISON_METRICS)}}}{{c}}{{{m}}}'
                               for m in modes) + r' \\',
        ' '.join(rf'\cmidrule(lr){{{4 + i * len(COMPARISON_METRICS)}-'
                 rf'{3 + (i + 1) * len(COMPARISON_METRICS)}}}' for i in range(len(modes))),
        'scenario & eval & method & '
        + ' & '.join(hdr for _m in modes for _k, hdr, _d, _p, _s in COMPARISON_METRICS)
        + r' \\',
        r'\midrule']

    daggered = []
    for b, sid in enumerate(sids):
        if not any(blocks[(sid, m)] for m in modes):
            continue
        if b > 0:
            lines.append(r'\midrule')
        acc_true = next((r['best_acc_true'] for r in metrics_rows if r['sid'] == sid
                         and np.isfinite(r.get('best_acc_true', float('nan')))),
                        float('nan'))
        # best per (mode, metric) column across the comparison rows, direction-aware
        best = {}
        for mode in modes:
            for key, _hdr, dirn, _dec, _scale in COMPARISON_METRICS:
                vals = [cell[key][0] for cell in blocks[(sid, mode)].values() if key in cell]
                if vals:
                    best[(mode, key)] = max(vals) if dirn > 0 else min(vals)
        for i, (_method, _handler, label) in enumerate(COMPARISON_ROWS):
            cells = []
            for mode in modes:
                cell = blocks[(sid, mode)].get(i, {})
                for key, _hdr, _dirn, dec, _scale in COMPARISON_METRICS:
                    if key not in cell:
                        cells.append('--')
                        continue
                    mean, std, _n = cell[key]
                    txt = f'{mean:.{dec}f}$\\pm${std:.{dec}f}'
                    if np.isclose(mean, best.get((mode, key), np.nan)):
                        txt = rf'\textbf{{{txt}}}'
                    # rtol=0: np.isclose's DEFAULT rtol=1e-5 is ~9.4e-4 at a 94%
                    # accuracy, wide enough to dagger a cell where one seed in
                    # twenty missed the optimum -- which is exactly the claim
                    # the footnote makes and must not make falsely.
                    if (key == 'best_acc' and np.isfinite(acc_true)
                            and np.isclose(mean, acc_true, rtol=0.0, atol=_ACC_OPTIMAL_ATOL)):
                        txt += r'$^\dagger$'   # dagger outside the bold: it is a marker, not emphasis
                        daggered.append(sid)
                    cells.append(txt)
            head = ([sid, _EVAL_KIND_TEX[SC.eval_kind(sid)]] if i == 0 else ['', ''])
            lines.append(' & '.join(head + [label] + cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}'])
    if daggered:
        # note sits outside the resizebox so it keeps its own font size
        lines += [r'\par\smallskip',
                  r'{\footnotesize\raggedright $^\dagger$ every seed reached the best '
                  r'feasible accuracy attainable anywhere in the search space '
                  r'(exhaustively enumerated).\par}']
    lines.extend([r'\end{table*}', ''])
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out_path


# ─── critical difference (autorank + scikit-posthocs) ──────────────────────

# Blocks for the Friedman test. 'both' pools the modes (N = 6 scenarios x 2
# modes = 12); 'hard'/'soft' split them (N = 6 each, and Nemenyi's CD grows as
# 1/sqrt(N), so those two are reported as a breakdown rather than as the
# headline). One block is one (scenario, mode) cell, and its value is the cell
# MEAN OVER SEEDS -- never one block per seed: seeds of the same scenario are
# repeated runs of one problem, not independent problems, and blocking on them
# would shrink CD by ~sqrt(20) on replication that carries no new information.
CD_SCOPES = ('both', 'hard', 'soft')

# Plain-text metric names for matplotlib titles. COMPARISON_METRICS' headers
# are LaTeX for the .tex table and do not survive being dropped into a plot.
CD_METRIC_LABEL = {
    'hv_run':      'final feasible hypervolume',
    'hv_auc':      'anytime feasible hypervolume',
    'hv_all':      'total hypervolume',
    'best_acc':    'best feasible accuracy',
    'exact_waste': 'wasted evaluations',
}


def _cd_matrix(metrics_rows, metric, scope):
    """(blocks x methods) DataFrame for one (metric, scope), or None when
    fewer than three blocks are complete. Friedman needs every method present
    in every block, so a block missing any of COMPARISON_ROWS is dropped whole
    rather than imputed."""
    import pandas as pd
    modes = [scope] if scope in SC.MODES else list(SC.MODES)
    labels = [lbl for _m, _h, lbl in COMPARISON_ROWS]
    sids = sorted({r['sid'] for r in metrics_rows}, key=lambda s: int(s[1:]))
    data = {}
    for sid in sids:
        for mode in modes:
            cells = _comparison_cells(metrics_rows, sid, mode)
            if len(cells) < len(COMPARISON_ROWS):
                continue
            vals = [cells[i].get(metric, (float('nan'),))[0]
                    for i in range(len(COMPARISON_ROWS))]
            if all(np.isfinite(v) for v in vals):
                data[f'{sid}/{mode}'] = vals
    if len(data) < 3:
        return None
    return pd.DataFrame.from_dict(data, orient='index', columns=labels)


def compute_critical_difference(metrics_rows, alpha=STATS_ALPHA):
    """Friedman + Nemenyi over COMPARISON_ROWS, one analysis per (scope,
    metric) -> {(scope, metric): {frame, ranks, sig, cd, pvalue, omnibus,
    n_blocks, all_normal, homoscedastic}}.

    force_mode='nonparametric' -- autorank would otherwise pick between
    repeated-measures ANOVA + Tukey and Friedman + Nemenyi per analysis, from a
    normality screen run on 12 blocks. Flipping test families between the four
    metric panels of one figure would make their CDs incomparable, and Demsar's
    recommendation for comparing several methods over several problems is the
    nonparametric route regardless. What the screen found is kept in the result
    (all_normal / homoscedastic) and printed, so forcing it hides nothing.

    The pairwise matrix comes from scikit-posthocs' posthoc_nemenyi_friedman,
    which ranks within blocks itself; it is two-sided on rank differences and
    so needs no direction argument, unlike autorank's ``order``."""
    try:
        from autorank import autorank
        import scikit_posthocs as sp
    except ImportError as exc:
        print(f'[analyse] critical difference skipped: {exc} '
              f'(pip install autorank scikit-posthocs).')
        return {}

    import contextlib
    import io

    out = {}
    for scope in CD_SCOPES:
        for metric, _hdr, dirn, _dec, _scale in COMPARISON_METRICS:
            frame = _cd_matrix(metrics_rows, metric, scope)
            if frame is None:
                continue
            order = 'descending' if dirn > 0 else 'ascending'
            # autorank prints its rankdf and its forced-mode notice on every
            # call; the concise per-scope summary below is printed instead.
            with contextlib.redirect_stdout(io.StringIO()):
                res = autorank(frame, alpha=alpha, verbose=False, order=order,
                               force_mode='nonparametric')
            out[(scope, metric)] = dict(
                frame=frame, ranks=res.rankdf['meanrank'],
                sig=sp.posthoc_nemenyi_friedman(frame),
                cd=res.cd, pvalue=res.pvalue, omnibus=res.omnibus,
                n_blocks=len(frame), all_normal=res.all_normal,
                homoscedastic=res.homoscedastic)
    return out


def critical_difference_rows(cd_results):
    """Flat CSV rows: one per (scope, metric, method)."""
    rows = []
    for (scope, metric), r in cd_results.items():
        for method, rank in r['ranks'].items():
            rows.append(dict(
                scope=scope, metric=metric, method=method,
                mean_rank=float(rank), n_blocks=r['n_blocks'], cd=float(r['cd']),
                omnibus=r['omnibus'], omnibus_p=float(r['pvalue']),
                significant=bool(r['pvalue'] < STATS_ALPHA),
                median=float(r['frame'][method].median()),
                all_normal=bool(r['all_normal']), homoscedastic=bool(r['homoscedastic'])))
    return rows


def plot_critical_difference(cd_results, plot_dir, plt):
    """One figure per scope: a Demsar critical-difference diagram per metric,
    drawn by scikit-posthocs from autorank's mean ranks and CD."""
    import scikit_posthocs as sp
    written = []
    for scope in CD_SCOPES:
        panels = [m for m, _hdr, _d, _p, _s in COMPARISON_METRICS
                  if (scope, m) in cd_results]
        if not panels:
            continue
        fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 2.5 * len(panels)),
                                 squeeze=False)
        axes = axes.ravel()
        for ax, metric in zip(axes, panels):
            r = cd_results[(scope, metric)]
            sp.critical_difference_diagram(r['ranks'], r['sig'], cd=r['cd'], ax=ax,
                                           color_palette=COMPARISON_COLOURS)
            # The CD ruler is drawn at a hardcoded y=0.5/0.65 in data
            # coordinates, right where the rank axis puts its tick labels;
            # padding the labels clear of it is the only lever from out here.
            ax.tick_params(axis='x', pad=16)
            sig = 'significant' if r['pvalue'] < STATS_ALPHA else 'NOT significant'
            # pad: scikit-posthocs draws its CD ruler and the rank axis ABOVE
            # the number line, straight into where a default-placed title sits.
            ax.set_title(f'{CD_METRIC_LABEL[metric]}   '
                         f'(Friedman p={r["pvalue"]:.2g}, {sig};  '
                         f'N={r["n_blocks"]} blocks, CD={r["cd"]:.2f})',
                         fontsize=9, pad=26)
        n = cd_results[(scope, panels[0])]['n_blocks']
        scope_label = 'hard + soft pooled' if scope == 'both' else f'{scope} mode only'
        fig.suptitle(f'Critical difference (Friedman + Nemenyi, '
                     f'$\\alpha$={STATS_ALPHA}) -- {scope_label}, N={n}', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_png = os.path.join(plot_dir, f'cd_{scope}.png')
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        written.append(out_png)
    return written


def write_critical_difference_table(cd_results, tables_dir):
    """Mean Friedman ranks: rows = method (a block per scope), columns =
    metric, best (lowest) rank per column within a scope in bold. Rank 1 is
    always best -- autorank's ``order`` already absorbs the fact that waste is
    minimised while the other three are maximised."""
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, 'critical_difference.tex')
    if not cd_results:
        with open(out_path, 'w') as fh:
            fh.write('% ---- critical difference skipped: autorank/scikit-posthocs '
                     'unavailable or too few blocks ----\n')
        return out_path

    metrics = [(m, hdr) for m, hdr, _d, _p, _s in COMPARISON_METRICS
               if any((sc, m) in cd_results for sc in CD_SCOPES)]
    labels = [lbl for _m, _h, lbl in COMPARISON_ROWS]
    lines = [
        '% ---- Friedman mean ranks / critical difference ----',
        '%',
        '% Test:   Friedman omnibus + Nemenyi post-hoc (autorank, forced',
        '%         nonparametric; scikit-posthocs for the pairwise matrix and the',
        '%         diagrams in plots/cd_{both,hard,soft}.png).',
        '% Blocks: one per (scenario, mode) cell, value = mean over seeds. NOT one',
        '%         per seed -- seeds are repeated runs of one problem, not extra',
        '%         problems, and blocking on them would shrink CD by ~sqrt(20) on',
        '%         replication carrying no new information.',
        '% Rank 1: always best; the direction of each metric is already absorbed',
        '%         (waste is minimised, the other three maximised).',
        '% CD:     two methods differ significantly iff their mean ranks differ by',
        '%         more than CD. It scales as 1/sqrt(N), so the hard/soft blocks',
        '%         (N=6) are a breakdown, not the headline -- read `both` first.',
        '%',
        r'\begin{table}[t]', r'\centering',
        r'\caption{Mean Friedman ranks over the scenario suite (rank 1 = best). '
        r'Blocks are (scenario, mode) cells; two methods differ significantly at '
        r'$\alpha=' + str(STATS_ALPHA) + r'$ iff their mean ranks differ by more '
        r'than the critical difference CD.}',
        r'\label{tab:scenario-critical-difference}',
        r'\resizebox{\columnwidth}{!}{%',
        rf"\begin{{tabular}}{{l{'r' * len(metrics)}}}", r'\toprule',
        'method & ' + ' & '.join(hdr for _m, hdr in metrics) + r' \\']
    for scope in CD_SCOPES:
        present = [(m, hdr) for m, hdr in metrics if (scope, m) in cd_results]
        if not present:
            continue
        r0 = cd_results[(scope, present[0][0])]
        scope_label = 'hard + soft pooled' if scope == 'both' else f'{scope} only'
        lines += [r'\midrule',
                  rf"\multicolumn{{{len(metrics) + 1}}}{{l}}{{\textit{{{scope_label}}} "
                  rf"($N={r0['n_blocks']}$, $\mathrm{{CD}}={r0['cd']:.2f}$)}} \\"]
        best = {m: min(cd_results[(scope, m)]['ranks']) for m, _h in present}
        for lbl in labels:
            cells = []
            for m, _hdr in metrics:
                r = cd_results.get((scope, m))
                if r is None or lbl not in r['ranks'].index:
                    cells.append('--')
                    continue
                txt = f"{r['ranks'][lbl]:.2f}"
                if np.isclose(r['ranks'][lbl], best[m], rtol=0.0, atol=1e-9):
                    txt = rf'\textbf{{{txt}}}'
                cells.append(txt)
            lines.append(' & '.join([lbl] + cells) + r' \\')
        lines.append(' & '.join([r'\quad Friedman $p$']
                                + [f"{cd_results[(scope, m)]['pvalue']:.1e}"
                                   if (scope, m) in cd_results else '--'
                                   for m, _h in metrics]) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}', r'\end{table}', ''])
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out_path


def print_critical_difference_summary(cd_results):
    """Mean ranks and the Nemenyi verdict per scope on stdout."""
    print('\n' + '=' * 100)
    print(f'Critical difference (Friedman + Nemenyi, alpha={STATS_ALPHA}), '
          f'blocks = (scenario, mode) cells')
    print('=' * 100)
    if not cd_results:
        print('(autorank / scikit-posthocs unavailable, or too few complete blocks)')
        print('=' * 100)
        return
    for scope in CD_SCOPES:
        present = [(m, hdr) for m, hdr, _d, _p, _s in COMPARISON_METRICS
                   if (scope, m) in cd_results]
        if not present:
            continue
        r0 = cd_results[(scope, present[0][0])]
        print(f"\nscope={scope}  N={r0['n_blocks']} blocks  CD={r0['cd']:.3f}")
        for metric, _hdr in present:
            r = cd_results[(scope, metric)]
            ranked = ', '.join(f'{m} {v:.2f}' for m, v in r['ranks'].sort_values().items())
            flag = '' if r['pvalue'] < STATS_ALPHA else '  [omnibus NOT significant]'
            print(f'  {metric:<12s} p={r["pvalue"]:.2e}{flag}')
            print(f'    {ranked}')
    print('=' * 100)


# Scenarios whose runs collapse to a front this small cannot support a
# front-quality comparison: every test comes back a tie for want of anything to
# separate, which is indistinguishable from a genuine null. Flagged, not hidden.
UNDERPOWERED_FRONT = 1


def _underpowered(metrics_rows, sid):
    sizes = [r['m4_final_front_size'] for r in metrics_rows
             if r['sid'] == sid and np.isfinite(r['m4_final_front_size'])]
    return bool(sizes) and float(np.median(sizes)) <= UNDERPOWERED_FRONT


def _budget_outcomes(trajectories, sid, fracs, metric_dir=1, alpha=STATS_ALPHA,
                     min_seeds=STATS_MIN_SEEDS, exclude=('b0-as-obj',)):
    """hard-wins count per truncated budget, re-testing the hard-vs-soft pairing
    on the HV trajectory cut at each fraction of the run. Answers 'is the verdict
    an artifact of where we stopped?'

    `exclude` handlers are dropped from the *counts* but kept in the Holm family,
    matching compute_hard_vs_soft, which corrects over every (method, handler)
    pair and leaves the filtering to the table. Correcting over a smaller family
    would make these columns systematically more permissive than the W--L--T one
    they sit beside; with it, frac=1.0 reproduces that column exactly."""
    out = []
    for frac in fracs:
        prs = []
        for mh, hseeds in trajectories.get('hard', {}).get(sid, {}).items():
            sseeds = trajectories.get('soft', {}).get(sid, {}).get(mh)
            if not sseeds:
                continue
            common = sorted(set(hseeds) & set(sseeds))
            if len(common) < min_seeds:
                continue
            def cut(seeds, s):
                t = seeds[s]['hv_traj']
                return t[max(0, int(round(len(t) * frac)) - 1)]
            a = np.array([cut(hseeds, s) for s in common], dtype=float)
            b = np.array([cut(sseeds, s) for s in common], dtype=float)
            _, p = _paired_wilcoxon(a, b)
            prs.append((p, float(np.median(a)), float(np.median(b)), mh[1]))
        if not prs:
            out.append(0)
            continue
        wins = 0
        for (p, ma, mb, handler), padj in zip(prs, _holm([x[0] for x in prs])):
            if handler in exclude:
                continue
            if padj < alpha and ((ma > mb) if metric_dir > 0 else (ma < mb)):
                wins += 1
        out.append(wins)
    return out


def write_hard_vs_soft_table(hard_soft_rows, metrics_rows, trajectories, tables_dir,
                             metric='hv_run'):
    """Headline table: per scenario, does soft handling beat hard, and does the
    answer track the suite's predicted soft potential? Rows ordered by soft_10 so
    the trend reads off the page; b0-as-obj excluded (constraint-as-objective has
    no hard/soft distinction)."""
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, 'hard_vs_soft.tex')
    rows = [r for r in hard_soft_rows if r['metric'] == metric and r['handler'] != 'b0-as-obj']
    if not rows:
        with open(out_path, 'w') as fh:
            fh.write(f'% ---- hard-vs-soft ({metric}) skipped: no testable pair ----\n')
        return out_path

    FRACS = (0.25, 0.5, 1.0)
    suite = {r['sid']: r for r in metrics_rows}
    per = {}
    for sid in sorted({r['sid'] for r in rows}, key=lambda s: int(s[1:])):
        d = [r for r in rows if r['sid'] == sid]
        per[sid] = dict(
            hard=sum(1 for r in d if r['better'] == 'hard'),
            soft=sum(1 for r in d if r['better'] == 'soft'),
            tie=sum(1 for r in d if r['better'] == 'tie'),
            soft10=suite[sid]['soft_at_10'], hard10=suite[sid]['hard_at_10'],
            weak=_underpowered(metrics_rows, sid),
            budget=_budget_outcomes(trajectories, sid, FRACS),
        )
    order = sorted(per, key=lambda s: per[s]['soft10'])
    # trend across scenarios: does predicted soft potential track how often hard wins?
    rho = (spearmanr([per[s]['soft10'] for s in order],
                     [per[s]['hard'] for s in order]).correlation
           if len(order) > 2 else float('nan'))
    n_pairs = max((per[s]['hard'] + per[s]['soft'] + per[s]['tie']) for s in order)
    weak = [s for s in order if per[s]['weak']]

    # Detail lives here rather than in the caption, so the surrounding text can
    # absorb whichever parts it needs without a caption that runs half a column.
    lines = [
        f'% ---- Hard vs soft constraint handling ({STATS_METRIC_LABEL[metric]}) ----',
        '%',
        f'% Test:    Holm-corrected paired Wilcoxon signed-rank over seeds, on '
        f'{STATS_METRIC_LABEL[metric]}, alpha={STATS_ALPHA}.',
        f'% W--L--T: tests won by hard / won by soft / tied, over the {n_pairs} '
        f'(method, handler) pairs per scenario.',
        '%          b0-as-obj is in the Holm family but excluded from the counts:',
        '%          constraint-as-objective has no hard/soft distinction.',
        '% Budget:  the same test on the HV trajectory truncated to 25/50/100% of the',
        '%          evaluation budget, reporting hard wins. A budget-dependent verdict',
        '%          would show as a changing count; the 100% column equals W above.',
        f'% Order:   by soft_10, the suite\'s predicted soft potential. '
        f'Spearman(soft_10, hard wins) = {rho:+.2f} over {len(order)} scenarios.',
        f'% Dagger:  flagged automatically at median final front size <= '
        f'{UNDERPOWERED_FRONT}; see the note under the table.',
        '% eval:    tab = tabular lookup (exact, enumerable), sur = surrogate predictor.',
        '% Layout:  single-column; for a full-width float use table* and \\textwidth.',
        '%',
        r'\begin{table}[t]', r'\centering',
        rf'\caption{{Hard vs.\ soft constraint handling ({STATS_METRIC_LABEL[metric]}), '
        r'ordered by predicted soft potential $\mathrm{soft}_{10}$.}',
        r'\label{tab:scenario-hard-vs-soft}',
        r'\resizebox{\columnwidth}{!}{%',
        r'\begin{tabular}{llrrcccc}', r'\toprule',
        r'& & & & & \multicolumn{3}{c}{hard wins @ budget} \\',
        r'\cmidrule(lr){6-8}',
        r'scenario & eval & $\mathrm{soft}_{10}$ & $\mathrm{hard}_{10}$ & W--L--T '
        r'& 25\% & 50\% & 100\% \\',
        r'\midrule']
    for sid in order:
        p = per[sid]
        lines.append(' & '.join([
            sid + (r'$^\dagger$' if p['weak'] else ''),
            _EVAL_KIND_TEX[SC.eval_kind(sid)],
            f'{p["soft10"]:.2f}', f'{p["hard10"]:.2f}',
            f'{p["hard"]}--{p["soft"]}--{p["tie"]}',
            *[str(b) for b in p['budget']],
        ]) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}'])
    if weak:
        # note sits outside the resizebox so it keeps its own font size
        lines += [r'\par\smallskip',
                  r'{\footnotesize\raggedright $^\dagger$ ' + ', '.join(weak) +
                  r': runs collapse to a single-point front, so every test ties for '
                  r'want of resolution rather than absence of effect.\par}']
    lines.extend([r'\end{table}', ''])
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out_path


def write_method_effect_summary_table(method_effect_rows, tables_dir, metric='hv_run'):
    """Condensed counterpart to write_method_effect_table: one row per (scenario,
    mode) with win counts over handlers instead of a cell per handler. The full
    per-handler grid says the same thing in 7x the width."""
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, 'method_effect_summary.tex')
    rows = [r for r in method_effect_rows if r['metric'] == metric and r['handler'] != 'b0-as-obj']
    if not rows:
        with open(out_path, 'w') as fh:
            fh.write(f'% ---- method-effect summary ({metric}) skipped ----\n')
        return out_path
    sid_modes = sorted({(r['sid'], r['mode']) for r in rows}, key=lambda sm: (int(sm[0][1:]), sm[1]))
    n_h = len({r['handler'] for r in rows})
    lines = [
        f'% ---- Method effect summary (nsga2 vs samos, {STATS_METRIC_LABEL[metric]}) ----',
        '%',
        f'% Test:  Holm-corrected paired Wilcoxon signed-rank over seeds, on '
        f'{STATS_METRIC_LABEL[metric]}, alpha={STATS_ALPHA}.',
        f'% Cells: how many of the {n_h} constraint handlers each method wins, per '
        f'(scenario, mode); b0-as-obj excluded.',
        '%        The condensed form of tab:scenario-method-effect, which gives the',
        '%        same result one cell per handler.',
        "% Point to make in the text: samos's advantage is confined to hard handling",
        '%        (totals below) -- under soft handling neither method has an edge.',
        '% eval:  tab = tabular lookup (exact, enumerable), sur = surrogate predictor.',
        '%',
        r'\begin{table}[t]', r'\centering',
        rf'\caption{{Method effect (nsga2 vs.\ samos, {STATS_METRIC_LABEL[metric]}), '
        rf'handler wins per scenario and mode.}}',
        r'\label{tab:scenario-method-effect-summary}',
        r'\resizebox{\columnwidth}{!}{%',
        r'\begin{tabular}{lllccc}', r'\toprule',
        r'scenario & eval & mode & samos & nsga2 & tie \\', r'\midrule']
    for sid, mode in sid_modes:
        d = [r for r in rows if r['sid'] == sid and r['mode'] == mode]
        lines.append(' & '.join([
            sid, _EVAL_KIND_TEX[SC.eval_kind(sid)], mode,
            str(sum(1 for r in d if r['better'] == 'samos')),
            str(sum(1 for r in d if r['better'] == 'nsga2')),
            str(sum(1 for r in d if r['better'] == 'tie')),
        ]) + r' \\')
    totals = [sum(1 for r in rows if r['mode'] == m and r['better'] == b)
              for m in ('hard', 'soft') for b in ('samos', 'nsga2', 'tie')]
    lines.extend([
        r'\midrule',
        r'\multicolumn{3}{l}{total hard} & ' + ' & '.join(str(t) for t in totals[:3]) + r' \\',
        r'\multicolumn{3}{l}{total soft} & ' + ' & '.join(str(t) for t in totals[3:]) + r' \\',
        r'\bottomrule', r'\end{tabular}}', r'\end{table}', ''])
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out_path


def write_method_effect_table(method_effect_rows, tables_dir, metric='hv_run'):
    """nsga2 vs samos, same handler: rows = (scenario, mode), columns =
    handlers, cell = winner ('nsga2'/'samos') or 'tie'."""
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, 'method_effect.tex')
    rows = [r for r in method_effect_rows if r['metric'] == metric]
    if not rows:
        with open(out_path, 'w') as fh:
            fh.write(f'% ---- method-effect ({metric}) skipped: no testable handler ----\n')
        return out_path
    handlers = sorted({r['handler'] for r in rows}, key=_handler_sort_key)
    sid_modes = sorted({(r['sid'], r['mode']) for r in rows}, key=lambda sm: (int(sm[0][1:]), sm[1]))
    lines = [
        f'% ---- Method effect (nsga2 vs samos, same handler, {STATS_METRIC_LABEL[metric]}) ----',
        r'\begin{table*}[t]', r'\centering',
        rf'\caption{{Method effect (nsga2 vs.\ samos, same handler, {STATS_METRIC_LABEL[metric]}): '
        r'Holm-corrected paired Wilcoxon signed-rank over seeds, per (scenario, mode), across every '
        r'handler present under both methods. Cell = winning method, or ``tie" when not '
        r"Holm-significant. ``eval'' is how the benchmark produces its metrics: "
        r'\emph{tab}ular lookup (exact, enumerable search space) or a \emph{sur}rogate '
        r'predictor.}',
        r'\label{tab:scenario-method-effect}', r'\resizebox{\textwidth}{!}{%',
        rf"\begin{{tabular}}{{lll{'c' * len(handlers)}}}", r'\toprule',
        'scenario & eval & mode & ' + ' & '.join(_tex_escape(h) for h in handlers) + r' \\',
        r'\midrule']
    for sid, mode in sid_modes:
        d = {r['handler']: r for r in rows if r['sid'] == sid and r['mode'] == mode}
        row = ([sid, _EVAL_KIND_TEX[SC.eval_kind(sid)], mode]
               + [d[h]['better'] if h in d else '--' for h in handlers])
        lines.append(' & '.join(row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}', r'\end{table*}', ''])
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out_path


# ─── main ───────────────────────────────────────────────────────────────────

def _cache_is_fresh(results_root, output_dir):
    """True when plot_data.pkl is newer than every seed_*.pkl under
    results_root, i.e. re-analysing would reproduce the same cache."""
    cache_path = os.path.join(output_dir, 'plot_data.pkl')
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    newest = max((os.path.getmtime(p)
                  for p in glob.iglob(os.path.join(results_root, '**', 'seed_*.pkl'),
                                      recursive=True)),
                 default=None)
    return newest is not None and newest <= cache_mtime


def plots_only(args):
    """Regenerate plots + LaTeX tables from plot_data.pkl / the CSVs, no pkl
    re-analysis."""
    cache_path = os.path.join(args.output_dir, 'plot_data.pkl')
    if not os.path.exists(cache_path):
        print(f'[analyse] --plots_only: missing {cache_path} (run the full analysis first).')
        return 1
    with open(cache_path, 'rb') as fh:
        cache = pickle.load(fh)
    # caches written before the hard-vs-soft table existed have no such key;
    # the CSV alongside them carries the same rows
    hard_soft_rows = cache.get('hard_soft_rows')
    if not hard_soft_rows:
        hard_soft_rows = _load_hard_vs_soft_csv(os.path.join(args.output_dir, 'hard_vs_soft.csv'))
    render_outputs(args.output_dir, cache['refs'], cache['fronts'], cache['trajectories'],
                   cache['metrics_rows'], cache['winloss_rows'], cache['method_effect_rows'],
                   hard_soft_rows)
    return 0


def _load_hard_vs_soft_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ('median_hard', 'median_soft', 'statistic', 'p_raw', 'p_holm'):
            r[k] = float(r[k])
        r['n'] = int(r['n'])
        r['significant'] = r['significant'] == 'True'
    return rows


def render_outputs(output_dir, refs, fronts, trajectories, metrics_rows, winloss_rows,
                   method_effect_rows, hard_soft_rows=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    written = plot_trajectory_grids(trajectories, refs, plot_dir, plt)
    for (sid, mode), fronts_by_mh in sorted(fronts.items()):
        ref = refs.get(sid)
        if ref is None:
            continue
        out_png = os.path.join(plot_dir, f'{sid}_{mode}_attainment.png')
        if plot_attainment_grid(sid, mode, ref, fronts_by_mh, out_png, plt):
            written.append(out_png)
    written.extend(plot_final_bars(metrics_rows, plot_dir, plt))

    # Derived purely from metrics_rows, so it lives here rather than in main():
    # that keeps --plots_only able to regenerate the whole CD analysis (CSV
    # included) without re-reading a single result pkl.
    cd_results = compute_critical_difference(metrics_rows)
    written.extend(plot_critical_difference(cd_results, plot_dir, plt))
    print(f'[analyse] wrote {len(written)} plot(s) -> {plot_dir}')

    cd_rows = critical_difference_rows(cd_results)
    cd_csv = os.path.join(output_dir, 'critical_difference.csv')
    with open(cd_csv, 'w', newline='') as fh:
        if cd_rows:
            writer = csv.DictWriter(fh, fieldnames=list(cd_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cd_rows)
    print(f'[analyse] wrote {len(cd_rows)} rows -> {cd_csv}')

    written_tex = write_handler_winloss_table(winloss_rows, tables_dir)
    written_tex.append(write_method_effect_table(method_effect_rows, tables_dir))
    written_tex.append(write_method_effect_summary_table(method_effect_rows, tables_dir))
    written_tex.append(write_method_comparison_table(metrics_rows, tables_dir))
    written_tex.append(write_critical_difference_table(cd_results, tables_dir))
    if len(hard_soft_rows):
        written_tex.append(write_hard_vs_soft_table(hard_soft_rows, metrics_rows,
                                                    trajectories, tables_dir))
    print(f'[analyse] wrote {len(written_tex)} LaTeX table(s) -> {tables_dir}')
    print_critical_difference_summary(cd_results)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    suite_table = _load_scenario_suite_table(_SCENARIOS_CSV)

    refs = {}
    metrics_rows = []
    stat_records = []
    trajectories = {mode: {} for mode in args.modes}       # mode -> sid -> (method,handler) -> seed -> {...}
    fronts = {}                                             # (sid, mode) -> (method,handler) -> [fronts]
    warned_meta_mismatch = set()

    for sid in args.scenarios:
        ref = _build_ref(sid, args.enum_limit, args.cache_dir)
        refs[sid] = ref
        suite_row = suite_table.get(sid, {})

        for mode in args.modes:
            pairs = _pairs_for_mode(mode, args.methods, args.handlers)
            for method, handler in pairs:
                run_dir = os.path.join(args.results_root, sid, mode, method, handler)
                seeds = args.seeds if args.seeds is not None else _discover_seeds(run_dir)
                for seed in seeds:
                    path = os.path.join(run_dir, f'seed_{seed}.pkl')
                    if not os.path.exists(path):
                        continue
                    try:
                        with open(path, 'rb') as fh:
                            data = pickle.load(fh)
                    except Exception as exc:
                        print(f'  [analyse] WARN: failed to load {path}: {exc}; skipping.')
                        continue

                    meta = data.get('meta') or {}
                    if meta and (meta.get('sid') != sid or meta.get('mode') != mode):
                        key = (sid, mode, method, handler)
                        if key not in warned_meta_mismatch:
                            warned_meta_mismatch.add(key)
                            print(f'  [analyse] WARN: {path}: meta says '
                                 f'{meta.get("sid")}/{meta.get("mode")} -- mismatched pkl, skipping its rows.')
                        continue

                    m = analyse_run_scenario(data, ref['obj_idx'], ref['constr_idx'], ref['tau'],
                                            ref['sense'], ref['ref_point'], ref['benchmark'], ref['cache'])
                    hv_run = _hv(m['final_front'], ref['ref_point'])
                    hv_ratio = (hv_run / ref['hv_true']) if ref['enumerable'] and ref['hv_true'] > 0 else float('nan')
                    n_feasible_final = float(data['n_feasible'][-1]) if data.get('n_feasible') else float('nan')
                    n_total_final = float(data['n_total'][-1]) if data.get('n_total') else float('nan')

                    row = dict(
                        sid=sid, mode=mode, method=method, handler=handler, seed=seed,
                        n_gen=meta.get('n_gen', m['n_gen']),
                        n_eval_realised=meta.get('n_eval_realised', float('nan')),
                        pop_size=meta.get('pop_size', float('nan')), n_evals=meta.get('n_evals', float('nan')),
                        hv_run=hv_run, hv_all=m['hv_all'], hv_auc=m['hv_auc'],
                        hv_true=ref['hv_true'], hv_ratio=hv_ratio,
                        best_err=m['m8_best_err'], best_err_slack=m['m8_best_err_slack'],
                        best_err_true=ref['best_err_true'],
                        best_acc=_accuracy_pct(m['m8_best_err'], ref['err_utopian'],
                                               ref['err_nadir']),
                        best_acc_true=ref['best_acc_true'],
                        n_feasible_final=n_feasible_final, n_total_final=n_total_final,
                        m3_final_infeasible_waste=m['m3_final_cumulative'], exact_waste=m['exact_waste'],
                        m4_final_front_size=m['m4_final'],
                        m6_min_slack=m['m6_min_slack'], m6_median_slack=m['m6_median_slack'],
                        soft_hv=m['soft_hv'], eval_kind=ref['eval_kind'],
                        enumerable=ref['enumerable'],
                        h1_rejected_count=meta.get('h1_rejected_count', float('nan')),
                        hard_at_10=suite_row.get('hard_at_10', float('nan')),
                        soft_at_10=suite_row.get('soft_at_10', float('nan')),
                        rho=suite_row.get('rho', float('nan')),
                    )
                    metrics_rows.append(row)
                    stat_records.append(dict(sid=sid, mode=mode, method=method, handler=handler, seed=seed,
                                            hv_run=hv_run, best_err=m['m8_best_err']))
                    trajectories[mode].setdefault(sid, {}).setdefault((method, handler), {})[seed] = dict(
                        hv_traj=m['hv_traj'], n_total_traj=m['n_total_traj'],
                        n_eval_traj=m['n_eval_traj'])
                    fronts.setdefault((sid, mode), {}).setdefault((method, handler), []).append(m['final_front'])

        if args.cache_dir:
            _save_eval_cache(args.cache_dir, sid, _eval_cache_key(sid, args.enum_limit), ref)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'scenario_metrics.csv')
    fieldnames = list(metrics_rows[0].keys()) if metrics_rows else [
        'sid', 'mode', 'method', 'handler', 'seed']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f'\n[analyse] wrote {len(metrics_rows)} rows -> {csv_path}')

    # ── stats ────────────────────────────────────────────────────────────────
    winloss_rows = compute_handler_winloss(stat_records)
    method_effect_rows = compute_method_effect(stat_records)
    baseline_rows = compute_baseline_effect(stat_records)
    hard_soft_rows = compute_hard_vs_soft(stat_records)

    for name, rows in (('handler_winloss.csv', winloss_rows),
                       ('method_effect.csv', method_effect_rows),
                       ('baseline_effect.csv', baseline_rows),
                       ('hard_vs_soft.csv', hard_soft_rows)):
        path = os.path.join(args.output_dir, name)
        with open(path, 'w', newline='') as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f'[analyse] wrote {len(rows)} rows -> {path}')

    # ── trajectories JSON ────────────────────────────────────────────────────
    traj_json = {mode: {sid: {f'{m}:{h}': {str(s): v for s, v in seeds.items()}
                              for (m, h), seeds in by_mh.items()}
                        for sid, by_mh in by_sid.items()}
                for mode, by_sid in trajectories.items()}
    json_path = os.path.join(args.output_dir, 'scenario_trajectories.json')
    with open(json_path, 'w') as fh:
        json.dump(traj_json, fh, indent=2)
    print(f'[analyse] wrote trajectories -> {json_path}')

    # ── plot cache ───────────────────────────────────────────────────────────
    # the live benchmark (patched closures) is unpicklable and the eval cache is
    # huge; neither is used by render_outputs
    refs_picklable = {sid: {k: v for k, v in ref.items() if k not in ('benchmark', 'cache')}
                      for sid, ref in refs.items()}
    plot_cache = dict(refs=refs_picklable, fronts=fronts, trajectories=trajectories,
                      metrics_rows=metrics_rows, winloss_rows=winloss_rows,
                      method_effect_rows=method_effect_rows, hard_soft_rows=hard_soft_rows)
    cache_path = os.path.join(args.output_dir, 'plot_data.pkl')
    with open(cache_path, 'wb') as fh:
        pickle.dump(plot_cache, fh)
    print(f'[analyse] wrote plot cache -> {cache_path}')

    if not args.no_plots:
        render_outputs(args.output_dir, refs, fronts, trajectories, metrics_rows,
                       winloss_rows, method_effect_rows, hard_soft_rows)

    # ── stdout summary ──────────────────────────────────────────────────────
    print('\n' + '=' * 118)
    print('Scenario_run feasibility-region metrics summary (mean over seeds)')
    print('=' * 118)
    for sid in args.scenarios:
        rows_sid = [r for r in metrics_rows if r['sid'] == sid]
        if not rows_sid:
            continue
        suite_row = suite_table.get(sid, {})
        suite_note = (f"  [suite: hard@10%={suite_row['hard_at_10']:.3f} "
                     f"soft@10%={suite_row['soft_at_10']:.3f} rho={suite_row['rho']:.3f}]"
                     if suite_row else '')
        sc = SC.SCENARIOS[sid]
        print(f'\nScenario {sid} ({sc["space"]}, {SC.eval_kind(sid)}){suite_note}')
        for mode in args.modes:
            rows_m = [r for r in rows_sid if r['mode'] == mode]
            if not rows_m:
                continue
            print(f'  mode={mode}')
            print(f'  {"method:handler":>26s}  {"n":>2s}  {"hv_run":>18s}  {"best_err":>18s}  '
                 f'{"n_feas/tot":>12s}  {"waste":>8s}')
            for method in args.methods:
                handlers_m = sorted({r['handler'] for r in rows_m if r['method'] == method}, key=_handler_sort_key)
                for handler in handlers_m:
                    rows_h = [r for r in rows_m if r['method'] == method and r['handler'] == handler]
                    n = len(rows_h)
                    hv_vals = [r['hv_run'] for r in rows_h]
                    err_vals = [r['best_err'] for r in rows_h if np.isfinite(r['best_err'])]
                    nf = np.nanmean([r['n_feasible_final'] for r in rows_h])
                    nt = np.nanmean([r['n_total_final'] for r in rows_h])
                    waste = np.nanmean([r['exact_waste'] for r in rows_h])
                    hv_str = f'{np.mean(hv_vals):.4f}+-{np.std(hv_vals):.4f}'
                    err_str = f'{np.mean(err_vals):.4f}+-{np.std(err_vals):.4f}' if err_vals else 'n/a'
                    name = f'{method}:{handler}'
                    print(f'  {name:>26s}  {n:>2d}  {hv_str:>18s}  {err_str:>18s}  '
                         f'{nf:>5.1f}/{nt:<5.1f}  {waste:>7.3f}')
    print('=' * 118)
    print_winloss_matrices(winloss_rows)
    print_method_effect_summary(method_effect_rows)
    print_baseline_effect_summary(baseline_rows)
    print_hard_vs_soft_summary(hard_soft_rows)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_root', default=_DEFAULT_RESULTS_ROOT,
                   help='Root written by run_scenario.py. Point at a dedicated smoke-test '
                        'folder when testing -- never analyse test data under results2/ '
                        '(see CLAUDE.md).')
    p.add_argument('--output_dir', default=None, help='Default: {results_root}/analysis.')
    p.add_argument('--scenarios', nargs='+', default=list(SC.SCENARIOS), choices=list(SC.SCENARIOS))
    p.add_argument('--modes', nargs='+', default=list(SC.MODES), choices=list(SC.MODES))
    p.add_argument('--methods', nargs='+', default=list(SC.METHODS), choices=SC.METHODS)
    p.add_argument('--handlers', nargs='+', default=list(SC.ALL_HANDLERS),
                   choices=SC.ALL_HANDLERS,
                   help='Handler subset. Defaults to the 7-handler row PLUS '
                        'b1-unconstrained, which nsga2, samos and ssansga2 all '
                        'run as the unconstrained floor; the per-handler figures '
                        'still key their handler axis off GRID_METHODS only.')
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='Seed subset (default: discover every seed_*.pkl present per cell).')
    p.add_argument('--enum_limit', type=int, default=200_000,
                   help='Max search-space cardinality for exhaustive true-front enumeration.')
    p.add_argument('--plots_only', action='store_true',
                   help='Regenerate plots + LaTeX tables from plot_data.pkl without re-analysing pkls.')
    p.add_argument('--no_plots', action='store_true',
                   help='Skip plot/table rendering (CSV + JSON + stdout only).')
    p.add_argument('--force', action='store_true',
                   help='Re-analyse even when plot_data.pkl is newer than every result pkl.')
    p.add_argument('--cache_dir', default=None,
                   help='Persistent genotype->objective cache, reused across analyses '
                        '(default: {output_dir}/eval_cache).')
    p.add_argument('--no_cache', action='store_true',
                   help='Ignore and do not write the genotype->objective cache.')
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_root, 'analysis')
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.output_dir, 'eval_cache')
    if args.no_cache:
        args.cache_dir = None
    if not args.plots_only and not args.force and _cache_is_fresh(args.results_root, args.output_dir):
        print('[analyse] plot_data.pkl is newer than every seed_*.pkl -- reusing it '
              '(--force to re-analyse).')
        if args.no_plots:
            sys.exit(0)
        args.plots_only = True
    if args.plots_only:
        sys.exit(plots_only(args))
    sys.exit(main(args))
