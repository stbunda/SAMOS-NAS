"""experiments2/scenario_run/analyse_scenarios.py -- post-hoc analysis for the
scenario_run S1-S8 campaign (scenarios.py). Consumes the per-seed pkls written
by run_scenario.py:

  {results_root}/{sid}/{mode}/{method}/{handler}/seed_{N}.pkl

Every pkl is self-describing via its ``meta`` dict (see run_scenario.run_single).

Grid, vs. experiments/constraint2/analyse_constraint.py
--------------------------------------------------------
constraint2's comparison axis was HANDLER (methods samos/samos-cheap, one
instance per scenario). Here the grid is scenario x mode x method x handler:
  scenarios : S1..S8 (scenarios.SCENARIOS), one fixed (space, suite, pid,
              objectives, constraint) per sid -- no per-instance discovery
              needed, unlike constraint2's multi-instance-per-scenario tree.
  modes     : hard / soft, SAME tau -- they differ only in TREATMENT (gate
              vs. archive-and-penalise), not in what counts as feasible.
  methods   : nsga2, samos (the comparison the user cares about); random is
              a reference row that exists ONLY for scenarios.DEFAULT_HANDLER
              [mode] -- a RAGGED grid, handled explicitly everywhere below
              (never assumed present for every handler).
  handlers  : the 7 in scenarios.HANDLERS, run under BOTH modes (mode does
              not gate which handlers are legal, only the default).

Reuse from analyse_constraint.py
---------------------------------
Imported and reused UNCHANGED: true_feasible_front, sample_cloud,
enumerate_full_space, space_size, attainment_surfaces, _hv, _paired_wilcoxon,
_holm, _pad_mean, plus two lower-level, direction-agnostic helpers analyse_run
itself is built from: _full_reeval (true-eval + normalize an archive, no
feasibility direction baked in) and _norm_once.

NOT reused: analyse_run and _build_ref.
  - _build_ref: instructed to write local (its assert cross-checks
    run_constraint.py's DESIGN_FEASIBLE_FRACTION dicts, which do not exist
    here; scenarios.py's own ``feasible_fraction`` is the right reference).
  - analyse_run: hardcodes REF_POINT = np.ones(2) * 1.05 as a MODULE
    GLOBAL for both its per-generation hv_traj and its soft_hv -- exactly
    the bug this campaign must avoid (S6/S7 need (1.05, 2.626), or ~70% of
    MoSegNAS architectures silently score zero HV). Monkeypatching that
    global per call was considered and rejected as fragile spooky-action-
    at-a-distance; analyse_run_scenario below is a local, single-constraint,
    ref_point-parameterised port of its M1-M8/soft-HV logic instead, reusing
    _full_reeval/_hv/NonDominatedSorting for the actual numeric work.
  - true_feasible_front's own violation formula is ceiling-only (ravelled
    into its _max_violation helper). Reused as-is for S1-S7; for S8 (a
    floor: feasible <=> metric >= tau) _true_feasible_front_signed below
    reflects the constrained column around tau (metric' = 2*tau - metric)
    before calling it and reflects the returned column back afterwards --
    an algebraic involution (feasible <=> metric' <= tau <=> metric >= tau),
    so the ceiling-only helper is reused exactly, not forked.
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
  {output_dir}/hard_vs_soft.csv         hard vs soft, same (method, handler),
                                        grouped within sid.
  {output_dir}/plot_data.pkl            everything the plot/table layer needs
                                        (--plots_only reruns rendering only).
  {output_dir}/plots/traj_{mode}.png            feasible HV vs evaluations,
                                        scenario subplots, colour=handler,
                                        linestyle=method.
  {output_dir}/plots/{sid}_{mode}_attainment.png   subplot per handler: cloud,
                                        true/sampled feasible front, per-method
                                        attainment surfaces.
  {output_dir}/plots/{sid}_final_bars.png   hard/soft subplots, x=handler,
                                        bars=method, final feasible HV +
                                        best feasible error.
  {output_dir}/tables/handler_winloss_{method}.tex   W/L/T per handler, per
                                        (scenario, mode) row.
  {output_dir}/tables/method_effect.tex   nsga2 vs samos outcome per handler,
                                        per (scenario, mode) row.
  stdout summary table + win/tie/loss blocks.

Examples
--------
  python experiments2/scenario_run/analyse_scenarios.py
  python experiments2/scenario_run/analyse_scenarios.py --scenarios S1 S6 --seeds 0 1 2
  python experiments2/scenario_run/analyse_scenarios.py --results_root smoke_tests/scenario_run \\
      --output_dir smoke_tests/scenario_run/analysis
"""

import argparse
import csv
import itertools
import json
import math
import os
import pickle
import sys

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_CONSTRAINT2_DIR = os.path.abspath(os.path.join(_REPO_ROOT, 'experiments', 'constraint2'))
for _p in (_CONSTRAINT2_DIR, _REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from analyse_constraint import (true_feasible_front, sample_cloud, enumerate_full_space,
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

HANDLER_COLOURS = {
    'h1-rejection':      '#1f77b4',
    'h2-static_penalty': '#ff7f0e',
    'h3-adaptive':       '#2ca02c',
    'h4-cdp':            '#d62728',
    'h5-epsilon':        '#9467bd',
    'h6-DSR':            '#8c564b',
    'b0-as-obj':         '#7f7f7f',
}
METHOD_LINESTYLE = {'samos': '-', 'nsga2': '--', 'random': ':'}
METHOD_COLOURS   = {'nsga2': '#0072B2', 'samos': '#D55E00', 'random': '#009E73'}
CLOUD_FEASIBLE_COLOUR   = '#b8d4ea'
CLOUD_INFEASIBLE_COLOUR = '#f4c7b8'


def _handler_sort_key(handler):
    try:
        return (SC.HANDLERS.index(handler), handler)
    except ValueError:
        return (len(SC.HANDLERS), handler)


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
    <= 0). For a floor constraint (sense=-1, S8: feasible <=> metric >= tau)
    reflect the constrained column around tau -- metric' = 2*tau - metric --
    before calling it (feasible <=> metric' <= tau <=> metric >= tau, an
    algebraic involution), then reflect the returned constraint column back
    to the real metric so callers never see the reflected value."""
    if sense == -1:
        F_signed = np.array(F_all, dtype=float, copy=True)
        F_signed[:, constr_idx] = 2.0 * tau - F_signed[:, constr_idx]
    else:
        F_signed = F_all
    frac, front, front_c = true_feasible_front(F_signed, obj_indices, [constr_idx], [tau])
    if sense == -1 and len(front_c) > 0:
        front_c = front_c.copy()
        front_c[:, -1] = 2.0 * tau - front_c[:, -1]
    return frac, front, front_c


# ─── per-scenario reference (local _build_ref; see module docstring) ───────

def _build_ref(sid, enum_limit):
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
    if enumerable:
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

    cache = {tuple(int(v) for v in x): np.asarray(f, dtype=float) for x, f in zip(X_cloud, F_cloud)}

    return dict(sid=sid, scenario=scenario, benchmark=benchmark, obj_idx=obj_idx,
                constr_idx=constr_idx, tau=tau, sense=sense, ref_point=ref_point,
                enumerable=enumerable, n_space=n_space, F_cloud=F_cloud, X_cloud=X_cloud,
                cloud_label=cloud_label, true_front=true_front, true_front_c=true_front_c,
                hv_true=hv_true, best_err_true=best_err_true, frac=frac, cache=cache)


# ─── per-run metrics (local analyse_run; see module docstring) ─────────────

def analyse_run_scenario(data, obj_idx, constr_idx, tau, sense, ref_point, benchmark, cache):
    """Per-run metrics: feasibility trajectories, feasible-HV trajectory
    (ref_point-aware), M2/M6/M8-style boundary metrics and soft-HV, single
    constraint, sense-aware. Mirrors analyse_constraint.analyse_run's logic
    (re-eval every generation's var_archive via _full_reeval, distinguish
    feasible/infeasible via the signed violation, feasible HV against the
    scenario's own ref_point) -- see module docstring for why analyse_run
    itself is not reused directly."""
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
        shifted = F_last_fin[:, obj_idx] + np.maximum(0.0, v_last)[:, None]
        nd_idx_soft = NonDominatedSorting().do(shifted, only_non_dominated_front=True)
        soft_hv = _hv(shifted[nd_idx_soft], ref_point)
    else:
        soft_hv = 0.0

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
        m2_evals_to_first_feasible=m2_evals_to_first_feasible, m2_censored=m2_censored,
        m2_n_evaluated=m2_n_evaluated,
        m3_final_cumulative=float(m3_cumulative[-1]) if n_gen > 0 else float('nan'),
        m4_final=m4_front_size[-1] if m4_front_size else 0,
        m6_min_slack=m6_min_slack, m6_median_slack=m6_median_slack,
        m8_best_err=m8_best_err, m8_best_err_slack=m8_best_err_slack,
        soft_hv=soft_hv, exact_waste=exact_waste, exact_waste_traj=exact_waste_traj,
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
    """(method, handler) work list for one mode: random only exists for
    scenarios.DEFAULT_HANDLER[mode] -- a ragged grid, mirroring
    run_scenario.main()'s own pairing rule exactly."""
    default_handler = SC.DEFAULT_HANDLER[mode]
    pairs = []
    for method in methods:
        for handler in handlers:
            if method == 'random' and handler != default_handler:
                continue
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
        if r['method'] == 'random':
            continue
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


# ─── plotting ───────────────────────────────────────────────────────────────

def plot_trajectory_grids(trajectories, plot_dir, plt):
    """One figure per mode: scenario subplots, x = evaluations (mean n_total
    per generation over seeds), y = feasible HV (mean over seeds), colour =
    handler, linestyle = method. Evaluation-budgeted campaign -- see module
    docstring: never plotted against generation index."""
    from matplotlib.lines import Line2D
    written = []
    for mode, by_sid in sorted(trajectories.items()):
        sids = sorted(by_sid, key=lambda s: int(s[1:]))
        if not sids:
            continue
        ncols = 3
        nrows = math.ceil((len(sids) + 1) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
        axes = axes.ravel()
        seen_handlers, seen_methods = set(), set()
        for ax, sid in zip(axes, sids):
            for method, handler in sorted(by_sid[sid]):
                seeds_dict = by_sid[sid][(method, handler)]
                x_series = [v['n_total_traj'] for v in seeds_dict.values() if v.get('n_total_traj')]
                y_series = [v['hv_traj'] for v in seeds_dict.values() if v.get('hv_traj')]
                if not x_series or not y_series:
                    continue
                x_mean = _pad_mean(x_series)
                y_mean = _pad_mean(y_series)
                seen_handlers.add(handler)
                seen_methods.add(method)
                ax.plot(x_mean, y_mean, color=HANDLER_COLOURS.get(handler, '#333333'),
                       ls=METHOD_LINESTYLE.get(method, '-'), lw=1.2)
            ax.set_title(sid, fontsize=9)
            ax.set_xlabel('evaluations (mean over seeds)', fontsize=8)
            ax.set_ylabel('feasible HV', fontsize=8)
            ax.tick_params(labelsize=8)
        leg_ax = axes[len(sids)]
        leg_ax.axis('off')
        h_handles = [Line2D([0], [0], color=HANDLER_COLOURS.get(h, '#333333'), lw=2, label=h)
                    for h in SC.HANDLERS if h in seen_handlers]
        m_handles = [Line2D([0], [0], color='black', ls=METHOD_LINESTYLE.get(m, '-'), lw=1.5, label=m)
                    for m in SC.METHODS if m in seen_methods]
        leg1 = leg_ax.legend(handles=h_handles, loc='upper left', fontsize=7, ncol=1,
                             title='handler (colour)', title_fontsize=8, frameon=False)
        leg_ax.add_artist(leg1)
        leg_ax.legend(handles=m_handles, loc='lower left', fontsize=7, ncol=1,
                     title='method (linestyle)', title_fontsize=8, frameon=False)
        for ax in axes[len(sids) + 1:]:
            ax.axis('off')
        fig.suptitle(f'Feasible HV vs. evaluations -- {mode} mode', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_png = os.path.join(plot_dir, f'traj_{mode}.png')
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        written.append(out_png)
    return written


def _draw_front_set(ax, fronts, colour, label, lw=1.6, band=True):
    fronts = [np.asarray(f, dtype=float) for f in fronts if f is not None and len(f) > 0]
    if not fronts:
        return
    if len(fronts) == 1:
        f = fronts[0][np.argsort(fronts[0][:, 0], kind='stable')]
        ax.step(f[:, 0], np.minimum.accumulate(f[:, 1]), where='post', color=colour, lw=lw, label=label)
        return
    n = len(fronts)
    k_med = math.ceil(n / 2)
    xs, surf = attainment_surfaces(fronts, ks=[1, k_med, n])
    to_nan = lambda a: np.where(np.isfinite(a), a, np.nan)
    ax.step(xs, to_nan(surf[k_med]), where='post', color=colour, lw=lw, label=f'{label} (median, n={n})')
    if band:
        ax.fill_between(xs, to_nan(surf[1]), to_nan(surf[n]), step='post', color=colour, alpha=0.15, linewidth=0)


def plot_attainment_grid(sid, mode, ref, fronts_by_mh, out_png, plt):
    """One figure per (sid, mode): subplot per handler, feasible/infeasible
    attainable cloud, true/sampled feasible front, per-method (nsga2, samos)
    attainment staircases, random as a grey reference."""
    handlers = [h for h in SC.HANDLERS if any(mh[1] == h and mh[0] != 'random' for mh in fronts_by_mh)]
    if not handlers:
        return False
    random_fronts = [f for (m, _h), fr in fronts_by_mh.items() if m == 'random' for f in fr]

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
            ax.step(tf[:, 0], np.minimum.accumulate(tf[:, 1]), where='post', color='black', lw=1.0)
            ax.plot(tf[:, 0], tf[:, 1], '.', color='black', ms=3.5)
        _draw_front_set(ax, random_fronts, '#999999', 'random', lw=1.0, band=False)
        for method in ('nsga2', 'samos'):
            _draw_front_set(ax, fronts_by_mh.get((method, handler), []), METHOD_COLOURS[method], method)
        ax.set_title(handler, fontsize=9)
        ax.tick_params(labelsize=7)

    from matplotlib.patches import Patch
    leg_ax = axes[len(handlers)]
    leg_ax.axis('off')
    handles = [Patch(color=CLOUD_FEASIBLE_COLOUR, label='feasible cloud'),
              Patch(color=CLOUD_INFEASIBLE_COLOUR, label='infeasible cloud')]
    if tf is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='black', marker='.', lw=1.0, label='true feasible front'))
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='#999999', lw=1.0, label='random (median attainment)'))
    handles += [Line2D([0], [0], color=METHOD_COLOURS[m], lw=1.6, label=f'{m} (median attainment)')
               for m in ('nsga2', 'samos')]
    leg_ax.legend(handles=handles, loc='center left', fontsize=8, frameon=False)
    for ax in axes[len(handlers) + 1:]:
        ax.axis('off')

    cloud_note = '' if ref['enumerable'] else f"   [cloud {ref['cloud_label']} -- approximation]"
    scenario = ref['scenario']
    fig.suptitle(f"{sid} / {mode}: {scenario['space']} {scenario['suite']}/pid{scenario['pid']} -- "
                f"{'/'.join(scenario['obj_metrics'])} objectives, "
                f"{scenario['constr_metric']} {'>=' if sense == -1 else '<='} {tau:.4f}{cloud_note}",
                fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def plot_final_bars(metrics_rows, plot_dir, plt):
    """Per sid: hard/soft subplots, x=handler, grouped bars=method (mean +-
    std over seeds), one figure for feasible HV and one for best feasible
    error."""
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
                                     squeeze=False)
            axes = axes[0]
            for ax, mode in zip(axes, modes_present):
                rows_m = [r for r in rows_sid if r['mode'] == mode]
                handlers = sorted({r['handler'] for r in rows_m}, key=_handler_sort_key)
                methods = [m for m in ('nsga2', 'samos') if any(r['method'] == m for r in rows_m)]
                width = 0.8 / max(len(methods), 1)
                x = np.arange(len(handlers))
                for mi, method in enumerate(methods):
                    means, stds = [], []
                    for h in handlers:
                        vals = [r[value_col] for r in rows_m if r['method'] == method and r['handler'] == h
                               and np.isfinite(r[value_col])]
                        means.append(np.mean(vals) if vals else np.nan)
                        stds.append(np.std(vals) if vals else np.nan)
                    ax.bar(x + mi * width, means, width, yerr=stds, label=method,
                          color=METHOD_COLOURS.get(method, '#333333'), capsize=2)
                random_vals = [r[value_col] for r in rows_m if r['method'] == 'random'
                              and np.isfinite(r[value_col])]
                if random_vals:
                    ax.axhline(np.mean(random_vals), color=METHOD_COLOURS['random'], ls=':', lw=1.2,
                              label='random')
                ax.set_xticks(x + width * (len(methods) - 1) / 2)
                ax.set_xticklabels(handlers, rotation=35, ha='right', fontsize=7)
                ax.set_title(mode, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.legend(fontsize=7)
            fig.suptitle(f'{sid}: {ylabel} (mean $\\pm$ std over seeds)', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            out_png = os.path.join(plot_dir, f'{sid}_{stem}_bars.png')
            fig.savefig(out_png, dpi=130)
            plt.close(fig)
            written.append(out_png)
    return written


# ─── LaTeX tables ───────────────────────────────────────────────────────────

def _tex_escape(s):
    return str(s).replace('_', r'\_')


def write_handler_winloss_table(winloss_rows, tables_dir, metric='hv_run'):
    """One .tex per method (nsga2, samos): rows = (scenario, mode), columns
    = handlers, cell = aggregated W-L-T count over that handler's pairwise
    tests within the (sid, mode, method) group."""
    written = []
    os.makedirs(tables_dir, exist_ok=True)
    for method in ('nsga2', 'samos'):
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
            r'(scenario, mode). Most-wins handler per row in bold.}',
            rf'\label{{tab:scenario-handler-winloss-{method}}}',
            r'\resizebox{\textwidth}{!}{%',
            rf"\begin{{tabular}}{{ll{'c' * len(handlers)}}}", r'\toprule',
            'scenario & mode & ' + ' & '.join(_tex_escape(h) for h in handlers) + r' \\',
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
            row = [sid, mode]
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
        r'Holm-significant.}',
        r'\label{tab:scenario-method-effect}', r'\resizebox{\textwidth}{!}{%',
        rf"\begin{{tabular}}{{ll{'c' * len(handlers)}}}", r'\toprule',
        'scenario & mode & ' + ' & '.join(_tex_escape(h) for h in handlers) + r' \\', r'\midrule']
    for sid, mode in sid_modes:
        d = {r['handler']: r for r in rows if r['sid'] == sid and r['mode'] == mode}
        row = [sid, mode] + [d[h]['better'] if h in d else '--' for h in handlers]
        lines.append(' & '.join(row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}', r'\end{table*}', ''])
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out_path


# ─── main ───────────────────────────────────────────────────────────────────

def plots_only(args):
    """Regenerate plots + LaTeX tables from plot_data.pkl / the CSVs, no pkl
    re-analysis."""
    cache_path = os.path.join(args.output_dir, 'plot_data.pkl')
    if not os.path.exists(cache_path):
        print(f'[analyse] --plots_only: missing {cache_path} (run the full analysis first).')
        return 1
    with open(cache_path, 'rb') as fh:
        cache = pickle.load(fh)
    render_outputs(args.output_dir, cache['refs'], cache['fronts'], cache['trajectories'],
                   cache['metrics_rows'], cache['winloss_rows'], cache['method_effect_rows'])
    return 0


def render_outputs(output_dir, refs, fronts, trajectories, metrics_rows, winloss_rows, method_effect_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    written = plot_trajectory_grids(trajectories, plot_dir, plt)
    for (sid, mode), fronts_by_mh in sorted(fronts.items()):
        ref = refs.get(sid)
        if ref is None:
            continue
        out_png = os.path.join(plot_dir, f'{sid}_{mode}_attainment.png')
        if plot_attainment_grid(sid, mode, ref, fronts_by_mh, out_png, plt):
            written.append(out_png)
    written.extend(plot_final_bars(metrics_rows, plot_dir, plt))
    print(f'[analyse] wrote {len(written)} plot(s) -> {plot_dir}')

    written_tex = write_handler_winloss_table(winloss_rows, tables_dir)
    written_tex.append(write_method_effect_table(method_effect_rows, tables_dir))
    print(f'[analyse] wrote {len(written_tex)} LaTeX table(s) -> {tables_dir}')


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
        ref = _build_ref(sid, args.enum_limit)
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
                        hv_run=hv_run, hv_true=ref['hv_true'], hv_ratio=hv_ratio,
                        best_err=m['m8_best_err'], best_err_slack=m['m8_best_err_slack'],
                        best_err_true=ref['best_err_true'],
                        n_feasible_final=n_feasible_final, n_total_final=n_total_final,
                        m3_final_infeasible_waste=m['m3_final_cumulative'], exact_waste=m['exact_waste'],
                        m4_final_front_size=m['m4_final'],
                        m6_min_slack=m['m6_min_slack'], m6_median_slack=m['m6_median_slack'],
                        soft_hv=m['soft_hv'], enumerable=ref['enumerable'],
                        h1_rejected_count=meta.get('h1_rejected_count', float('nan')),
                        hard_at_10=suite_row.get('hard_at_10', float('nan')),
                        soft_at_10=suite_row.get('soft_at_10', float('nan')),
                        rho=suite_row.get('rho', float('nan')),
                    )
                    metrics_rows.append(row)
                    stat_records.append(dict(sid=sid, mode=mode, method=method, handler=handler, seed=seed,
                                            hv_run=hv_run, best_err=m['m8_best_err']))
                    trajectories[mode].setdefault(sid, {}).setdefault((method, handler), {})[seed] = dict(
                        hv_traj=m['hv_traj'], n_total_traj=m['n_total_traj'])
                    fronts.setdefault((sid, mode), {}).setdefault((method, handler), []).append(m['final_front'])

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
    hard_soft_rows = compute_hard_vs_soft(stat_records)

    for name, rows in (('handler_winloss.csv', winloss_rows),
                       ('method_effect.csv', method_effect_rows),
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
    plot_cache = dict(refs=refs, fronts=fronts, trajectories=trajectories,
                      metrics_rows=metrics_rows, winloss_rows=winloss_rows,
                      method_effect_rows=method_effect_rows)
    cache_path = os.path.join(args.output_dir, 'plot_data.pkl')
    with open(cache_path, 'wb') as fh:
        pickle.dump(plot_cache, fh)
    print(f'[analyse] wrote plot cache -> {cache_path}')

    if not args.no_plots:
        render_outputs(args.output_dir, refs, fronts, trajectories, metrics_rows,
                       winloss_rows, method_effect_rows)

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
        print(f'\nScenario {sid}{suite_note}')
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
    p.add_argument('--handlers', nargs='+', default=list(SC.HANDLERS), choices=SC.HANDLERS)
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='Seed subset (default: discover every seed_*.pkl present per cell).')
    p.add_argument('--enum_limit', type=int, default=200_000,
                   help='Max search-space cardinality for exhaustive true-front enumeration.')
    p.add_argument('--plots_only', action='store_true',
                   help='Regenerate plots + LaTeX tables from plot_data.pkl without re-analysing pkls.')
    p.add_argument('--no_plots', action='store_true',
                   help='Skip plot/table rendering (CSV + JSON + stdout only).')
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_root, 'analysis')
    if args.plots_only:
        sys.exit(plots_only(args))
    sys.exit(main(args))
