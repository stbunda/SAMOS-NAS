"""analyse_obj_ga_pred.py -- Analysis driver for the predictor-guided GA sweep (Experiment 2).

Consumes the per-seed pickles written by ``experiment_obj_ga_pred.py`` and, for
each predictor in the config, produces the same artefacts as Experiment 1 --- but
scored on the TRUE objectives of each GA's final (predicted-space) non-dominated
front. Outputs are namespaced per predictor.

Directory contract (fixed) --- note the extra ``{predictor}`` level vs Experiment 1:
WFG       : results/obj_ga_pred/wfg/{n_obj}_obj/{problem}/{predictor}/{method}/ga_obj_pred/seed_*.pkl
EvoXBench : results/obj_ga_pred/evoxbench/{suite}/pid{pid}/{predictor}/{method}/ga_obj_pred/seed_*.pkl
Pareto    : results/obj_ga_pred/evoxbench/{suite}/pid{pid}/ga_obj_pred/pareto_approx.pkl
            (pooled across ALL predictors + methods + seeds -> a common reference)

Pickle keys: var_archive, obj_archive (PREDICTED), test_obj_archive (TRUE),
indicators, config, time.

For WFG the ``indicators`` list already holds HV / IGD+ computed live on the TRUE
objectives during the run. For EvoXBench the indicators are recomputed here against
the pooled Pareto approximation (the ``indicators`` dicts in the pickle are empty).

CLI
---
python experiments/obj_ga_pred/analyse_obj_ga_pred.py --n_obj all --benchmark all \
    --output_dir results/obj_ga_pred/analysis
"""

import argparse
import os
import pickle
import sys

# Allow running from any CWD: put the repo root (two levels up) on sys.path so the
# root packages (strategy/, problem/, analysis/) import regardless of where this
# script lives or is launched from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import yaml

from analysis import convergence
from analysis import plotter
from analysis.latex_table_generator import (
    generate_obj_ga_table,
    generate_obj_ga_nobj_comparison_table,
)
from problem.evoxbench.benchmark_meta import BENCHMARK_META

# Sub-directory inserted between the per-method folder and the seed pickles.
EXPERIMENT = 'ga_obj_pred'

# ─── method colour palette ────────────────────────────────────────────────────
_METHOD_COLOURS = {
    'random':           '#4e79a7',  # blue
    'nsga2':            '#f28e2b',  # orange
    'nsga3':            '#e15759',  # red
    'moead':            '#59a14f',  # green
    'sms-emoa':         '#a0cbe8',  # light blue
    'rvea':             '#b07aa1',  # purple
    'age-moea':         '#ff9da7',  # pink
    'age-moea2':        '#8cd17d',  # light green
}


# --------- config loading ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _methods_and_labels(cfg: dict):
    """Return (ordered method keys, {method: label}) from the config."""
    algos = cfg.get('algorithms', [])
    methods, labels = [], {}
    for a in algos:
        if isinstance(a, dict):
            name = a.get('name')
            labels[name] = a.get('label', name)
        else:
            name = a
            labels[name] = name
        methods.append(name)
    return methods, labels


def _predictors(cfg: dict):
    """Return (ordered predictor keys, {predictor: label}) from the config."""
    preds, labels = [], {}
    for p in cfg.get('predictors', []):
        name = p['name']
        preds.append(name)
        labels[name] = p.get('label', name)
    return preds, labels


def _iter_experiments(cfg: dict):
    """Yield (n_obj, group_dict) for every sub-experiment in the config."""
    experiments = cfg.get('experiments', {})
    groups = experiments.values() if isinstance(experiments, dict) else experiments
    for group in groups:
        if not isinstance(group, dict):
            continue
        n_obj = group.get('n_obj')
        if n_obj is None:
            continue
        yield int(n_obj), group


def _wfg_problems(group: dict) -> list:
    wfg = group.get('wfg')
    if wfg is None:
        return []
    if isinstance(wfg, dict):
        return list(wfg.get('problems', []))
    return list(wfg)


def _evox_problems(group: dict) -> list:
    """Return a list of (suite, pid) tuples for a group's EvoXBench problems."""
    evox = group.get('evoxbench')
    if evox is None:
        return []
    out = []
    if isinstance(evox, dict):
        inner = evox.get('problems', evox)
        if isinstance(inner, dict):
            for suite, pids in inner.items():
                for pid in pids:
                    out.append((suite, int(pid)))
        else:
            for item in inner:
                out.extend(_parse_evox_item(item))
    else:
        for item in evox:
            out.extend(_parse_evox_item(item))
    return out


def _parse_evox_item(item):
    if isinstance(item, dict):
        return [(item['suite'], int(item['pid']))]
    if isinstance(item, str) and '/' in item:
        suite, pid = item.split('/')
        return [(suite, int(pid))]
    return []


# --------- path helpers (predictor-aware obj-GA-pred layout) ---------------------------------------------------------------------------

def _wfg_pred_root(results_root, n_obj, problem, predictor) -> str:
    return os.path.join(results_root, 'wfg', f'{n_obj}_obj', problem, predictor)


def _evox_pid_root(results_root, suite, pid) -> str:
    return os.path.join(results_root, 'evoxbench', suite, f'pid{pid}')


def _evox_pred_root(results_root, suite, pid, predictor) -> str:
    return os.path.join(_evox_pid_root(results_root, suite, pid), predictor)


# --------- EvoXBench Pareto approximation (pooled across predictors) ---------------------------------------------------

def _pareto_approx_path(pid_root: str) -> str:
    return os.path.join(pid_root, EXPERIMENT, 'pareto_approx.pkl')


def get_or_build_pareto_approx(
    suite: str, pid: int, methods: list, predictors: list, n_gen: int,
    results_root: str, force: bool,
) -> dict | None:
    """Load (or build + save) the shared Pareto approximation for one PID.

    The front is pooled across ALL (predictor, method) seed archives so every GA
    is scored on a single common reference. Combined ``"{predictor}/{method}"``
    tokens are passed as the ``methods`` argument to the shared convergence builder
    (it ``os.path.join``s them, so the extra predictor path level is handled).
    """
    pid_root = _evox_pid_root(results_root, suite, pid)
    approx_path = _pareto_approx_path(pid_root)
    combined = [f'{pred}/{m}' for pred in predictors for m in methods]

    if os.path.isfile(approx_path) and not force:
        try:
            with open(approx_path, 'rb') as fh:
                saved = pickle.load(fh)
            print(f'  [analyse] Loaded cached pareto_approx: {approx_path}')
            return {
                'pareto_approx': np.asarray(saved['pareto_front']),
                'ref_point': np.asarray(saved['ref_point']),
                'methods': saved.get('methods', combined),
                'seeds': saved.get('seeds', []),
            }
        except Exception as exc:
            print(f'  [analyse] WARN: failed to read {approx_path}: {exc}; rebuilding.')

    approx = convergence.build_pareto_approximation(
        suite, pid, combined,
        pop_size=1,             # irrelevant: results_root supplied explicitly
        n_gen=n_gen,
        force_rebuild=True,     # we manage our own contract pickle below
        results_root=pid_root,
        experiment_subdir=EXPERIMENT,
    )
    if approx is None:
        print(f'  [analyse] No data for {suite}/pid{pid}; skipping Pareto approx.')
        return None

    seeds = _collect_seed_ids(pid_root, combined)
    contract = {
        'pareto_front': approx['pareto_approx'],
        'ref_point': approx['ref_point'],
        'methods': approx.get('methods', sorted(combined)),
        'seeds': seeds,
    }
    os.makedirs(os.path.dirname(approx_path), exist_ok=True)
    with open(approx_path, 'wb') as fh:
        pickle.dump(contract, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  [analyse] Saved pareto_approx -> {approx_path}')

    return {
        'pareto_approx': approx['pareto_approx'],
        'ref_point': approx['ref_point'],
        'methods': contract['methods'],
        'seeds': seeds,
    }


def _collect_seed_ids(pid_root: str, method_tokens: list) -> list:
    """Return the sorted union of seed ids found across all method tokens."""
    seeds = set()
    for token in method_tokens:
        seed_dir = os.path.join(pid_root, token, EXPERIMENT)
        if not os.path.isdir(seed_dir):
            continue
        for fname in os.listdir(seed_dir):
            if fname.startswith('seed_') and fname.endswith('.pkl'):
                try:
                    seeds.add(int(fname[len('seed_'):-len('.pkl')]))
                except ValueError:
                    pass
    return sorted(seeds)


# --------- indicator-trajectory loaders (per predictor) ---------------------------------------------------------------------------------------

def _wfg_trajectories(wfg_pred_root: str, methods: list, n_gen: int) -> dict:
    """Load WFG HV / IGD+ trajectories (live, true-objective) from the pickles."""
    out = {}
    for method in methods:
        traj = plotter.load_indicator_trajectories(
            os.path.join(method, EXPERIMENT), n_gen, wfg_pred_root,
        )
        if traj is not None:
            out[method] = traj
    return out


def _evox_trajectories(evox_pred_root: str, methods: list, n_gen: int,
                       approx: dict) -> dict:
    """Recompute EvoXBench trajectories against the shared Pareto approx."""
    out = {}
    for method in methods:
        traj = convergence.recompute_indicator_trajectories(
            method, n_gen, evox_pred_root,
            approx['ref_point'], approx['pareto_approx'],
            experiment_subdir=EXPERIMENT,
        )
        if traj is not None:
            out[method] = traj
    return out


# --------- per-seed checkpoint matrices (for Wilcoxon significance) ------------------------------------------------------

def _seed_matrix_from_indicators(seed_dir: str, n_gen: int):
    if not os.path.isdir(seed_dir):
        return None, None
    hv_rows, igd_rows = [], []
    for fname in sorted(os.listdir(seed_dir)):
        if not fname.endswith('.pkl'):
            continue
        try:
            with open(os.path.join(seed_dir, fname), 'rb') as fh:
                data = pickle.load(fh)
        except Exception:
            continue
        inds = data.get('indicators', [])
        if not inds:
            continue
        hv = [d.get('hv', np.nan) for d in inds]
        igd = [d.get('igd_plus', np.nan) for d in inds]
        hv_rows.append(_fit_len(hv, n_gen))
        igd_rows.append(_fit_len(igd, n_gen))
    if not hv_rows:
        return None, None
    return np.asarray(hv_rows, float), np.asarray(igd_rows, float)


def _evox_seed_matrix(evox_pred_root: str, method: str, n_gen: int, approx: dict):
    """Per-seed ``(n_seeds, n_gen)`` (hv, igd_plus) for one predictor's method,
    recomputed against the shared Pareto approximation."""
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd_plus import IGDPlus

    seed_dir = os.path.join(evox_pred_root, method, EXPERIMENT)
    if not os.path.isdir(seed_dir):
        return None, None

    hv_ind = HV(ref_point=approx['ref_point'])
    igd_ind = IGDPlus(approx['pareto_approx'])

    hv_rows, igd_rows = [], []
    for fname in sorted(os.listdir(seed_dir)):
        if not fname.endswith('.pkl'):
            continue
        try:
            with open(os.path.join(seed_dir, fname), 'rb') as fh:
                data = pickle.load(fh)
        except Exception:
            continue
        archive = data.get('test_obj_archive', [])
        if not archive:
            continue
        hv_series, igd_series = [], []
        for F in archive:
            if F is None or len(F) == 0:
                hv_series.append(0.0)
                igd_series.append(np.nan)
                continue
            F = np.asarray(F, float)
            F = np.where(np.isfinite(F), F, 1.0)
            hv_series.append(float(hv_ind(F)))
            igd_series.append(float(igd_ind(F)))
        hv_rows.append(_fit_len(hv_series, n_gen))
        igd_rows.append(_fit_len(igd_series, n_gen))
    if not hv_rows:
        return None, None
    return np.asarray(hv_rows, float), np.asarray(igd_rows, float)


def _fit_len(series: list, n_gen: int) -> list:
    """Down-sample / pad a series to length n_gen (same logic as plotter)."""
    if len(series) == n_gen:
        return list(series)
    if len(series) > n_gen:
        step = len(series) // n_gen
        return [series[min((g + 1) * step - 1, len(series) - 1)] for g in range(n_gen)]
    out = list(series)
    while len(out) < n_gen:
        out.append(out[-1] if out else np.nan)
    return out


# --------- final-generation archives for attainment plots ---------------------------------------------------------------------------------

def _final_archives_per_method(root: str, methods: list) -> dict:
    """``{method: list[(n_i, 2) ndarray]}`` from each seed's final-gen TRUE archive."""
    out = {}
    for method in methods:
        seed_dir = os.path.join(root, method, EXPERIMENT)
        if not os.path.isdir(seed_dir):
            continue
        fronts = []
        for fname in sorted(os.listdir(seed_dir)):
            if not fname.endswith('.pkl'):
                continue
            try:
                with open(os.path.join(seed_dir, fname), 'rb') as fh:
                    data = pickle.load(fh)
            except Exception:
                continue
            archive = data.get('test_obj_archive', [])
            if not archive:
                continue
            F = np.asarray(archive[-1], float)
            if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] < 2:
                continue
            finite = np.isfinite(F[:, :2]).all(axis=1)
            F = F[finite][:, :2]
            if len(F) > 0:
                fronts.append(F)
        if fronts:
            out[method] = fronts
    return out


# --------- per-group analysis ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def analyse_group(
    n_obj: int, group: dict, cfg: dict,
    methods: list, method_labels: dict,
    predictors: list, predictor_labels: dict,
    benchmark: str, output_dir: str, force: bool,
) -> dict:
    """Run all analysis steps for one (n_obj, benchmark) group, per predictor.

    Returns ``{predictor: [(section_header, problem_keys, table_data,
    problem_labels)]}`` for the combined n_obj tables.
    """
    results_root = cfg.get('results_root', 'results/obj_ga_pred')
    n_gen = int(cfg.get('n_gen', 100))
    checkpoints = list(cfg.get('metric_checkpoints', [10, 20, 50, 100]))
    baseline = 'nsga2' if 'nsga2' in methods else methods[0]

    wfg_problems = _wfg_problems(group)
    evox_problems = _evox_problems(group)

    # Pareto approximation per PID is shared across predictors --- build it once.
    evox_approx = {}
    if benchmark in ('evoxbench', 'all') and evox_problems:
        for suite, pid in evox_problems:
            evox_approx[(suite, pid)] = get_or_build_pareto_approx(
                suite, pid, methods, predictors, n_gen, results_root, force)

    pred_sections = {pred: [] for pred in predictors}

    for predictor in predictors:
        pred_label = predictor_labels.get(predictor, predictor)
        plot_dir = os.path.join(output_dir, 'plots', predictor)
        hv_plot_dir = os.path.join(plot_dir, 'hv_convergence')
        igd_plot_dir = os.path.join(plot_dir, 'igd_convergence')
        attainment_plot_dir = os.path.join(plot_dir, 'attainment')
        table_dir = os.path.join(output_dir, 'tables', predictor)
        os.makedirs(hv_plot_dir, exist_ok=True)
        os.makedirs(igd_plot_dir, exist_ok=True)
        os.makedirs(attainment_plot_dir, exist_ok=True)

        # ------ WFG ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if benchmark in ('wfg', 'all') and wfg_problems:
            wfg_cfg = group.get('wfg', {})
            n_var = wfg_cfg.get('n_var', '?') if isinstance(wfg_cfg, dict) else '?'
            print(f'\n[WFG] predictor={predictor} n_obj={n_obj}: {wfg_problems}')
            table_data = {}
            problem_labels = {}
            for prob in wfg_problems:
                proot = _wfg_pred_root(results_root, n_obj, prob, predictor)
                problem_labels[prob] = prob.upper()
                traj = _wfg_trajectories(proot, methods, n_gen)
                if not traj:
                    print(f'  [WFG] no data for {prob}/{predictor}; skipping.')
                    continue
                pdata = {}
                for method in methods:
                    hv_mat, igd_mat = _seed_matrix_from_indicators(
                        os.path.join(proot, method, EXPERIMENT), n_gen)
                    if hv_mat is not None:
                        pdata[method] = {'hv': hv_mat, 'igd_plus': igd_mat}
                if pdata:
                    table_data[prob] = pdata

                out_hv = os.path.join(hv_plot_dir, f'wfg_{n_obj}obj_{prob}_hv.png')
                plotter.plot_convergence_obj_ga(
                    traj, methods, method_labels, f'{prob.upper()} ({pred_label})',
                    'hv', out_hv, colours=_METHOD_COLOURS)
                out_igd = os.path.join(igd_plot_dir, f'wfg_{n_obj}obj_{prob}_igd_plus.png')
                plotter.plot_convergence_obj_ga(
                    traj, methods, method_labels, f'{prob.upper()} ({pred_label})',
                    'igd_plus', out_igd, colours=_METHOD_COLOURS)

                if n_obj == 2:
                    archives = _final_archives_per_method(proot, methods)
                    if archives:
                        out_png = os.path.join(
                            attainment_plot_dir, f'wfg_{prob}_attainment.png')
                        plotter.plot_attainment_surface(
                            archives, method_labels, f'{prob.upper()} ({pred_label})',
                            out_png, colours=_METHOD_COLOURS)

            if table_data:
                ordered_probs = [p for p in wfg_problems if p in table_data]
                for metric in ('hv', 'igd_plus'):
                    out_tex = os.path.join(
                        table_dir, f'wfg_{n_obj}obj_{metric}_table.tex')
                    generate_obj_ga_table(
                        table_data, methods, method_labels,
                        ordered_probs, problem_labels,
                        metric, checkpoints, out_tex, baseline_method=baseline)
                pred_sections[predictor].append((
                    f'WFG ({n_obj} objectives, {n_var} variables)',
                    ordered_probs, table_data, problem_labels,
                ))

        # ------ EvoXBench ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if benchmark in ('evoxbench', 'all') and evox_problems:
            print(f'\n[EvoXBench] predictor={predictor} n_obj={n_obj}: {evox_problems}')
            table_data = {}
            problem_labels = {}
            ordered_keys = []
            for suite, pid in evox_problems:
                key = f'{suite}_pid{pid}'
                ordered_keys.append(key)
                meta = BENCHMARK_META.get(suite, {}).get(pid, {})
                problem_labels[key] = meta.get('label', key)

                approx = evox_approx.get((suite, pid))
                if approx is None:
                    continue
                epred_root = _evox_pred_root(results_root, suite, pid, predictor)

                traj = _evox_trajectories(epred_root, methods, n_gen, approx)
                if not traj:
                    print(f'  [EvoXBench] no trajectories for {key}/{predictor}; skipping.')
                    continue

                pdata = {}
                for method in methods:
                    hv_mat, igd_mat = _evox_seed_matrix(epred_root, method, n_gen, approx)
                    if hv_mat is not None:
                        pdata[method] = {'hv': hv_mat, 'igd_plus': igd_mat}
                if pdata:
                    table_data[key] = pdata

                out_hv = os.path.join(hv_plot_dir, f'evox_{n_obj}obj_{key}_hv.png')
                plotter.plot_convergence_obj_ga(
                    traj, methods, method_labels,
                    f"{meta.get('label', key)} ({pred_label})", 'hv', out_hv,
                    colours=_METHOD_COLOURS)
                out_igd = os.path.join(igd_plot_dir, f'evox_{n_obj}obj_{key}_igd_plus.png')
                plotter.plot_convergence_obj_ga(
                    traj, methods, method_labels,
                    f"{meta.get('label', key)} ({pred_label})", 'igd_plus', out_igd,
                    colours=_METHOD_COLOURS)

                if n_obj == 2:
                    archives = _final_archives_per_method(epred_root, methods)
                    if archives:
                        out_png = os.path.join(
                            attainment_plot_dir, f'evox_{key}_attainment.png')
                        plotter.plot_attainment_surface(
                            archives, method_labels,
                            f"{meta.get('label', key)} ({pred_label})", out_png,
                            colours=_METHOD_COLOURS)

            if table_data:
                ordered_keys = [k for k in ordered_keys if k in table_data]
                for metric in ('hv', 'igd_plus'):
                    out_tex = os.path.join(
                        table_dir, f'evox_{n_obj}obj_{metric}_table.tex')
                    generate_obj_ga_table(
                        table_data, methods, method_labels,
                        ordered_keys, problem_labels,
                        metric, checkpoints, out_tex, baseline_method=baseline)
                pred_sections[predictor].append((
                    f'EvoXBench ({n_obj} objectives)',
                    ordered_keys, table_data, problem_labels,
                ))

    return pred_sections


# --------- main ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyse obj-GA-pred sweep results.')
    parser.add_argument('--n_obj', default='all',
                        help='Objective count to analyse: an int or "all".')
    parser.add_argument('--benchmark', default='all',
                        choices=['wfg', 'evoxbench', 'all'])
    parser.add_argument('--predictor', default='all',
                        help='Predictor to analyse: a key from the config or "all".')
    parser.add_argument('--config',
                        default=os.path.join(os.path.dirname(__file__),
                                             'config', 'experiment_obj_ga_pred.yaml'))
    parser.add_argument('--output_dir', default='results/obj_ga_pred/analysis')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild the EvoXBench Pareto approximation pickles.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    methods, method_labels = _methods_and_labels(cfg)
    predictors, predictor_labels = _predictors(cfg)
    if str(args.predictor).lower() != 'all':
        if args.predictor not in predictors:
            raise SystemExit(
                f'Unknown predictor {args.predictor!r}; config has {predictors}.')
        predictors = [args.predictor]

    want_n_obj = None if str(args.n_obj).lower() == 'all' else int(args.n_obj)

    all_pred_sections = {pred: [] for pred in predictors}
    for n_obj, group in _iter_experiments(cfg):
        if want_n_obj is not None and n_obj != want_n_obj:
            continue
        pred_sections = analyse_group(
            n_obj, group, cfg, methods, method_labels,
            predictors, predictor_labels,
            args.benchmark, args.output_dir, args.force,
        )
        for pred, secs in pred_sections.items():
            all_pred_sections[pred].extend(secs)

    # Combined n_obj comparison table per predictor (gen=100, all benchmarks).
    n_gen = int(cfg.get('n_gen', 100))
    if want_n_obj is None:
        baseline = 'nsga2' if 'nsga2' in methods else methods[0]
        print(f'\n{"=" * 70}\n Combined n_obj comparison tables\n{"=" * 70}')
        for predictor in predictors:
            sections = all_pred_sections[predictor]
            if not sections:
                continue
            table_dir = os.path.join(args.output_dir, 'tables', predictor)
            for metric in ('hv', 'igd_plus'):
                out_tex = os.path.join(table_dir, f'combined_nobj_{metric}_table.tex')
                generate_obj_ga_nobj_comparison_table(
                    sections, methods, method_labels,
                    metric, n_gen, out_tex, baseline_method=baseline)

    print('\n[analyse] Done.')


if __name__ == '__main__':
    main()
