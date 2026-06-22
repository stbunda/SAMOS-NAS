"""analyse_obj_ga.py -- Analysis driver for the obj-GA sweep (Experiment 1).

Consumes the per-seed pickles written by ``experiment_obj_ga.py`` and produces:

1. A shared EvoXBench Pareto approximation per (suite, pid), cached to
   ``results/obj_ga/evoxbench/{suite}/pid{pid}/ga_obj/pareto_approx.pkl``.
2. Recomputed EvoXBench HV / IGD+ indicator trajectories (against that shared
   approximation), filling the empty indicator dicts left by the experiment.
3. WFG indicators loaded directly from the pickles (live-computed at run time).
4. LaTeX tables (one file per metric, one table per benchmark x n_obj group).
5. Convergence line plots (full trajectory, no subsampling) per problem group.
6. 50 % attainment-surface plots for the 2-obj problems only (sub-exp 1.1).

Directory contract (fixed)
--------------------------
WFG       : results/obj_ga/wfg/{n_obj}_obj/{problem}/{method}/ga_obj/seed_*.pkl
EvoXBench : results/obj_ga/evoxbench/{suite}/pid{pid}/{method}/ga_obj/seed_*.pkl
Pareto    : results/obj_ga/evoxbench/{suite}/pid{pid}/ga_obj/pareto_approx.pkl

Pickle keys: var_archive, obj_archive, test_obj_archive, indicators, config, time.

CLI
---
python analyse_obj_ga.py --n_obj all --benchmark all \
    --config config/experiment_obj_ga.yaml --output_dir results/obj_ga/analysis
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
EXPERIMENT = 'ga_obj'

# How the config groups problems by objective count.  The YAML keys are not
# rigidly specified, so we accept several spellings.
_NOBJ_GROUP_KEYS = ('1.1', '1.2', '1.3')

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


def _iter_experiments(cfg: dict):
    """Yield (n_obj, group_dict) for every sub-experiment in the config.

    The config ``experiments`` section may be a dict keyed by sub-exp id
    (``'1.1'``) or a list of group dicts.  Each group must expose ``n_obj`` and
    may carry ``wfg`` and/or ``evoxbench`` problem specs.
    """
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
    """Return the list of WFG problem names for a group (e.g. ['wfg1', ...])."""
    wfg = group.get('wfg')
    if wfg is None:
        return []
    if isinstance(wfg, dict):
        probs = wfg.get('problems', [])
    else:
        probs = wfg
    return list(probs)


def _evox_problems(group: dict) -> list:
    """Return a list of (suite, pid) tuples for a group's EvoXBench problems.

    Accepts several shapes:
      evoxbench: {c10mop: [1, 8], in1kmop: [1, 4, 7]}
      evoxbench: [{suite: c10mop, pid: 1}, ...]
      evoxbench: ['c10mop/1', 'in1kmop/4']
    """
    evox = group.get('evoxbench')
    if evox is None:
        return []
    out = []
    if isinstance(evox, dict):
        # may be {'problems': {...}} or directly {suite: [pids]}
        inner = evox.get('problems', evox)
        if isinstance(inner, dict):
            for suite, pids in inner.items():
                for pid in pids:
                    out.append((suite, int(pid)))
        else:
            inner_list = inner
            for item in inner_list:
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


# --------- path helpers (new obj-GA layout) ---------------------------------------------------------------------------------------------------------------------------

def _wfg_results_root(results_root: str, n_obj: int, problem: str) -> str:
    return os.path.join(results_root, 'wfg', f'{n_obj}_obj', problem)


def _evox_pid_root(results_root: str, suite: str, pid: int) -> str:
    return os.path.join(results_root, 'evoxbench', suite, f'pid{pid}')


# --------- EvoXBench Pareto approximation ---------------------------------------------------------------------------------------------------------------------------------

def _pareto_approx_path(pid_root: str) -> str:
    return os.path.join(pid_root, EXPERIMENT, 'pareto_approx.pkl')


def get_or_build_pareto_approx(
    suite: str, pid: int, methods: list, n_gen: int,
    results_root: str, force: bool,
) -> dict | None:
    """Load (or build + save) the shared Pareto approximation for one PID.

    Returns the dict returned by ``build_pareto_approximation`` (extended with
    the on-disk contract keys), or ``None`` if no data exists.  The saved
    pickle uses the 4 contract keys ``pareto_front``, ``ref_point``,
    ``methods``, ``seeds`` and also retains the full builder dict under
    ``_approx`` so trajectory recomputation can reuse ``pareto_approx`` etc.
    """
    pid_root = _evox_pid_root(results_root, suite, pid)
    approx_path = _pareto_approx_path(pid_root)

    if os.path.isfile(approx_path) and not force:
        try:
            with open(approx_path, 'rb') as fh:
                saved = pickle.load(fh)
            print(f'  [analyse] Loaded cached pareto_approx: {approx_path}')
            # Reconstruct the builder-style dict expected downstream.
            return {
                'pareto_approx': np.asarray(saved['pareto_front']),
                'ref_point': np.asarray(saved['ref_point']),
                'methods': saved.get('methods', methods),
                'seeds': saved.get('seeds', []),
            }
        except Exception as exc:
            print(f'  [analyse] WARN: failed to read {approx_path}: {exc}; rebuilding.')

    # Build pooling all methods + seeds.  Pass results_root explicitly at the
    # pid level and let convergence insert the EXPERIMENT sub-dir.
    approx = convergence.build_pareto_approximation(
        suite, pid, methods,
        pop_size=1,           # irrelevant: results_root supplied explicitly
        n_gen=n_gen,
        force_rebuild=True,    # we manage our own contract pickle below
        results_root=pid_root,
        experiment_subdir=EXPERIMENT,
    )
    if approx is None:
        print(f'  [analyse] No data for {suite}/pid{pid}; skipping Pareto approx.')
        return None

    seeds = _collect_seed_ids(pid_root, methods)

    contract = {
        'pareto_front': approx['pareto_approx'],
        'ref_point': approx['ref_point'],
        'methods': approx.get('methods', sorted(methods)),
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


def _collect_seed_ids(pid_root: str, methods: list) -> list:
    """Return the sorted union of seed ids found across all methods."""
    seeds = set()
    for method in methods:
        seed_dir = os.path.join(pid_root, method, EXPERIMENT)
        if not os.path.isdir(seed_dir):
            continue
        for fname in os.listdir(seed_dir):
            if fname.startswith('seed_') and fname.endswith('.pkl'):
                try:
                    seeds.add(int(fname[len('seed_'):-len('.pkl')]))
                except ValueError:
                    pass
    return sorted(seeds)


# --------- indicator-trajectory loaders ---------------------------------------------------------------------------------------------------------------------------------------

def _wfg_trajectories(problem_root: str, methods: list, n_gen: int) -> dict:
    """Load WFG HV / IGD+ trajectories directly from pickle ``indicators``.

    Returns ``{method: (hv_mean, hv_std, igd_mean, igd_std)}``.
    """
    out = {}
    for method in methods:
        traj = plotter.load_indicator_trajectories(
            os.path.join(method, EXPERIMENT), n_gen, problem_root,
        )
        if traj is not None:
            out[method] = traj
    return out


def _evox_trajectories(pid_root: str, methods: list, n_gen: int,
                       approx: dict) -> dict:
    """Recompute EvoXBench trajectories against the shared Pareto approx."""
    out = {}
    for method in methods:
        traj = convergence.recompute_indicator_trajectories(
            method, n_gen, pid_root,
            approx['ref_point'], approx['pareto_approx'],
            experiment_subdir=EXPERIMENT,
        )
        if traj is not None:
            out[method] = traj
    return out


def _trajectories_to_table_dict(trajectories: dict) -> dict:
    """Convert ``{method: (hv_m, hv_s, igd_m, igd_s)}`` into the table data
    shape ``{method: {'hv': hv_mean, 'igd_plus': igd_mean}}``.

    Only the mean trajectory is available here, so Wilcoxon markers are not
    produced for these cells (the table function degrades gracefully).
    """
    out = {}
    for method, traj in trajectories.items():
        if traj is None:
            continue
        hv_mean, _hv_std, igd_mean, _igd_std = traj
        out[method] = {'hv': np.asarray(hv_mean), 'igd_plus': np.asarray(igd_mean)}
    return out


# --------- per-seed checkpoint matrices (for Wilcoxon significance) ------------------------------------------------------

def _wfg_seed_matrix(problem_root: str, method: str, n_gen: int):
    """Return per-seed ``(n_seeds, n_gen)`` matrices ``(hv, igd_plus)`` for WFG.

    Reads each seed pickle's live-computed ``indicators`` list.  Returns
    ``(None, None)`` when no data is found.
    """
    seed_dir = os.path.join(problem_root, method, EXPERIMENT)
    return _seed_matrix_from_indicators(seed_dir, n_gen)


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


def _evox_seed_matrix(pid_root: str, method: str, n_gen: int, approx: dict):
    """Return per-seed ``(n_seeds, n_gen)`` matrices ``(hv, igd_plus)`` for
    EvoXBench, recomputed against the shared Pareto approximation."""
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd_plus import IGDPlus

    seed_dir = os.path.join(pid_root, method, EXPERIMENT)
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
    """Return ``{method: list[(n_i, 2) ndarray]}`` from the final-gen
    ``test_obj_archive`` of every seed (first two objectives only)."""
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
    benchmark: str, output_dir: str, force: bool,
) -> list:
    """Run all analysis steps for one (n_obj, benchmark) group.

    Returns a list of ``(section_header, problem_keys, table_data,
    problem_labels)`` tuples for the combined n_obj table.
    """
    results_root = cfg.get('results_root', 'results/obj_ga')
    n_gen = int(cfg.get('n_gen', 100))
    checkpoints = list(cfg.get('metric_checkpoints', [10, 20, 50, 100]))
    baseline = 'nsga2' if 'nsga2' in methods else methods[0]

    plot_dir = os.path.join(output_dir, 'plots')
    hv_plot_dir = os.path.join(plot_dir, 'hv_convergence')
    igd_plot_dir = os.path.join(plot_dir, 'igd_convergence')
    attainment_plot_dir = os.path.join(plot_dir, 'attainment')
    table_dir = os.path.join(output_dir, 'tables')

    os.makedirs(hv_plot_dir, exist_ok=True)
    os.makedirs(igd_plot_dir, exist_ok=True)
    os.makedirs(attainment_plot_dir, exist_ok=True)

    sections = []   # collected for combined n_obj table

    # ------ WFG ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if benchmark in ('wfg', 'all'):
        wfg_problems = _wfg_problems(group)
        if wfg_problems:
            wfg_cfg = group.get('wfg', {})
            n_var = wfg_cfg.get('n_var', '?') if isinstance(wfg_cfg, dict) else '?'
            print(f'\n[WFG] n_obj={n_obj}: {wfg_problems}')
            table_data = {}
            problem_labels = {}
            for prob in wfg_problems:
                proot = _wfg_results_root(results_root, n_obj, prob)
                problem_labels[prob] = prob.upper()
                traj = _wfg_trajectories(proot, methods, n_gen)
                if not traj:
                    print(f'  [WFG] no data for {prob}; skipping.')
                    continue
                # Build per-seed matrices for significance markers.
                pdata = {}
                for method in methods:
                    hv_mat, igd_mat = _wfg_seed_matrix(proot, method, n_gen)
                    if hv_mat is not None:
                        pdata[method] = {'hv': hv_mat, 'igd_plus': igd_mat}
                table_data[prob] = pdata if pdata else _trajectories_to_table_dict(traj)

                # Convergence plots (full trajectory).
                out_hv = os.path.join(hv_plot_dir, f'wfg_{n_obj}obj_{prob}_hv.png')
                plotter.plot_convergence_obj_ga(
                    traj, methods, method_labels, prob.upper(), 'hv', out_hv,
                    colours=_METHOD_COLOURS)

                out_igd = os.path.join(igd_plot_dir, f'wfg_{n_obj}obj_{prob}_igd_plus.png')
                plotter.plot_convergence_obj_ga(
                    traj, methods, method_labels, prob.upper(), 'igd_plus', out_igd,
                    colours=_METHOD_COLOURS)

                # Attainment plots (2-obj only).
                if n_obj == 2:
                    archives = _final_archives_per_method(proot, methods)
                    if archives:
                        out_png = os.path.join(
                            attainment_plot_dir, f'wfg_{prob}_attainment.png')
                        plotter.plot_attainment_surface(
                            archives, method_labels, prob.upper(), out_png,
                            colours=_METHOD_COLOURS)

            if table_data:
                ordered_probs = [p for p in wfg_problems if p in table_data]
                for metric in ('hv', 'igd_plus'):
                    out_tex = os.path.join(
                        table_dir, f'wfg_{n_obj}obj_{metric}_table.tex')
                    generate_obj_ga_table(
                        table_data, methods, method_labels,
                        ordered_probs, problem_labels,
                        metric, checkpoints, out_tex, baseline_method=baseline)
                sections.append((
                    f'WFG ({n_obj} objectives, {n_var} variables)',
                    ordered_probs, table_data, problem_labels,
                ))

    # ------ EvoXBench ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if benchmark in ('evoxbench', 'all'):
        evox_problems = _evox_problems(group)
        if evox_problems:
            print(f'\n[EvoXBench] n_obj={n_obj}: {evox_problems}')
            table_data = {}
            problem_labels = {}
            ordered_keys = []
            for suite, pid in evox_problems:
                key = f'{suite}_pid{pid}'
                ordered_keys.append(key)
                meta = BENCHMARK_META.get(suite, {}).get(pid, {})
                problem_labels[key] = meta.get('label', key)
                pid_root = _evox_pid_root(results_root, suite, pid)

                approx = get_or_build_pareto_approx(
                    suite, pid, methods, n_gen, results_root, force)
                if approx is None:
                    continue

                traj = _evox_trajectories(pid_root, methods, n_gen, approx)
                if not traj:
                    print(f'  [EvoXBench] no trajectories for {key}; skipping.')
                    continue

                # Per-seed matrices for significance markers.
                pdata = {}
                for method in methods:
                    hv_mat, igd_mat = _evox_seed_matrix(pid_root, method, n_gen, approx)
                    if hv_mat is not None:
                        pdata[method] = {'hv': hv_mat, 'igd_plus': igd_mat}
                table_data[key] = pdata if pdata else _trajectories_to_table_dict(traj)

                # Convergence plots (full trajectory).
                out_hv = os.path.join(hv_plot_dir, f'evox_{n_obj}obj_{key}_hv.png')
                plotter.plot_convergence_obj_ga(
                    traj, methods, method_labels,
                    problem_labels[key], 'hv', out_hv,
                    colours=_METHOD_COLOURS)

                out_igd = os.path.join(igd_plot_dir, f'evox_{n_obj}obj_{key}_igd_plus.png')
                plotter.plot_convergence_obj_ga(
                    traj, methods, method_labels,
                    problem_labels[key], 'igd_plus', out_igd,
                    colours=_METHOD_COLOURS)

                # Attainment plots (2-obj only).
                if n_obj == 2:
                    archives = _final_archives_per_method(pid_root, methods)
                    if archives:
                        out_png = os.path.join(
                            attainment_plot_dir, f'evox_{key}_attainment.png')
                        plotter.plot_attainment_surface(
                            archives, method_labels, problem_labels[key], out_png,
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
                sections.append((
                    f'EvoXBench ({n_obj} objectives)',
                    ordered_keys, table_data, problem_labels,
                ))

    return sections


# --------- combined n_obj convergence plots ------------------------------------------------------------------------------------------------------------------------------

def plot_combined_convergence(
    cfg: dict, methods: list, method_labels: dict,
    benchmark: str, output_dir: str
):
    """Create combined convergence plots with subplots for each n_obj value.

    Produces one plot per metric (HV, IGD+) showing all n_obj benchmarks as subplots.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_root = cfg.get('results_root', 'results/obj_ga')
    n_gen = int(cfg.get('n_gen', 100))

    plot_dir = os.path.join(output_dir, 'plots')
    hv_plot_dir = os.path.join(plot_dir, 'hv_convergence')
    igd_plot_dir = os.path.join(plot_dir, 'igd_convergence')

    # Collect all n_obj/problem combinations
    all_data = {}  # {n_obj: {problem_key: {metric: traj_dict}}}

    for n_obj, group in _iter_experiments(cfg):
        all_data[n_obj] = {}

        # WFG
        if benchmark in ('wfg', 'all'):
            wfg_problems = _wfg_problems(group)
            for prob in wfg_problems:
                proot = _wfg_results_root(results_root, n_obj, prob)
                traj = _wfg_trajectories(proot, methods, n_gen)
                if traj:
                    all_data[n_obj][f'WFG {prob.upper()}'] = traj

        # EvoXBench
        if benchmark in ('evoxbench', 'all'):
            evox_problems = _evox_problems(group)
            for suite, pid in evox_problems:
                key = f'EvoXBench {suite.upper()} pid{pid}'
                pid_root = _evox_pid_root(results_root, suite, pid)
                approx = get_or_build_pareto_approx(
                    suite, pid, methods, n_gen, results_root, False)
                if approx:
                    traj = _evox_trajectories(pid_root, methods, n_gen, approx)
                    if traj:
                        all_data[n_obj][key] = traj

    if not all_data:
        return

    # Create combined plots per metric
    for metric in ('hv', 'igd_plus'):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f'{metric.upper()} Convergence across all Objectives',
            fontsize=14, fontweight='bold')

        for ax_idx, n_obj in sorted(all_data.keys()):
            ax = axes[ax_idx - 2] if n_obj < 3 else axes[n_obj - 2]
            ax.set_title(f'n_obj = {n_obj}')
            ax.set_xlabel('Generation')
            ax.set_ylabel(metric.upper())

            # Plot all problems on this subplot
            for prob_key, traj in all_data[n_obj].items():
                if metric not in traj:
                    continue
                mean_vals = traj[metric]['mean']
                std_vals = traj[metric]['std']
                gens = np.arange(len(mean_vals))

                # Get method color from plotter's default colors
                color = plotter._default_colors.get(methods[0], 'C0')  # fallback
                ax.plot(gens, mean_vals, label=prob_key, alpha=0.7)
                ax.fill_between(
                    gens, mean_vals - std_vals, mean_vals + std_vals,
                    alpha=0.2)

            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        out_metric = metric.replace('_', '_plus') if 'plus' in metric else metric
        out_hv = os.path.join(
            hv_plot_dir if metric == 'hv' else igd_plot_dir,
            f'{benchmark}_all_objectives_{metric}_combined.png')
        os.makedirs(os.path.dirname(out_hv), exist_ok=True)
        plt.savefig(out_hv, dpi=150, bbox_inches='tight')
        print(f'[plot] Saved {out_hv}')
        plt.close(fig)


# --------- main ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyse obj-GA sweep results.')
    parser.add_argument('--n_obj', default='all',
                        help='Objective count to analyse: an int or "all".')
    parser.add_argument('--benchmark', default='all',
                        choices=['wfg', 'evoxbench', 'all'])
    parser.add_argument('--config',
                        default=os.path.join(os.path.dirname(__file__),
                                             'config', 'experiment_obj_ga.yaml'))
    parser.add_argument('--output_dir', default='results/obj_ga/analysis')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild the EvoXBench Pareto approximation pickles.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    methods, method_labels = _methods_and_labels(cfg)

    want_n_obj = None if str(args.n_obj).lower() == 'all' else int(args.n_obj)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect experiments, grouping 4+ objectives together
    experiments = {}
    for n_obj, group in _iter_experiments(cfg):
        if n_obj <= 3:
            experiments[n_obj] = group
        else:
            # Merge all 4+ objective groups under key '4+'
            if '4+' not in experiments:
                experiments['4+'] = {
                    'n_obj': 4,  # Use 4 as representative n_obj for labeling
                    'wfg': {'n_var': None, 'problems': []},
                    'evoxbench': {}
                }
            # Merge problems from this group
            if 'wfg' in group:
                if experiments['4+']['wfg']['n_var'] is None:
                    experiments['4+']['wfg']['n_var'] = group['wfg'].get('n_var')
                experiments['4+']['wfg']['problems'].extend(group['wfg'].get('problems', []))
            if 'evoxbench' in group:
                for suite, pids in group['evoxbench'].items():
                    if suite not in experiments['4+']['evoxbench']:
                        experiments['4+']['evoxbench'][suite] = []
                    experiments['4+']['evoxbench'][suite].extend(pids)

    # Analyze each group; collect sections for the combined n_obj table.
    all_sections = []
    for key, group in sorted(experiments.items(), key=lambda x: (x[0] != '4+', x[0])):
        if isinstance(key, int):
            n_obj = key
            display_str = f'n_obj = {n_obj}'
        else:
            n_obj = group['n_obj']
            display_str = 'n_obj = 4+ (combined)'

        if want_n_obj is not None and n_obj != want_n_obj:
            continue

        print(f'\n{"=" * 70}\n {display_str}\n{"=" * 70}')
        sections = analyse_group(
            n_obj, group, cfg, methods, method_labels,
            args.benchmark, args.output_dir, args.force)
        all_sections.extend(sections)

    # Combined n_obj comparison table (gen=100 only, all benchmarks in one table).
    n_gen = int(cfg.get('n_gen', 100))
    if all_sections and want_n_obj is None:
        table_dir = os.path.join(args.output_dir, 'tables')
        baseline = 'nsga2' if 'nsga2' in methods else methods[0]
        print(f'\n{"=" * 70}\n Combined n_obj comparison table\n{"=" * 70}')
        for metric in ('hv', 'igd_plus'):
            out_tex = os.path.join(table_dir, f'combined_nobj_{metric}_table.tex')
            generate_obj_ga_nobj_comparison_table(
                all_sections, methods, method_labels,
                metric, n_gen, out_tex, baseline_method=baseline)

    print('\n[analyse] Done.')


if __name__ == '__main__':
    main()
