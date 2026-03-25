"""
Plotting utilities for NASBench-101 baseline results.

Loads per-seed result .pkl files from the RESULTS_ROOT directory and
produces HV / IGD+ trajectory plots (mean ± std over seeds).
"""

import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus


# ─── method styling ───────────────────────────────────────────────────────────

COLOURS = {
    'random':           '#4e79a7',
    'nsga2':            '#f28e2b',
    'nsga2-single':     '#59a14f',
    'samos-rfr':        '#e15759',
    'samos-xgb':        '#b07aa1',
    'mosmac':           '#76b7b2',
    # ── comparison methods ───────────────────────────────────────────────────
    'ssa-nsga2-default': '#ff9da7',
    'ssa-nsga2-rfr':     '#c0392b',
    'ssa-nsga2-xgb':     '#922b21',
    'gpsaf-default':    '#a0cbe8',
    'gpsaf-rfr':        '#2980b9',
    'gpsaf-xgb':        '#1a5276',
    'cobra':            '#59a14f',
    'parego':           '#9c755f',
}
LABELS = {
    'random':           'Random',
    'nsga2':            'NSGA-II (2-pt XO, unif. mut)',
    'nsga2-single':     'NSGA-II (no XO, single-pt mut)',
    'samos-rfr':        'SAMOS (RFR surrogate)',
    'samos-xgb':        'SAMOS (XGBoost surrogate)',
    'mosmac':           'MO-SMAC',
    # ── comparison methods ───────────────────────────────────────────────────
    'ssa-nsga2-default': 'SSA-NSGA-II (default)',
    'ssa-nsga2-rfr':     'SSA-NSGA-II (RFR)',
    'ssa-nsga2-xgb':     'SSA-NSGA-II (XGBoost)',
    'gpsaf-default':    'GPSAF-NSGA-II (default)',
    'gpsaf-rfr':        'GPSAF-NSGA-II (RFR)',
    'gpsaf-xgb':        'GPSAF-NSGA-II (XGBoost)',
    'cobra':            'IOC-SAMO-COBRA',
    'parego':           'ParEGO',
}

# Groups used by the grid plot.  Each entry is (group_title, [method_keys]).
METHOD_GROUPS = [
    ('Baselines',          ['random', 'nsga2']),
    ('SAMOS',              ['samos-rfr', 'samos-xgb']),
    ('MO-Bayesian',        ['mosmac', 'parego', 'cobra']),
    ('SSA-NSGA-II',        ['ssa-nsga2-default', 'ssa-nsga2-rfr', 'ssa-nsga2-xgb']),
    ('GPSAF-NSGA-II',      ['gpsaf-default', 'gpsaf-rfr', 'gpsaf-xgb']),
]


def _resolve_style(method: str, colours: dict, labels: dict):
    """Return (colour, label) for *method*, with prefix-matching fallback.

    Exact match is tried first.  If not found, the longest COLOURS key that
    is a strict prefix of *method* (followed by '-') is used, allowing e.g.
    'samos-xgb-i200-g20' to inherit 'samos-xgb' colour/label.
    """
    colour = colours.get(method)
    label  = labels.get(method)
    if colour is None or label is None:
        best_key, best_len = None, 0
        for key in colours:
            prefix = key + '-'
            if method.startswith(prefix) and len(prefix) > best_len:
                best_key, best_len = key, len(prefix)
        if best_key is not None:
            if colour is None:
                colour = colours[best_key]
            if label is None:
                suffix = method[len(best_key):].lstrip('-')
                label  = labels.get(best_key, best_key) + f' [{suffix}]'
    return colour, (label if label is not None else method)


# ─── data loading ─────────────────────────────────────────────────────────────

def load_indicator_trajectories(method: str, n_gen: int, results_root: str):
    """Return (hv_mean, hv_std, igd_mean, igd_std) arrays of length n_gen,
    or None if no result files are found for the method."""
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None

    hv_runs, igd_runs = [], []
    for pkl_file in sorted(os.listdir(seed_dir)):
        if not pkl_file.endswith('.pkl'):
            continue
        with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
            data = pickle.load(f)

        indicators = data.get('indicators', [])
        hv_series  = [ind.get('hv',       0.0)  for ind in indicators]
        igd_series = [ind.get('igd_plus', np.nan) for ind in indicators]

        # Each generation appends pop_size entries; keep one per generation
        # by sampling every pop_size-th entry (last entry of each gen).
        # If the list length == n_gen already (one per gen), use as-is.
        if len(hv_series) == n_gen:
            hv_runs.append(hv_series)
            igd_runs.append(igd_series)
        elif len(hv_series) >= n_gen:
            step = len(hv_series) // n_gen
            hv_runs.append( [hv_series [min((g + 1) * step - 1, len(hv_series)  - 1)] for g in range(n_gen)])
            igd_runs.append([igd_series[min((g + 1) * step - 1, len(igd_series) - 1)] for g in range(n_gen)])
        else:
            hv_runs.append(hv_series)
            igd_runs.append(igd_series)

    if not hv_runs:
        return None

    # Pad shorter runs to the same length with their last value
    max_len = max(len(r) for r in hv_runs)
    for r in hv_runs:
        while len(r) < max_len:
            r.append(r[-1])
    for r in igd_runs:
        while len(r) < max_len:
            r.append(r[-1])

    hv_arr  = np.array(hv_runs,  dtype=float)
    igd_arr = np.array(igd_runs, dtype=float)
    return (
        hv_arr.mean(axis=0),  hv_arr.std(axis=0),
        igd_arr.mean(axis=0), igd_arr.std(axis=0),
    )


# ─── plotting ─────────────────────────────────────────────────────────────────

def plot_results(
    methods: list,
    n_gen: int,
    pop_size: int,
    hv_ceiling: float,
    out_path: str,
    results_root: str,
    colours: dict = None,
    labels: dict = None,
    title: str = None,
):
    """Plot HV and IGD+ trajectories (mean ± std over seeds) for each method.

    Parameters
    ----------
    methods:      method keys to plot (must match sub-directories in results_root)
    n_gen:        number of generations (x-axis is scaled by pop_size)
    pop_size:     population size used in the runs
    hv_ceiling:   optimal HV value drawn as a dashed reference line
    out_path:     file path to save the figure (PNG)
    results_root: root directory containing per-method result folders
    colours:      optional dict overriding COLOURS for specific methods
    labels:       optional dict overriding LABELS for specific methods
    title:        optional suptitle; defaults to the NASBench-101 description
    """
    _colours = {**COLOURS, **(colours or {})}
    _labels  = {**LABELS,  **(labels  or {})}

    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for method in methods:
        result = load_indicator_trajectories(method, n_gen, results_root)
        if result is None:
            print(f'  [plot] No data found for method={method}, skipping.')
            continue
        hv_mean, hv_std, igd_mean, igd_std = result
        x      = np.arange(1, len(hv_mean) + 1) * pop_size
        colour, label = _resolve_style(method, _colours, _labels)

        axes[0].plot(x, hv_mean, label=label, color=colour, linewidth=1.8)
        axes[0].fill_between(x, hv_mean - hv_std, hv_mean + hv_std,
                             alpha=0.15, color=colour)

        axes[1].plot(x, igd_mean, label=label, color=colour, linewidth=1.8)
        axes[1].fill_between(x, np.maximum(0.0, igd_mean - igd_std), igd_mean + igd_std,
                             alpha=0.15, color=colour)

    axes[0].axhline(hv_ceiling, color='black', linestyle='--', linewidth=1.0,
                    label=f'Optimal HV ({hv_ceiling:.4f})')

    axes[0].set_title('Hypervolume (Higher is better)')
    axes[0].set_xlabel('Evaluations')
    axes[0].set_ylabel('Hypervolume')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('IGD+ (Lower is better)')
    axes[1].set_xlabel('Evaluations')
    axes[1].set_ylabel('IGD+')
    axes[1].grid(True, alpha=0.3)

    # Create shared legend below both plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

    _title = title if title is not None else (
        f'NASBench-101  --  test_acc@108  x  n_params\n'
        f'(pop={pop_size}, {n_gen} generations = {pop_size * n_gen} evals, mean ± std over seeds)'
    )
    fig.suptitle(_title, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'  Plot saved -> {out_path}')
    plt.close(fig)


def plot_results_grid(
    methods: list,
    n_gen: int,
    pop_size: int,
    hv_ceiling: float,
    out_path: str,
    results_root: str,
    groups: list = None,
    colours: dict = None,
    labels: dict = None,
    title: str = None,
):
    """Extended comparison plot: one column per method group, two rows (HV / IGD+).

    Keeps each algorithm family in its own subplot column so lines don't
    overlap each other, while still sharing the same x-axis scale across
    columns for easy visual comparison.

    Parameters
    ----------
    methods:  all method keys to include (subset determined by groups)
    groups:   list of ``(group_title, [method_keys])`` tuples.
              Defaults to ``METHOD_GROUPS`` defined in this module, filtered
              to only include methods that appear in *methods*.
    All other parameters: same as ``plot_results``.
    """
    _colours = {**COLOURS, **(colours or {})}
    _labels  = {**LABELS,  **(labels  or {})}

    # Build active groups: keep only groups whose methods intersect with
    # the requested method list, and filter each group's method list.
    method_set = set(methods)
    _groups = groups if groups is not None else METHOD_GROUPS
    active_groups = [
        (title_g, [m for m in ms if m in method_set])
        for title_g, ms in _groups
        if any(m in method_set for m in ms)
    ]
    # Methods not covered by any group go into a catch-all column
    covered = {m for _, ms in active_groups for m in ms}
    leftover = [m for m in methods if m not in covered]
    if leftover:
        active_groups.append(('SAMOS', leftover))

    n_cols = len(active_groups)

    matplotlib.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 150,
    })

    col_width  = 3.5
    row_height = 3.2
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(col_width * n_cols, row_height * 2),
        sharey='row',
        sharex='col',
    )
    # Ensure axes is always 2-D
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col_idx, (group_title, group_methods) in enumerate(active_groups):
        ax_hv  = axes[0, col_idx]
        ax_igd = axes[1, col_idx]

        for method in group_methods:
            result = load_indicator_trajectories(method, n_gen, results_root)
            if result is None:
                print(f'  [plot_grid] No data for method={method}, skipping.')
                continue
            hv_mean, hv_std, igd_mean, igd_std = result
            x      = np.arange(1, len(hv_mean) + 1) * pop_size
            colour, label = _resolve_style(method, _colours, _labels)

            ax_hv.plot(x, hv_mean, label=label, color=colour, linewidth=1.6)
            ax_hv.fill_between(x, hv_mean - hv_std, hv_mean + hv_std,
                               alpha=0.15, color=colour)

            ax_igd.plot(x, igd_mean, label=label, color=colour, linewidth=1.6)
            ax_igd.fill_between(x, np.maximum(0.0, igd_mean - igd_std), igd_mean + igd_std,
                                alpha=0.15, color=colour)

        ax_hv.axhline(hv_ceiling, color='black', linestyle='--', linewidth=0.8,
                      label=f'Opt. HV ({hv_ceiling:.3f})')

        ax_hv.set_title(group_title)
        ax_hv.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=True)
        ax_hv.grid(True, alpha=0.3)

        ax_igd.set_xlabel('Evaluations')
        ax_igd.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=True)
        ax_igd.grid(True, alpha=0.3)

        if col_idx == 0:
            ax_hv.set_ylabel('Hypervolume ↑')
            ax_igd.set_ylabel('IGD+ ↓')

    _title = title if title is not None else (
        f'Extended comparison  —  HV / IGD+ convergence\n'
        f'(pop={pop_size}, {n_gen} gens = {pop_size * n_gen} evals, mean ± std)'
    )
    fig.suptitle(_title, y=1.01, fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'  Plot saved -> {out_path}')
    plt.close(fig)


# ─── val_acc_12 trajectory helpers ────────────────────────────────────────────

_REF_POINT_VAL = np.array([1.05, 1.05])


def load_val_indicator_trajectories(
    method: str,
    n_gen: int,
    results_root: str,
    bench_db: dict,
    val_pareto_ref: np.ndarray,
):
    """Recompute HV / IGD+ trajectories using val_acc_12 instead of test_acc_108.

    For each generation we re-evaluate the stored var_archive on val_acc_12,
    build a non-dominated front, and compute HV / IGD+ against val_pareto_ref.
    Returns (hv_mean, hv_std, igd_mean, igd_std) arrays of length n_gen,
    or None if no result files are found.
    """
    hv_ind  = HV(ref_point=_REF_POINT_VAL)
    igd_ind = IGDPlus(val_pareto_ref)

    from problem.nasbench101_utils import _vec_to_arch_str, MIN_PARAMS, MAX_PARAMS

    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None

    hv_runs, igd_runs = [], []
    for pkl_file in sorted(os.listdir(seed_dir)):
        if not pkl_file.endswith('.pkl'):
            continue
        with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
            data = pickle.load(f)

        var_archives = data.get('var_archive', [])
        hv_series, igd_series = [], []

        for var_arch in var_archives:
            val_objs = []
            for vec in var_arch:
                arch_str = _vec_to_arch_str(np.array(vec), bench_db)
                if arch_str is None or arch_str not in bench_db:
                    continue
                entry = bench_db[arch_str]
                if 'val_acc_12' not in entry:
                    continue
                val_err = 1.0 - entry['val_acc_12']
                n_params_norm = (entry['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)
                val_objs.append((val_err, n_params_norm))

            if not val_objs:
                hv_series.append(0.0)
                igd_series.append(np.nan)
                continue

            F = np.array(val_objs)
            # keep only non-dominated points
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and F[j, 0] <= F[i, 0] and F[j, 1] <= F[i, 1] and (F[j] != F[i]).any():
                        nd_mask[i] = False
                        break
            F_nd = F[nd_mask]
            hv_series.append(float(hv_ind(F_nd)))
            igd_series.append(float(igd_ind(F_nd)))

        # Downsample / pad to n_gen — same logic as load_indicator_trajectories
        if len(hv_series) == n_gen:
            hv_runs.append(hv_series)
            igd_runs.append(igd_series)
        elif len(hv_series) >= n_gen:
            step = len(hv_series) // n_gen
            hv_runs.append([hv_series[min((g + 1) * step - 1, len(hv_series) - 1)]
                            for g in range(n_gen)])
            igd_runs.append([igd_series[min((g + 1) * step - 1, len(igd_series) - 1)]
                             for g in range(n_gen)])
        else:
            hv_runs.append(hv_series)
            igd_runs.append(igd_series)

    if not hv_runs:
        return None

    max_len = max(len(r) for r in hv_runs)
    for r in hv_runs:
        while len(r) < max_len:
            r.append(r[-1])
    for r in igd_runs:
        while len(r) < max_len:
            r.append(r[-1])

    hv_arr  = np.array(hv_runs,  dtype=float)
    igd_arr = np.array(igd_runs, dtype=float)
    return (
        hv_arr.mean(axis=0),  hv_arr.std(axis=0),
        igd_arr.mean(axis=0), igd_arr.std(axis=0),
    )


def plot_val_results(
    methods: list,
    n_gen: int,
    pop_size: int,
    hv_ceiling: float,
    out_path: str,
    results_root: str,
    bench_db: dict,
    val_pareto_ref: np.ndarray,
    colours: dict = None,
    labels: dict = None,
):
    """Plot HV and IGD+ trajectories based on val_acc_12 (mean ± std over seeds).

    Parameters mirror plot_results(); val_pareto_ref and bench_db are additionally
    required to recompute objectives from the stored var_archive.
    """
    _colours = {**COLOURS, **(colours or {})}
    _labels  = {**LABELS,  **(labels  or {})}

    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for method in methods:
        result = load_val_indicator_trajectories(
            method, n_gen, results_root, bench_db, val_pareto_ref
        )
        if result is None:
            print(f'  [plot_val] No data found for method={method}, skipping.')
            continue
        hv_mean, hv_std, igd_mean, igd_std = result
        x      = np.arange(1, len(hv_mean) + 1) * pop_size
        colour = _colours.get(method)
        label  = _labels.get(method, method)

        axes[0].plot(x, hv_mean, label=label, color=colour, linewidth=1.8)
        axes[0].fill_between(x, hv_mean - hv_std, hv_mean + hv_std,
                             alpha=0.15, color=colour)

        axes[1].plot(x, igd_mean, label=label, color=colour, linewidth=1.8)
        axes[1].fill_between(x, np.maximum(0.0, igd_mean - igd_std), igd_mean + igd_std,
                             alpha=0.15, color=colour)

    axes[0].axhline(hv_ceiling, color='black', linestyle='--', linewidth=1.0,
                    label=f'Optimal HV ({hv_ceiling:.4f})')

    axes[0].set_title('Hypervolume (Higher is better)')
    axes[0].set_xlabel('Evaluations')
    axes[0].set_ylabel('Hypervolume')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('IGD+ (Lower is better)')
    axes[1].set_xlabel('Evaluations')
    axes[1].set_ylabel('IGD+')
    axes[1].grid(True, alpha=0.3)

    # Create shared legend below both plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

    fig.suptitle(
        f'NASBench-101  --  val_acc@12  x  n_params\n'
        f'(pop={pop_size}, {n_gen} generations = {pop_size * n_gen} evals, mean ± std over seeds)',
        y=1.01,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'  Plot saved -> {out_path}')
    plt.close(fig)


# ─── exploration coverage helpers ─────────────────────────────────────────────

def _value_at_budget(hv_x, hv_y, budget):
    """Return the last HV value recorded at or before *budget* evaluations."""
    hv_x = np.asarray(hv_x)
    hv_y = np.asarray(hv_y)
    idx = np.searchsorted(hv_x, budget, side='right') - 1
    if idx < 0:
        return float('nan')
    return float(hv_y[min(idx, len(hv_y) - 1)])


def _arch_counts_across_seeds(entries, bench_db, checkpoint=None):
    """Return ``{arch_str: seed_count}`` for evaluated architectures.

    Parameters
    ----------
    entries:    List of seed dicts, each with a ``log_archs`` key (list of
                26-element int vectors, ordered by evaluation time).
    bench_db:   Full NASBench-101 dict (arch_str -> entry).
    checkpoint: If not None, only consider the first *checkpoint* unique
                architectures per seed.
    """
    from problem.nasbench101_utils import _vec_to_arch_str
    counts = {}
    for entry in entries:
        log_archs = entry['log_archs']
        arch_slice = log_archs if checkpoint is None else log_archs[:checkpoint]
        seen_this_seed = set()
        for vec in arch_slice:
            arch_str = _vec_to_arch_str(np.array(vec), bench_db)
            if arch_str is None or arch_str not in bench_db:
                continue
            if arch_str not in seen_this_seed:
                seen_this_seed.add(arch_str)
                counts[arch_str] = counts.get(arch_str, 0) + 1
    return counts


# ─── exploration coverage plot ────────────────────────────────────────────────

def plot_exploration_coverage(
    methods,
    results_root,
    bench_db,
    pareto_ref=None,
    *,
    acc_key='test_acc_108',
    eval_checkpoints=None,
    pop_size=20,
    out_path,
    xlim=(75, 100),
    ylim=(0, 0.25),
    show=False,
):
    """Side-by-side density maps of the architectures explored by each method.

    Background: all benchmark architectures as light-gray dots (rasterised).
    Foreground: architectures evaluated by each method (union of all seeds),
    coloured by how many seeds evaluated that architecture (1 → n_seeds).

    If *eval_checkpoints* is given (e.g. ``(500, 1000)``), the figure becomes a
    grid where each row corresponds to a cumulative-evaluation budget and only
    architectures seen up to that budget are shown.  Pass ``None`` (default) to
    show the full-run coverage in a single row.

    Parameters
    ----------
    methods:          List of method keys (sub-directories under results_root).
    results_root:     Root directory containing per-method result folders.
    bench_db:         Full NASBench-101 benchmark dict (arch_str -> entry).
    pareto_ref:       Optional Pareto front array (N x 2, minimisation space).
    acc_key:          Benchmark field for accuracy (x-axis); multiplied by 100.
    eval_checkpoints: Tuple of cumulative-evaluation budgets to show as rows.
    pop_size:         Population size used in the runs (for HV x-axis).
    out_path:         Path to save the output figure.
    xlim:             (min, max) for the accuracy x-axis in percent (default: (75, 100)).
    ylim:             (min, max) for the n_params y-axis (default: (0, 0.25)).
    show:             If True, call plt.show() after saving.
    """
    from problem.nasbench101_utils import MIN_PARAMS, MAX_PARAMS

    # ── load per-seed PKL data ─────────────────────────────────────────────────
    method_entries = {}   # method -> list of {log_archs, hv_x, hv_y}
    for method in methods:
        seed_dir = os.path.join(results_root, method)
        entries = []
        if os.path.isdir(seed_dir):
            for pkl_file in sorted(os.listdir(seed_dir)):
                if not pkl_file.endswith('.pkl'):
                    continue
                with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
                    data = pickle.load(f)
                indicators = data.get('indicators', [])
                hv_y = [ind.get('hv', 0.0) for ind in indicators]
                hv_x = [(g + 1) * pop_size for g in range(len(hv_y))]
                entries.append({
                    'log_archs': data.get('log_archs', []),
                    'hv_x': hv_x,
                    'hv_y': hv_y,
                })
        method_entries[method] = entries

    # ── background: full benchmark in plot coordinates ────────────────────────
    all_acc = np.array([
        v[acc_key] * 100.0
        for v in bench_db.values()
        if acc_key in v
    ])
    all_par = np.array([
        (v['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)
        for v in bench_db.values()
        if acc_key in v
    ], dtype=float)

    # ── Pareto front overlay in plot coordinates ──────────────────────────────
    pf_plot_x = pf_plot_y = None
    if pareto_ref is not None:
        pf_plot_x = (1.0 - pareto_ref[:, 0]) * 100.0
        pf_plot_y = pareto_ref[:, 1]
        order_pf  = np.argsort(pf_plot_x)
        pf_plot_x = pf_plot_x[order_pf]
        pf_plot_y = pf_plot_y[order_pf]

    # ── figure layout ─────────────────────────────────────────────────────────
    rows   = list(eval_checkpoints) if eval_checkpoints else [None]
    n_rows = len(rows)
    n_cols = len(methods)

    matplotlib.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'legend.fontsize': 7,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharex=True, sharey=True,
        squeeze=False,
    )

    for row_idx, checkpoint in enumerate(rows):
        row_label = (
            f'\u2264 {checkpoint:,} evals' if checkpoint is not None else 'Full run'
        )

        for col_idx, method in enumerate(methods):
            ax      = axes[row_idx, col_idx]
            entries = method_entries[method]
            n_seeds = len(entries)

            # Background
            ax.scatter(
                all_acc, all_par,
                s=0.4, c='#d4d4d4', alpha=1.0,
                rasterized=True, zorder=1, linewidths=0,
            )

            # Arch counts and HV stats
            arch_counts = _arch_counts_across_seeds(entries, bench_db, checkpoint)

            hv_suffix = ''
            hv_vals = []
            for entry in entries:
                hv_x = np.asarray(entry['hv_x'])
                hv_y = np.asarray(entry['hv_y'])
                if len(hv_y) == 0:
                    continue
                val = (
                    float(hv_y[-1])
                    if checkpoint is None
                    else _value_at_budget(hv_x, hv_y, checkpoint)
                )
                hv_vals.append(val)
            hv_vals = [v for v in hv_vals if not np.isnan(v)]
            if hv_vals:
                hv_suffix = (
                    f'\nHV: {np.mean(hv_vals):.4f} \u00b1 {np.std(hv_vals):.4f}'
                )

            display_label = LABELS.get(method, method)

            if arch_counts:
                eval_hashes = list(arch_counts.keys())
                eval_counts = np.array(
                    [arch_counts[h] for h in eval_hashes], dtype=float
                )
                eval_acc = np.array(
                    [bench_db[h][acc_key] * 100.0 for h in eval_hashes]
                )
                eval_par = np.array(
                    [(bench_db[h]['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)
                     for h in eval_hashes],
                    dtype=float,
                )
                order = np.argsort(eval_counts)
                sc = ax.scatter(
                    eval_acc[order], eval_par[order],
                    c=eval_counts[order], cmap='plasma',
                    s=6, alpha=0.85, zorder=2,
                    vmin=1, vmax=max(n_seeds, int(eval_counts.max())),
                    rasterized=True, linewidths=0,
                )
                cbar = plt.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label('Seeds evaluated', fontsize=7)
                if n_seeds <= 10:
                    cbar.set_ticks(range(1, n_seeds + 1))
                pct       = len(eval_counts) / len(bench_db) * 100
                col_title = (
                    f'{display_label}\n'
                    f'{len(eval_counts):,} unique archs  ({pct:.1f}%,  '
                    f'{n_seeds} seeds)'
                    f'{hv_suffix}'
                )
            else:
                col_title = f'{display_label}  (no data)'

            if row_idx == 0:
                ax.set_title(col_title, fontsize=9)
            else:
                if arch_counts:
                    pct = len(arch_counts) / len(bench_db) * 100
                    ax.set_title(
                        f'{len(arch_counts):,} unique archs  ({pct:.1f}%)'
                        f'{hv_suffix}',
                        fontsize=9,
                    )

            if col_idx == 0:
                ax.set_ylabel(
                    f'{row_label}\nn_params (norm.)  \u2192 lower is better',
                    fontsize=9,
                )

            if row_idx == n_rows - 1:
                ax.set_xlabel(f'{acc_key} (%)', fontsize=8)

            if pf_plot_x is not None:
                ax.step(
                    pf_plot_x, pf_plot_y,
                    where='post',
                    color='black', lw=1.5, ls='--',
                    label='True Pareto Front',
                    zorder=3, alpha=0.3,
                )
                ax.legend(loc='upper left', fontsize=7, markerscale=3)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  [OK] Saved: {out_path}')
    if show:
        plt.show()
    plt.close(fig)


# ─── 50 % attainment surface ──────────────────────────────────────────────────

def compute_attainment_surface(fronts_2d: list, q: float = 0.5, n_grid: int = 500):
    """Compute the q-attainment surface for a collection of 2-D Pareto fronts.

    Parameters
    ----------
    fronts_2d : list of (n_i, 2) arrays
        One ND-front per seed (minimisation space, F1 on axis 0, F2 on axis 1).
    q         : float
        Quantile level (0.5 == 50 % attainment surface).
    n_grid    : int
        Number of evenly spaced points along the F1 axis.

    Returns
    -------
    f1_grid : (n_grid,) array
    f2_att  : (n_grid,) array  — q-th quantile of achievable F2 across seeds.
    """
    if not fronts_2d:
        return np.array([]), np.array([])

    all_f1 = np.concatenate([f[:, 0] for f in fronts_2d])
    f1_min, f1_max = all_f1.min(), all_f1.max()
    f1_grid = np.linspace(f1_min, f1_max, n_grid)

    f2_matrix = np.full((len(fronts_2d), n_grid), np.inf)
    for s, front in enumerate(fronts_2d):
        if len(front) == 0:
            continue
        f1s, f2s = front[:, 0], front[:, 1]
        order = np.argsort(f1s)
        f1s, f2s = f1s[order], f2s[order]
        for gi, q1 in enumerate(f1_grid):
            mask = f1s <= q1
            if mask.any():
                f2_matrix[s, gi] = f2s[mask].min()

    # replace inf with nan before computing quantile so seeds that do not
    # reach a given F1 value do not distort the surface (they are ignored)
    f2_matrix[np.isinf(f2_matrix)] = np.nan
    with np.errstate(all='ignore'):
        f2_att = np.nanquantile(f2_matrix, q, axis=0)
    return f1_grid, f2_att


# ─── Pareto snapshot plot ─────────────────────────────────────────────────────

def _load_nd_fronts_at_gen(method: str, gen_idx: int, results_root: str,
                            is_samos: bool) -> list:
    """Return a list of (n_nd, 2) ND-front arrays, one per seed pkl file.

    For SAMOS methods the archive at gen_idx already holds all evaluated
    points; for others (random, nsga2, mosmac) we stack obj_pop[0..gen_idx]
    and then filter to the non-dominated front.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return []

    fronts = []
    for pkl_file in sorted(os.listdir(seed_dir)):
        if not pkl_file.endswith('.pkl'):
            continue
        with open(os.path.join(seed_dir, pkl_file), 'rb') as fh:
            data = pickle.load(fh)
        obj_pop = data.get('obj_pop', [])
        if not obj_pop:
            continue
        idx = min(gen_idx, len(obj_pop) - 1)
        if is_samos:
            F = np.asarray(obj_pop[idx])
        else:
            F = np.vstack([np.asarray(obj_pop[g]) for g in range(idx + 1)
                           if len(obj_pop[g]) > 0])
        if len(F) == 0 or F.ndim != 2 or F.shape[1] < 2:
            continue
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        fronts.append(F[nd_idx][:, :2])
    return fronts


def plot_pareto_snapshots(
    methods: list,
    n_gen: int,
    pop_size: int,
    n_var: int,
    pf: np.ndarray,
    out_path: str,
    results_root: str,
    checkpoints_d: list = None,
    colours: dict = None,
    labels: dict = None,
    title: str = None,
):
    """Plot 50 % attainment surfaces at several evaluation budget checkpoints.

    One subplot per method, arranged in a grid.  Each subplot shows the
    reference Pareto front (dashed black) and four attainment surface curves
    coloured light-to-dark according to the evaluation budget.

    Parameters
    ----------
    methods        : list of method keys (sub-directories in results_root)
    n_gen          : total number of outer generations
    pop_size       : evaluations per generation
    n_var          : number of decision variables (used to compute checkpoints)
    pf             : reference Pareto front (N, 2), minimisation space
    out_path       : output PNG path
    results_root   : root directory containing per-method result folders
    checkpoints_d  : list of multipliers d such that budget = d * n_var
                     (default: [10, 25, 50, 100])
    """
    if checkpoints_d is None:
        checkpoints_d = [10, 25, 50, 100]

    _colours = {**COLOURS, **(colours or {})}
    _labels  = {**LABELS,  **(labels  or {})}

    checkpoint_palette = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
    while len(checkpoint_palette) < len(checkpoints_d):
        checkpoint_palette.append('#08306b')

    n_methods = len(methods)
    n_cols    = min(3, n_methods)
    n_rows    = int(np.ceil(n_methods / n_cols))

    matplotlib.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    # Sort reference PF by F1.
    # Use scatter (not a connected line) so disconnected fronts (e.g. WFG2)
    # are rendered correctly without spurious cross-segment lines.
    pf_sorted = pf[np.argsort(pf[:, 0])]

    for idx, method in enumerate(methods):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        colour, label = _resolve_style(method, _colours, _labels)
        is_samos = method.startswith('samos-')

        ax.scatter(pf_sorted[:, 0], pf_sorted[:, 1],
                   c='black', s=4, marker='.', label='Reference PF', zorder=5)

        for ci, cd in enumerate(checkpoints_d):
            budget  = cd * n_var
            gen_idx = max(0, min(budget // pop_size - 1, n_gen - 1))
            fronts  = _load_nd_fronts_at_gen(method, gen_idx, results_root, is_samos)
            if not fronts:
                continue
            f1_grid, f2_att = compute_attainment_surface(fronts)
            if len(f1_grid) == 0:
                continue
            # Use scatter (not a connected step-line) so disconnected fronts
            # (e.g. WFG2) don't have spurious lines drawn across the gaps.
            valid = ~np.isnan(f2_att)
            ax.scatter(f1_grid[valid], f2_att[valid],
                       c=checkpoint_palette[ci], s=3, marker='.',
                       label=f'{budget} evals ({cd}d)', zorder=4 - ci)

        ax.set_title(label)
        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    # Hide unused subplots
    for idx in range(n_methods, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    _title = title or 'Pareto snapshot — 50 % attainment surfaces'
    fig.suptitle(_title, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'  Plot saved -> {out_path}')
    plt.close(fig)
