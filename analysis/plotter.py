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

from problem.nasbench101_utils import _vec_to_arch_str, MIN_PARAMS, MAX_PARAMS


# ─── method styling ───────────────────────────────────────────────────────────

COLOURS = {
    'random':       '#4e79a7',
    'nsga2':        '#f28e2b',
    'nsga2-single': '#59a14f',
    'samos-rfr':    '#e15759',
    'samos-xgb':    '#b07aa1',
}
LABELS = {
    'random':       'Random',
    'nsga2':        'NSGA-II (2-pt XO, unif. mut)',
    'nsga2-single': 'NSGA-II (no XO, single-pt mut)',
    'samos-rfr':    'SAMOS (RFR surrogate)',
    'samos-xgb':    'SAMOS (XGBoost surrogate)',
}


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
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('IGD+ (Lower is better)')
    axes[1].set_xlabel('Evaluations')
    axes[1].set_ylabel('IGD+')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f'NASBench-101  --  test_acc@108  x  n_params\n'
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
    show:             If True, call plt.show() after saving.
    """
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

            ax.set_xlim(75, 100)
            ax.set_ylim(0, 0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  [OK] Saved: {out_path}')
    if show:
        plt.show()
    plt.close(fig)
