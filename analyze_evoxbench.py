"""analyze_evoxbench.py — Post-run analysis for EvoXBench experiments.

Produces per-PID convergence plots (HV / IGD+), an across-PIDs grid plot,
and a LaTeX summary table.

Results are expected at:
  results/evoxbench/<suite>/pid<pid>/B<budget>_P<pop_size>/<method>/seed_<seed>.pkl

Run examples:
  # C10MOP — all 9 PIDs, all methods
  python analyze_evoxbench.py --suite c10mop --pop_size 20 --n_gen 50

  # IN1K — specific PIDs and methods
  python analyze_evoxbench.py \\
      --suite in1kmop --pids 1 2 3 --pop_size 20 --n_gen 50 \\
      --methods random nsga2 samos-xgb samos2 parego gpsaf-default mosmac
"""

import argparse
import os
import pickle

import numpy as np
from pymoo.indicators.hv import HV
from scipy.stats import ranksums

from analysis.plotter import (
    plot_results,
    plot_results_grid,
    plot_results_precomputed,
    load_indicator_trajectories,
    _resolve_style,
    COLOURS,
    LABELS,
    plot_pareto_snapshots_evoxbench_overlay,
    plot_pareto_snapshots_evoxbench_subplots,
)
from analysis.convergence import (
    build_pareto_approximation,
    compute_empirical_norm_bounds,
    recompute_indicator_trajectories,
    recompute_final_indicators,
    recompute_final_indicators_seeds,
    get_or_recompute_trajectories,
    save_approx_cache,
)
from problem.evoxbench.utils import get_benchmark
from problem.evoxbench.benchmark_meta import pid_header, get_obj_labels, BENCHMARK_META

# ─── evoxbench-specific style overrides ───────────────────────────────────────

_EVOX_COLOURS = {
    **COLOURS,
    'gpsaf-default': '#a0cbe8',
    'parego':        '#9c755f',
    'samos2':        '#8b0000',
}
_EVOX_LABELS = {
    **LABELS,
    'nsga2':         'NSGA-II',
    'samos-xgb':     'SAMOS (XGBoost)',
    'samos2':        'SAMOS2 (XGBoost)',
    'gpsaf-default': 'GPSAF',
    'parego':        'ParEGO',
    'mosmac':        'MO-SMAC',
}

_DEFAULT_METHODS = [
    'random', 'nsga2', 'parego', 'mosmac', 'gpsaf-default', 'samos-xgb', 'samos2',
]


# ─── helpers ──────────────────────────────────────────────────────────────────
def _results_root(suite: str, pid: int, pop_size: int, n_gen: int, root: str = 'results/evoxbench') -> str:
    return os.path.join(
        root, suite,
        f'pid{pid}', f'B{n_gen * pop_size}_P{pop_size}',
    )


def _hv_ceiling(suite: str, pid: int) -> float | None:
    """Return the HV ceiling for this benchmark, or None if the PF is unavailable."""
    try:
        bm = get_benchmark(suite, pid)
    except Exception as e:
        print(f'  [WARN] Could not instantiate benchmark {suite}/pid{pid}: {e}')
        return None
    pf_raw = getattr(bm, 'pareto_front', None)
    if pf_raw is None or len(pf_raw) == 0:
        return None
    n_obj = bm.evaluator.n_objs
    # Normalize PF the same way callbacks.py does (raw → [utopian→0, nadir→1])
    # so the ceiling is in the same scale as the stored indicators.
    pf_norm = bm.normalize(pf_raw)
    pf_norm = np.where(np.isfinite(pf_norm), pf_norm, 1.0)
    ref = np.ones(n_obj) * 1.05
    return float(HV(ref_point=ref)(pf_norm))


def _load_final_indicators(method: str, results_root: str) -> dict | None:
    """Return {'hv': (mean, std), 'igd_plus': (mean, std)} from last generation."""
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None
    hvs, igds = [], []
    for pkl_file in sorted(os.listdir(seed_dir)):
        if not pkl_file.endswith('.pkl'):
            continue
        try:
            with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f'  [WARN] Failed to load {pkl_file}: {e}')
            continue
        indicators = data.get('indicators', [])
        if not indicators:
            continue
        last = indicators[-1]
        hvs.append(last.get('hv', float('nan')))
        igds.append(last.get('igd_plus', float('nan')))
    if not hvs:
        return None
    return {
        'hv':       (float(np.nanmean(hvs)),  float(np.nanstd(hvs))),
        'igd_plus': (float(np.nanmean(igds)), float(np.nanstd(igds))),
    }


def _load_final_indicators_seeds(method: str, results_root: str) -> dict | None:
    """Return {'hv': np.ndarray, 'igd_plus': np.ndarray} of per-seed values."""
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None
    hvs, igds = [], []
    for pkl_file in sorted(os.listdir(seed_dir)):
        if not pkl_file.endswith('.pkl'):
            continue
        try:
            with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f'  [WARN] Failed to load {pkl_file}: {e}')
            continue
        indicators = data.get('indicators', [])
        if not indicators:
            continue
        last = indicators[-1]
        hvs.append(last.get('hv', float('nan')))
        igds.append(last.get('igd_plus', float('nan')))
    if not hvs:
        return None
    return {
        'hv':       np.array(hvs),
        'igd_plus': np.array(igds),
    }


# ─── LaTeX table ──────────────────────────────────────────────────────────────

_WILCOXON_REF   = 'samos-xgb'
_WILCOXON_ALPHA = 0.05


def _wilcoxon_marker(ref_vals, other_vals, higher_is_better: bool) -> str:
    """Wilcoxon rank-sum significance marker vs. the reference method.

    Returns ``$(+)$`` when the method is significantly better than the reference,
    ``$(-)$`` when the method is significantly worse, or ``$(\\approx)$`` when
    there is no significant difference (p >= 0.05).
    """
    if ref_vals is None or other_vals is None:
        return ''
    rv = np.asarray(ref_vals)
    ov = np.asarray(other_vals)
    rv = rv[np.isfinite(rv)]
    ov = ov[np.isfinite(ov)]
    if len(rv) < 3 or len(ov) < 3:
        return ''
    try:
        _, p = ranksums(rv, ov)
    except Exception:
        return r'$(\approx)$'
    if p >= _WILCOXON_ALPHA:
        return r'$(\approx)$'
    ref_med   = np.median(rv)
    other_med = np.median(ov)
    if higher_is_better:
        # HV: higher is better — method better means other_med > ref_med
        return r'$(+)$' if other_med > ref_med else r'$(-)$'
    else:
        # IGD+: lower is better — method better means other_med < ref_med
        return r'$(+)$' if other_med < ref_med else r'$(-)$'

def _fmt(mean: float, std: float, bold: bool, marker: str = '') -> str:
    s = f'{mean:.4f}\\,\\textpm\\,{std:.4f}'
    cell = f'\\textbf{{{s}}}' if bold else s
    return f'{cell}{marker}'


def _method_latex_label(method: str) -> str:
    return _EVOX_LABELS.get(method, method)


def generate_latex_table(
    suite: str,
    pids: list,
    methods: list,
    pop_size: int,
    n_gen: int,
    out_path: str,
    approx_info: dict = None,
    root: str = 'results/evoxbench',
) -> None:
    """Write a booktabs LaTeX table: rows = methods, columns = PIDs × {HV, IGD+}.

    When *approx_info* is provided (a ``{pid: approx_dict}`` mapping from
    ``build_pareto_approximation``), HV / IGD+ are recomputed against the
    shared combined Pareto approximation for each PID.  Otherwise the stored
    ``indicators[-1]`` values are used.

    Wilcoxon rank-sum tests (p<0.05) are computed per PID using
    ``_WILCOXON_REF`` (samos-xgb) as the reference.  Each cell of a
    non-reference method is annotated with ``$(+)$`` when the method is
    significantly better, ``$(-)$`` when significantly worse, or
    ``$(\\approx)$`` when not significant.

    Methods are always displayed in the canonical order defined by
    ``_DEFAULT_METHODS``; any methods not in that list are appended in the
    order they are received.
    """
    # Enforce canonical method order
    order = {m: i for i, m in enumerate(_DEFAULT_METHODS)}
    methods = sorted(methods, key=lambda m: order.get(m, len(_DEFAULT_METHODS)))
    # Collect per-seed data; derive (mean, std) stats from the same arrays
    all_seeds: dict[int, dict] = {}
    all_stats: dict[int, dict] = {}
    best_hv:   dict[int, str]  = {}
    best_igd:  dict[int, str]  = {}
    for pid in pids:
        root_pid   = _results_root(suite, pid, pop_size, n_gen, root)
        pid_approx = (approx_info or {}).get(pid)
        pid_seeds, pid_stats = {}, {}
        for m in methods:
            if pid_approx is not None:
                s = recompute_final_indicators_seeds(
                    m, root_pid, pid_approx['ref_point'], pid_approx['pareto_approx'],
                    norm_bounds=pid_approx.get('norm_bounds'),
                )
            else:
                s = _load_final_indicators_seeds(m, root_pid)
            pid_seeds[m] = s
            pid_stats[m] = {
                'hv':       (float(np.mean(s['hv'])),       float(np.std(s['hv']))),
                'igd_plus': (float(np.mean(s['igd_plus'])), float(np.std(s['igd_plus']))),
            } if s is not None else None
        all_seeds[pid] = pid_seeds
        all_stats[pid] = pid_stats
        valid_hv  = {m: v['hv'][0]       for m, v in pid_stats.items() if v}
        valid_igd = {m: v['igd_plus'][0]  for m, v in pid_stats.items() if v}
        best_hv[pid]  = max(valid_hv,  key=valid_hv.get)  if valid_hv  else None
        best_igd[pid] = min(valid_igd, key=valid_igd.get) if valid_igd else None

    hv_col  = r'HV\,($\uparrow$)'
    igd_col = r'IGD\textsuperscript{+}\,($\downarrow$)'

    # Two-column layout: split PIDs left/right
    mid         = (len(pids) + 1) // 2
    left_pids   = pids[:mid]
    right_pids  = pids[mid:]

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        (f'\\caption{{{suite.upper()} PID 1-{max(pids)}: final HV and '
         r'IGD\textsuperscript{+} (mean\,\textpm\,std). '
         r'\textbf{Bold}: best per PID. '
         r'Wilcoxon rank-sum vs.\ SAMOS\,(XGBoost) ($p{<}0.05$): '
         r'$(+)$\,better, $(-)$\,worse, $(\approx)$\,no significant difference.}'),
        f'\\label{{tab:{suite}_convergence}}',
        r'\resizebox{\linewidth}{!}{',
        r'\begin{tabular}{l r r r r}',
        r'\toprule',
        f'Method & {hv_col} & {igd_col} & {hv_col} & {igd_col} \\\\',
    ]

    for li, l_pid in enumerate(left_pids):
        r_pid = right_pids[li] if li < len(right_pids) else None

        # PID block header
        l_hdr = pid_header(suite, l_pid)
        lines.append(r'\midrule')
        if r_pid is not None:
            r_hdr = pid_header(suite, r_pid)
            lines.append(
                f' & \\multicolumn{{2}}{{c}}{{{l_hdr}}}'
                f' & \\multicolumn{{2}}{{c}}{{{r_hdr}}} \\\\'
            )
            lines.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
        else:
            lines.append(
                f' & \\multicolumn{{2}}{{c}}{{{l_hdr}}}'
                r' &  &  \\'
            )
            lines.append(r'\cmidrule(lr){2-3}')

        # Pre-fetch reference seed arrays for Wilcoxon comparisons
        lref = all_seeds[l_pid].get(_WILCOXON_REF)
        l_ref_hv  = lref['hv']       if lref else None
        l_ref_igd = lref['igd_plus'] if lref else None
        if r_pid is not None:
            rref = all_seeds[r_pid].get(_WILCOXON_REF)
            r_ref_hv  = rref['hv']       if rref else None
            r_ref_igd = rref['igd_plus'] if rref else None

        for method in methods:
            mlbl = _method_latex_label(method)
            is_ref = (method == _WILCOXON_REF)

            ls = all_stats[l_pid].get(method)
            if ls and not is_ref:
                lseed = all_seeds[l_pid].get(method)
                lhv_m  = _wilcoxon_marker(l_ref_hv,  lseed['hv']       if lseed else None, True)
                ligd_m = _wilcoxon_marker(l_ref_igd, lseed['igd_plus'] if lseed else None, False)
            else:
                lhv_m = ligd_m = ''
            lhv  = _fmt(*ls['hv'],      method == best_hv[l_pid],  lhv_m)  if ls else '--'
            ligd = _fmt(*ls['igd_plus'], method == best_igd[l_pid], ligd_m) if ls else '--'

            if r_pid is not None:
                rs = all_stats[r_pid].get(method)
                if rs and not is_ref:
                    rseed = all_seeds[r_pid].get(method)
                    rhv_m  = _wilcoxon_marker(r_ref_hv,  rseed['hv']       if rseed else None, True)
                    rigd_m = _wilcoxon_marker(r_ref_igd, rseed['igd_plus'] if rseed else None, False)
                else:
                    rhv_m = rigd_m = ''
                rhv  = _fmt(*rs['hv'],      method == best_hv[r_pid],  rhv_m)  if rs else '--'
                rigd = _fmt(*rs['igd_plus'], method == best_igd[r_pid], rigd_m) if rs else '--'
            else:
                rhv = rigd = ''

            lines.append(f'{mlbl} & {lhv} & {ligd} & {rhv} & {rigd} \\\\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'}',
        r'\end{table*}',
    ]

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  LaTeX table saved -> {out_path}')


# ─── exploration scatter plot ──────────────────────────────────────────────────

def _pareto_front_2d(pts: np.ndarray) -> np.ndarray:
    """Return the Pareto front of a 2-objective minimisation problem.

    O(N log N) time, O(N) extra memory — safe for 400k+ points.
    Sort by f1 ascending; then sweep and keep a point only when its f2 is
    strictly less than the minimum f2 seen so far (i.e. it is not dominated).
    """
    order      = np.argsort(pts[:, 0], kind='stable')
    sorted_pts = pts[order]
    pf_rows    = [0]        # index into sorted_pts
    min_f2     = sorted_pts[0, 1]
    for i in range(1, len(sorted_pts)):
        f1, f2 = sorted_pts[i]
        if f2 < min_f2:     # not dominated: f1 >= prev (sorted), f2 strictly better
            pf_rows.append(i)
            min_f2 = f2
    return sorted_pts[pf_rows]


def _load_nb101_background_norm(bm) -> np.ndarray:
    """Return (N, 2) normalised objective array for all NASBench-101 architectures.

    Values are cached to a .npy file alongside the SQLite database so that
    subsequent calls are instant.
    """
    import sqlite3, json

    # Locate the SQLite file from Django settings
    from django.conf import settings
    db_path = settings.DATABASES['default']['NAME']
    cache_path = os.path.join(os.path.dirname(db_path), '_nb101_all_norm_err_params.npy')

    if os.path.isfile(cache_path):
        return np.load(cache_path)

    print('  [background] Building NASBench-101 objective cache (one-time, ~7 s)…')
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute('SELECT params, final_validation_accuracy FROM nasbench101_nasbench101result')
    rows = cur.fetchall()
    conn.close()

    utopian = np.array(bm.utopian_point)
    nadir   = np.array(bm.nadir_point)

    errs, params = [], []
    for p, va in rows:
        acc108 = json.loads(va).get('epoch108', [])
        if acc108:
            errs.append(1.0 - float(np.mean(acc108)))
            params.append(float(p))

    raw  = np.column_stack([errs, params])
    norm = (raw - utopian) / (nadir - utopian + 1e-12)
    norm = np.clip(norm, -0.05, 1.5)      # keep outliers visible but bounded
    np.save(cache_path, norm)
    print(f'  [background] Cached {len(norm):,} architectures -> {cache_path}')
    return norm


def plot_exploration(
    suite: str,
    pid: int,
    methods: list,
    pop_size: int,
    n_gen: int,
    n_seeds: int | None = None,
    out_path: str | None = None,
    font_scale: float = 1.0,
    approx_info: dict | None = None,
) -> None:
    """Scatter plot of all evaluated designs across seeds for each method.

    Each subplot shows:
    - Grey background: every architecture in the NASBench-101 database.
    - Coloured dots: every unique design evaluated, coloured by the number of
      seeds that evaluated that exact architecture (seed-visit count).
    - Black stars: the true Pareto front computed from the full NASBench-101
      database (computed once from ``bg`` via non-dominated sorting).
    - Subplot title: method label, unique arch count (% of total), and HV
      recomputed against the shared combined Pareto approximation (matching
      the values in the LaTeX table).

    x-axis = objective 2 (normalised #params), y-axis = objective 1 (norm. err).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from collections import Counter

    # ── paper-ready typography ────────────────────────────────────────────────
    BASE_FS = 11 * font_scale
    plt.rcParams.update({
        'font.size':         BASE_FS,
        'axes.titlesize':    BASE_FS,
        'axes.labelsize':    BASE_FS,
        'xtick.labelsize':   BASE_FS * 0.9,
        'ytick.labelsize':   BASE_FS * 0.9,
        'legend.fontsize':   BASE_FS * 0.9,
        'figure.titlesize':  BASE_FS * 1.1,
    })

    results_root = _results_root(suite, pid, pop_size, n_gen)

    # ── benchmark meta ────────────────────────────────────────────────────────
    bm     = get_benchmark(suite, pid)
    xlabel, ylabel = get_obj_labels(suite, pid)   # returns (f1_label, f2_label)
    # plot: x = col-1 (f2/#params), y = col-0 (f1/err)
    xlabel, ylabel = ylabel, xlabel
    n_total = 423624   # fixed NASBench-101 unique architecture count

    # ── background: all 423k normalised architectures ─────────────────────────
    bg = _load_nb101_background_norm(bm)  # (N,2): col0=f1(err), col1=f2(params)

    # ── TRUE Pareto front from the full database ───────────────────────────────
    # bm.pareto_front is only a pre-computed subset; compute the real front from
    # bg so that no evaluated design can appear "below" the displayed PF.
    # _pareto_front_2d is O(N log N) and avoids the O(N²) pymoo NDS memory cost.
    true_pf = _pareto_front_2d(bg)   # (K,2): col0=f1, col1=f2

    # ── load per-method data ──────────────────────────────────────────────────
    # For each method: collect the unique architectures each seed evaluated and
    # count how many seeds share each architecture (seed-visit count).
    #
    # We deliberately de-duplicate within each seed first so that an arch
    # evaluated multiple times in ONE seed still only contributes 1 to the
    # count — the colour represents "how many seeds explored here", not total
    # evaluations.
    method_data: dict = {}
    for method in methods:
        seed_dir = os.path.join(results_root, method)
        if not os.path.isdir(seed_dir):
            method_data[method] = None
            continue

        pkl_files = sorted(f for f in os.listdir(seed_dir) if f.endswith('.pkl'))
        if n_seeds is not None:
            pkl_files = pkl_files[:n_seeds]
        n_used = len(pkl_files)

        per_seed_sets: list[set] = []
        for fn in pkl_files:
            with open(os.path.join(seed_dir, fn), 'rb') as fh:
                d = pickle.load(fh)

            seed_pts: set = set()
            for gen_objs in d.get('obj_pop', []):
                arr = np.asarray(gen_objs, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    valid = arr[np.isfinite(arr[:, 0]) & (arr[:, 0] < 2.0)]
                    for row in np.round(valid, 6):
                        seed_pts.add((float(row[0]), float(row[1])))
            if seed_pts:
                per_seed_sets.append(seed_pts)

        if not per_seed_sets:
            method_data[method] = None
            continue

        # Cross-seed visit count per unique architecture
        visit_counts: Counter = Counter()
        for s in per_seed_sets:
            for pt in s:
                visit_counts[pt] += 1

        unique_pts = np.array(list(visit_counts.keys()))   # (M,2): col0=f1, col1=f2
        counts     = np.array(list(visit_counts.values()), dtype=int)  # (M,)

        # HV: recompute against shared combined Pareto approx (matches tables)
        if approx_info is not None:
            hv_data = recompute_final_indicators_seeds(
                method, results_root,
                approx_info['ref_point'],
                approx_info['pareto_approx'],
            )
            hvs = list(hv_data['hv']) if hv_data is not None else []
        else:
            # Fallback: stored indicators (different ref point — may differ from table)
            hvs = []
            for fn in pkl_files:
                with open(os.path.join(seed_dir, fn), 'rb') as fh:
                    d = pickle.load(fh)
                inds = d.get('indicators', [])
                if inds:
                    v = inds[-1].get('hv', float('nan'))
                    if np.isfinite(v) and v < _MAX_VALID_HV:
                        hvs.append(v)

        method_data[method] = {
            'pts':    unique_pts,
            'counts': counts,
            'n_used': n_used,
            'unique': len(unique_pts),
            'hvs':    hvs,
        }

    # ── figure layout ─────────────────────────────────────────────────────────
    n_methods = len(methods)
    ncols     = min(n_methods, 3)
    nrows     = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols * font_scale,
                                      5.0 * nrows * font_scale),
                             squeeze=False)

    for ax_idx, method in enumerate(methods):
        row, col = divmod(ax_idx, ncols)
        ax       = axes[row][col]

        # ── background ────────────────────────────────────────────────────────
        # ax.scatter(bg[:, 1], bg[:, 0],
        #            s=2, c='#cccccc', alpha=0.25, linewidths=0, zorder=1,
        #            rasterized=True)

        # ── true Pareto front ─────────────────────────────────────────────────
        ax.scatter(true_pf[:, 1], true_pf[:, 0],
                   s=20 * font_scale, c='black', marker='*', zorder=5,
                   label='True PF')

        # ── evaluated designs ─────────────────────────────────────────────────
        dat  = method_data.get(method)
        mlbl = _EVOX_LABELS.get(method, method)

        if dat is None:
            ax.set_title(f'{mlbl}\n(no data)')
        else:
            pts    = dat['pts']       # (M,2): col0=f1, col1=f2
            counts = dat['counts']    # (M,)  integers 1..n_used
            unique = dat['unique']
            n_used = dat['n_used']
            hvs    = dat['hvs']

            pct    = 100.0 * unique / n_total
            hv_str = (f'HV = {np.mean(hvs):.4f} \u00b1 {np.std(hvs):.4f}'
                      if hvs else 'HV = n/a')
            title  = f'{mlbl}\n{unique:,} unique ({pct:.1f}% of {n_total:,})\n{hv_str}'

            colour = _EVOX_COLOURS.get(method, '#1f77b4')
            cmap   = mcolors.LinearSegmentedColormap.from_list(
                'seeds', ['#e8e8e8', colour], N=max(n_used, 2))
            norm_c = mcolors.Normalize(vmin=0.5, vmax=n_used + 0.5)

            sc = ax.scatter(pts[:, 1], pts[:, 0],   # x=f2, y=f1
                            s=20 * font_scale, c=counts, cmap=cmap, norm=norm_c,
                            alpha=0.8, linewidths=0, zorder=3)
            # Show at most ~8 labelled ticks on the colorbar regardless of
            # how many seeds there are or how large font_scale is.
            _max_ticks = 8
            _step      = max(1, int(np.ceil(n_used / _max_ticks)))
            _ticks     = list(range(1, n_used + 1, _step))
            if _ticks[-1] != n_used:          # always include the maximum
                _ticks.append(n_used)
            cb = fig.colorbar(sc, ax=ax, shrink=0.7, ticks=_ticks)
            cb.set_label(f'# seeds (1\u2013{n_used})', fontsize=BASE_FS * 0.85)
            cb.ax.tick_params(labelsize=BASE_FS * 0.8)
            ax.set_title(title)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.02, 1.1)
        ax.set_ylim(-0.02, 1.1)
        ax.tick_params(axis='both', which='major')
        ax.grid(True, linewidth=0.5, alpha=0.5)

    # hide unused axes
    for ax_idx in range(len(methods), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f'{suite.upper()} PID {pid} — evaluated designs '
        f'(pop={pop_size}, {n_gen} gen, {n_seeds or "all"} seeds)',
        y=1.01,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(results_root, f'{suite}_pid{pid}_exploration.png')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Exploration plot saved -> {out_path}')


_MAX_VALID_HV = 2.0   # sentinel for NASBench-101 un-normalised HV values


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args) -> None:
    suite    = args.suite
    pids     = args.pids
    methods  = args.methods
    pop_size = args.pop_size
    n_gen    = args.n_gen

    suite_root = os.path.join(args.root, suite)

    # ── per-PID loop ──────────────────────────────────────────────────────────
    all_approx_info: dict[int, dict] = {}
    for pid in pids:
        root = _results_root(suite, pid, pop_size, n_gen, args.root)

        # When --no_norm, derive empirical normalization bounds from the raw
        # test_obj_archive values pooled across all methods and seeds under root.
        if getattr(args, 'no_norm', False):
            norm_bounds = compute_empirical_norm_bounds(methods, root)
        else:
            norm_bounds = None

        # Build (or load from cache) the combined Pareto approximation.
        # Always run — it is cheap (disk-cached) and needed by both the
        # convergence plot and the LaTeX table.
        approx = build_pareto_approximation(
            suite, pid, methods, pop_size, n_gen,
            force_rebuild=args.rebuild_approx,
            norm_bounds=norm_bounds,
            results_root=root,
        )
        all_approx_info[pid] = approx

        # Read benchmark metadata (lightweight; needed by multiple outputs).
        try:
            bm    = get_benchmark(suite, pid)
            n_var = bm.search_space.n_var
            n_obj = bm.evaluator.n_objs
        except Exception as _bm_err:
            print(f'  [pid{pid}] WARN: could not instantiate benchmark: {_bm_err}')
            bm, n_var, n_obj = None, None, None

        # ── convergence plot (HV / IGD+) ──────────────────────────────────────
        if args.convergence_plot:
            dim_str = (
                f'  |  {n_var} vars, {n_obj} objs'
                if n_var is not None and n_obj is not None else ''
            )

            if approx is not None:
                ceiling = float(HV(ref_point=approx['ref_point'])(approx['pareto_approx']))
                print(f'  [pid{pid}] Combined PF HV ceiling = {ceiling:.6f}')
            else:
                ceiling = _hv_ceiling(suite, pid)
                if ceiling is None:
                    _best = 0.0
                    for _m in methods:
                        _traj = load_indicator_trajectories(_m, n_gen, root)
                        if _traj is not None:
                            _best = max(_best, float(_traj[0][-1]))
                    ceiling = _best * 1.05 if _best > 0 else 1.0
                    print(f'  [pid{pid}] No PF available — proxy HV ceiling={ceiling:.4f}')
                else:
                    print(f'  [pid{pid}] Benchmark PF HV ceiling = {ceiling:.6f}')

            conv_out = os.path.join(root, f'{suite}_pid{pid}_hv_igd.png')

            if approx is not None:
                trajectories = {}
                for method in methods:
                    traj = get_or_recompute_trajectories(
                        method, n_gen, root, approx,
                        norm_bounds=norm_bounds,
                    )
                    if traj is None:
                        print(f'  [pid{pid}] No data for method={method}, skipping.')
                    trajectories[method] = traj

                save_approx_cache(root, approx)

                plot_results_precomputed(
                    trajectories=trajectories,
                    n_gen=n_gen,
                    pop_size=pop_size,
                    hv_ceiling=ceiling,
                    out_path=conv_out,
                    n_var=n_var,
                    n_obj=n_obj,
                    colours=_EVOX_COLOURS,
                    labels=_EVOX_LABELS,
                    title=(
                        f'{suite.upper()} - {pid}{dim_str}  '
                        f'(pop={pop_size}, {n_gen} gen = {pop_size * n_gen} evals, '
                        f'mean \u00b1 std over seeds, shared combined PF)'
                    ),
                )
            else:
                plot_results(
                    methods=methods,
                    n_gen=n_gen,
                    pop_size=pop_size,
                    hv_ceiling=ceiling,
                    out_path=conv_out,
                    results_root=root,
                    colours=_EVOX_COLOURS,
                    labels=_EVOX_LABELS,
                    title=(
                        f'{suite.upper()} - {pid}{dim_str}  '
                        f'(pop={pop_size}, {n_gen} gen = {pop_size * n_gen} evals, '
                        f'mean \u00b1 std over seeds)'
                    ),
                )

        # ── attainment surface plots ───────────────────────────────────────────
        if args.attainment:
            if n_obj != 2:
                print(f'  [pid{pid}] Skipping attainment plot (n_obj={n_obj}, only 2-obj supported)')
            else:
                if approx is not None:
                    pf_norm = approx['pareto_approx']
                elif bm is not None and getattr(bm, 'pareto_front', None) is not None:
                    pf_norm = bm.normalize(bm.pareto_front)
                    pf_norm = np.where(np.isfinite(pf_norm), pf_norm, 1.0)
                else:
                    pf_norm = None

                _meta = BENCHMARK_META.get(suite, {}).get(pid, {})
                _ss   = _meta.get('search_space', '')
                pid_label = f'{suite.upper()} - {pid}'
                if _ss:
                    pid_label += f'\n{_ss} Search Space'
                if n_var is not None:
                    pid_label += f'  |  {n_var} vars'

                overlay_out  = os.path.join(root, f'{suite}_pid{pid}_attainment_overlay.png')
                subplots_out = os.path.join(root, f'{suite}_pid{pid}_attainment_subplots.png')

                _xlabel, _ylabel = get_obj_labels(suite, pid)

                plot_pareto_snapshots_evoxbench_overlay(
                    methods=methods,
                    n_gen=n_gen,
                    pop_size=pop_size,
                    pf_norm=pf_norm,
                    out_path=overlay_out,
                    results_root=root,
                    checkpoints_gen=args.checkpoints_gen,
                    colours=_EVOX_COLOURS,
                    labels=_EVOX_LABELS,
                    title=(
                        f'{pid_label}  \n  50\u202f% attainment surfaces'
                        f' (gen\u202f{args.checkpoints_gen[-1]},'
                        f' {args.checkpoints_gen[-1] * pop_size}\u202fevals)'
                    ),
                    xlabel=_xlabel,
                    ylabel=_ylabel,
                    font_scale=args.font_scale,
                    axis_limits=args.attainment_limits,
                    attainment_type=args.attainment_type,
                )

                plot_pareto_snapshots_evoxbench_subplots(
                    methods=methods,
                    n_gen=n_gen,
                    pop_size=pop_size,
                    pf_norm=pf_norm,
                    out_path=subplots_out,
                    results_root=root,
                    checkpoints_gen=args.checkpoints_gen,
                    colours=_EVOX_COLOURS,
                    labels=_EVOX_LABELS,
                    title=(
                        f'{pid_label}  \n  50\u202f% attainment surfaces'
                        f' (pop={pop_size}, checkpoints:'
                        f' {", ".join(str(g) for g in args.checkpoints_gen)} gen)'
                    ),
                    xlabel=_xlabel,
                    ylabel=_ylabel,
                    font_scale=args.font_scale,
                    axis_limits=args.attainment_limits,
                    attainment_type=args.attainment_type,
                )

    # ── LaTeX table ───────────────────────────────────────────────────────────
    if args.latex_table:
        table_out = os.path.join(
            suite_root,
            f'B{n_gen * pop_size}_P{pop_size}',
            f'{suite}_convergence_table.tex',
        )
        generate_latex_table(
            suite=suite,
            pids=pids,
            methods=methods,
            pop_size=pop_size,
            n_gen=n_gen,
            out_path=table_out,
            approx_info=all_approx_info,
            root=args.root,
        )

    # ── exploration scatter plot ──────────────────────────────────────────
    if args.exploration:
        for pid in pids:
            plot_exploration(
                suite=suite,
                pid=pid,
                methods=methods,
                pop_size=pop_size,
                n_gen=n_gen,
                n_seeds=args.exploration_seeds,
                font_scale=args.font_scale,
                approx_info=all_approx_info.get(pid),
            )

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EvoXBench post-run analysis: convergence plots + LaTeX table'
    )
    parser.add_argument('--root', type=str, default='results/evoxbench',
                        help='Root directory for results (default: results/evoxbench)')
    parser.add_argument('--suite',    type=str, required=True,
                        choices=['c10mop', 'in1kmop', 'citysegmop'])
    parser.add_argument('--pids',     type=int, nargs='+', default=list(range(1, 10)),
                        help='Problem IDs to analyse (default: 1-9)')
    parser.add_argument('--methods',  type=str, nargs='+', default=_DEFAULT_METHODS,
                        help='Methods to include (default: all 7 benchmark methods)')
    parser.add_argument('--pop_size',       type=int,  default=20)
    parser.add_argument('--n_gen',           type=int,  default=60)
    parser.add_argument('--rebuild-approx', '--rebuild_approx', action='store_true', dest='rebuild_approx',
                        help='Force rebuild of the combined Pareto approximation cache')
    parser.add_argument('--convergence_plot', '--convergence-plot', action='store_true',
                        dest='convergence_plot',
                        help='Generate per-PID HV / IGD+ convergence plots')
    parser.add_argument('--latex_table', '--latex-table', action='store_true',
                        dest='latex_table',
                        help='Generate the LaTeX summary table')
    parser.add_argument('--attainment', action='store_true',
                        help='Generate 50%% attainment surface plots (2-obj PIDs only)')
    parser.add_argument('--checkpoints_gen', '--checkpoints-gen', type=int, nargs='+',
                        default=[15, 30, 45, 60], dest='checkpoints_gen',
                        help='Generation checkpoints for attainment surface plots (default: 15 30 45 60)')
    parser.add_argument('--font_scale', '--font-scale', type=float, default=1.0,
                        dest='font_scale',
                        help='Font size multiplier for all output plots (default: 1.0)')
    parser.add_argument('--exploration', action='store_true',
                        help='Generate evaluation-exploration scatter plot '
                             '(NASBench-101 only; shows all evaluated designs '
                             'heatmap-coloured by density against the full search space)')
    parser.add_argument('--exploration_seeds', '--exploration-seeds', type=int, default=None,
                        dest='exploration_seeds',
                        help='Limit number of seeds used for the exploration plot '
                             '(default: all available seeds)')
    parser.add_argument('--attainment_limits', '--attainment-limits', type=float, nargs=2,
                        default=None, metavar=('LO', 'HI'), dest='attainment_limits',
                        help='x- and y-axis limits for attainment plots, e.g. 0 0.5')
    parser.add_argument('--attainment_type', '--attainment-type', type=str,
                        default='lines', choices=['lines', 'dots'], dest='attainment_type',
                        help='Attainment surface style: lines or dots (default: lines)')
    parser.add_argument('--no_norm', '--no-norm', action='store_true', dest='no_norm',
                        help='Treat stored test_obj_archive as raw (un-normalized) values. '
                             'Empirical normalization bounds are derived from the data '
                             'and used to compute HV/IGD+ in a comparable scale. '
                             'Use together with --root results/evoxbench_no_norm.')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
