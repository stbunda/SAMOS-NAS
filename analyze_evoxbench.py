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
)
from analysis.convergence import (
    build_pareto_approximation,
    recompute_indicator_trajectories,
    recompute_final_indicators,
    recompute_final_indicators_seeds,
    get_or_recompute_trajectories,
    save_approx_cache,
)
from problem.evoxbench.utils import get_benchmark
from problem.evoxbench.benchmark_meta import pid_header

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

def _results_root(suite: str, pid: int, pop_size: int, n_gen: int) -> str:
    return os.path.join(
        'results', 'evoxbench', suite,
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
        root       = _results_root(suite, pid, pop_size, n_gen)
        pid_approx = (approx_info or {}).get(pid)
        pid_seeds, pid_stats = {}, {}
        for m in methods:
            if pid_approx is not None:
                s = recompute_final_indicators_seeds(
                    m, root, pid_approx['ref_point'], pid_approx['pareto_approx']
                )
            else:
                s = _load_final_indicators_seeds(m, root)
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


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args) -> None:
    suite    = args.suite
    pids     = args.pids
    methods  = args.methods
    pop_size = args.pop_size
    n_gen    = args.n_gen

    suite_root = os.path.join('results', 'evoxbench', suite)

    # ── per-PID convergence plots (HV / IGD+) ─────────────────────────────────
    all_approx_info: dict[int, dict] = {}
    if not args.table_only:
        for pid in pids:
            root = _results_root(suite, pid, pop_size, n_gen)

            # Build (or load from cache) the combined Pareto approximation
            approx = build_pareto_approximation(
                suite, pid, methods, pop_size, n_gen,
                force_rebuild=args.rebuild_approx,
            )
            all_approx_info[pid] = approx

            # Read benchmark metadata for the title (n_var, n_obj)
            try:
                bm    = get_benchmark(suite, pid)
                n_var = bm.search_space.n_var
                n_obj = bm.evaluator.n_objs
            except Exception as _bm_err:
                print(f'  [pid{pid}] WARN: could not instantiate benchmark: {_bm_err}')
                n_var, n_obj = None, None

            dim_str = (
                f'  |  {n_var} vars, {n_obj} objs'
                if n_var is not None and n_obj is not None else ''
            )

            # Derive HV ceiling
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
                # Recompute indicators against the shared combined Pareto approximation.
                # get_or_recompute_trajectories serves from the in-memory traj cache
                # on repeated calls (e.g. when re-running with only some PIDs changed).
                trajectories = {}
                for method in methods:
                    traj = get_or_recompute_trajectories(method, n_gen, root, approx)
                    if traj is None:
                        print(f'  [pid{pid}] No data for method={method}, skipping.')
                    trajectories[method] = traj

                # Persist the updated trajectory cache so the next run is instant
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
                        f'{suite.upper()} PID {pid}{dim_str}  '
                        f'(pop={pop_size}, {n_gen} gen = {pop_size * n_gen} evals, '
                        f'mean \u00b1 std over seeds, shared combined PF)'
                    ),
                )
            else:
                # Fall back to stored indicators when no combined PF is available
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
                        f'{suite.upper()} PID {pid}{dim_str}  '
                        f'(pop={pop_size}, {n_gen} gen = {pop_size * n_gen} evals, '
                        f'mean \u00b1 std over seeds)'
                    ),
                )

        # ── across-PIDs grid plot ─────────────────────────────────────────────────
        if len(pids) > 1:
            # One column per PID would be too many; use the grid for grouped methods
            grid_out = os.path.join(suite_root,
                                    f'B{n_gen * pop_size}_P{pop_size}',
                                    f'{suite}_all_pids_grid.png')
            # Merge all roots; plot_results_grid expects a single results_root so
            # we produce one combined plot per PID instead, using the per-PID outs.
            # (A true cross-PID grid would require a custom plotter; save for later.)
            print(f'  [grid] Per-PID convergence plots saved; grid across PIDs not yet implemented.')
    else:
        # Still need to build all_approx_info for the table
        for pid in pids:
            root = _results_root(suite, pid, pop_size, n_gen)
            approx = build_pareto_approximation(
                suite, pid, methods, pop_size, n_gen,
                force_rebuild=args.rebuild_approx,
            )
            all_approx_info[pid] = approx

    # ── LaTeX table ───────────────────────────────────────────────────────────
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
    )

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EvoXBench post-run analysis: convergence plots + LaTeX table'
    )
    parser.add_argument('--suite',    type=str, required=True,
                        choices=['c10mop', 'in1kmop', 'citysegmop'])
    parser.add_argument('--pids',     type=int, nargs='+', default=list(range(1, 10)),
                        help='Problem IDs to analyse (default: 1-9)')
    parser.add_argument('--methods',  type=str, nargs='+', default=_DEFAULT_METHODS,
                        help='Methods to include (default: all 7 benchmark methods)')
    parser.add_argument('--pop_size',       type=int,  default=20)
    parser.add_argument('--n_gen',           type=int,  default=50)
    parser.add_argument('--rebuild-approx', '--rebuild_approx', action='store_true', dest='rebuild_approx',
                        help='Force rebuild of the combined Pareto approximation cache')
    parser.add_argument('--table_only', '--table_only', action='store_true', dest='table_only',
                        help='Skip convergence plots, generate LaTeX table only')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
