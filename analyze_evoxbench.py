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

from analysis.plotter import (
    plot_results,
    plot_results_grid,
    load_indicator_trajectories,
    _resolve_style,
    COLOURS,
    LABELS,
)
from problem.evoxbench.utils import get_benchmark

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
    'random', 'nsga2', 'samos-xgb', 'samos2', 'parego', 'gpsaf-default', 'mosmac',
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


# ─── LaTeX table ──────────────────────────────────────────────────────────────

def _fmt(mean: float, std: float, bold: bool) -> str:
    s = f'{mean:.4f}\\,\\textpm\\,{std:.4f}'
    return f'\\textbf{{{s}}}' if bold else s


def _method_latex_label(method: str) -> str:
    return _EVOX_LABELS.get(method, method)


def generate_latex_table(
    suite: str,
    pids: list,
    methods: list,
    pop_size: int,
    n_gen: int,
    out_path: str,
) -> None:
    """Write a booktabs LaTeX table: rows = methods, columns = PIDs × {HV, IGD+}."""
    # Collect stats
    all_stats: dict[int, dict] = {}
    best_hv:   dict[int, str]  = {}
    best_igd:  dict[int, str]  = {}
    for pid in pids:
        root  = _results_root(suite, pid, pop_size, n_gen)
        stats = {m: _load_final_indicators(m, root) for m in methods}
        all_stats[pid] = stats
        valid_hv  = {m: s['hv'][0]       for m, s in stats.items() if s}
        valid_igd = {m: s['igd_plus'][0]  for m, s in stats.items() if s}
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
         r'IGD\textsuperscript{+} (mean\,\textpm\,std over 30 seeds).}'),
        f'\\label{{tab:{suite}_convergence}}',
        r'\resizebox{\linewidth}{!}{',
        r'\begin{tabular}{l c r r c r r}',
        r'\toprule',
        f'Method & PID & {hv_col} & {igd_col} & PID & {hv_col} & {igd_col} \\\\',
        r'\midrule',
    ]

    n_methods = len(methods)
    for li, l_pid in enumerate(left_pids):
        r_pid = right_pids[li] if li < len(right_pids) else None
        for mi, method in enumerate(methods):
            mlbl = _method_latex_label(method)

            ls = all_stats[l_pid].get(method)
            lhv  = _fmt(*ls['hv'],       method == best_hv[l_pid])       if ls else '--'
            ligd = _fmt(*ls['igd_plus'],  method == best_igd[l_pid])      if ls else '--'

            if r_pid is not None:
                rs   = all_stats[r_pid].get(method)
                rhv  = _fmt(*rs['hv'],       method == best_hv[r_pid])   if rs else '--'
                rigd = _fmt(*rs['igd_plus'],  method == best_igd[r_pid]) if rs else '--'
                right_part = (
                    f'\\multirow{{{n_methods}}}{{*}}{{PID {r_pid}}} & {rhv} & {rigd}'
                    if mi == 0 else f' & {rhv} & {rigd}'
                )
            else:
                right_part = ' &  & '

            bench_cell = f'\\multirow{{{n_methods}}}{{*}}{{PID {l_pid}}}' if mi == 0 else ''
            lines.append(f'{mlbl} & {bench_cell} & {lhv} & {ligd} & {right_part} \\\\')

        if li < len(left_pids) - 1:
            lines.append(r'\midrule')

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
    for pid in pids:
        root    = _results_root(suite, pid, pop_size, n_gen)
        ceiling = _hv_ceiling(suite, pid)
        if ceiling is None:
            # Use the highest observed HV as a proxy ceiling
            best_hv = 0.0
            for method in methods:
                traj = load_indicator_trajectories(method, n_gen, root)
                if traj is not None:
                    best_hv = max(best_hv, float(traj[0][-1]))
            ceiling = best_hv * 1.05 if best_hv > 0 else 1.0
            print(f'  [pid{pid}] No reference PF — using proxy HV ceiling={ceiling:.4f}')
        else:
            print(f'  [pid{pid}] HV ceiling = {ceiling:.6f}')

        conv_out = os.path.join(root, f'{suite}_pid{pid}_hv_igd.png')
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
                f'{suite.upper()} PID {pid}  '
                f'(pop={pop_size}, {n_gen} gen = {pop_size * n_gen} evals, '
                f'mean ± std over seeds)'
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
    parser.add_argument('--pop_size', type=int, default=20)
    parser.add_argument('--n_gen',    type=int, default=50)

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
