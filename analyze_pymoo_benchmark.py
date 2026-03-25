"""analyze_pymoo_benchmark.py — Post-run analysis for WFG benchmark experiments.

Produces per-problem convergence plots (HV / IGD+), Pareto snapshot plots
(50 % attainment surfaces at 10d / 25d / 50d / 100d evaluations), and a
combined LaTeX table for all benchmarks.

Run example (after results are collected from the cluster):
  python analyze_pymoo_benchmark.py \\
      --experiment_name pymoo_benchmark/2_obj \\
      --problems wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9 \\
      --n_obj 2 \\
      --methods random nsga2 samos-xgb-i200-g20 samos-rfr-i200-g20 mosmac \\
      --pop_size 20 \\
      --n_gen 60
"""

import argparse
import os
import pickle

import numpy as np
from pymoo.indicators.hv import HV

from analysis.plotter import (
    plot_results,
    plot_results_grid,
    plot_pareto_snapshots,
    _resolve_style,
    COLOURS,
    LABELS,
    METHOD_GROUPS,
)
from main_pymoo_benchmark import (
    _build_problem,
    _get_pareto_front,
    _default_ref_point,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _results_root(experiment_name: str, problem: str, n_gen: int, pop_size: int) -> str:
    return os.path.join('results', experiment_name, problem,
                        f'B{n_gen * pop_size}_P{pop_size}')


def _load_final_indicators(method: str, results_root: str) -> dict:
    """Return {'hv': (mean, std), 'igd_plus': (mean, std)} from last generation."""
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None
    hvs, igds = [], []
    for pkl_file in sorted(os.listdir(seed_dir)):
        if not pkl_file.endswith('.pkl'):
            continue
        with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
            data = pickle.load(f)
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


# ─── LaTeX table ─────────────────────────────────────────────────────────────

_METHOD_LATEX_LABELS = {
    'random':    'Random',
    'nsga2':     'NSGA-II',
    'samos-rfr': 'SAMOS-RFR',
    'samos-xgb': 'SAMOS-XGB',
    'mosmac':    'MOSMAC',
}


def _method_latex_label(method: str) -> str:
    """Map a method key to a short LaTeX display name."""
    if method in _METHOD_LATEX_LABELS:
        return _METHOD_LATEX_LABELS[method]
    # prefix match
    for key, lbl in _METHOD_LATEX_LABELS.items():
        if method.startswith(key + '-'):
            suffix = method[len(key):]
            return lbl + r'\textsubscript{' + suffix.strip('-') + '}'
    return method


def _fmt(mean: float, std: float, bold: bool) -> str:
    s = f'{mean:.4f}\\,\\textpm\\,{std:.4f}'
    return f'\\textbf{{{s}}}' if bold else s


def generate_latex_table(
    problems: list,
    methods: list,
    n_obj: int,
    experiment_name: str,
    pop_size: int,
    n_gen: int,
    out_path: str,
):
    """Write a combined booktabs LaTeX table for all WFG benchmarks.

    One row-block per benchmark, separated by \\midrule.  Columns:
      Benchmark | m (n_obj) | Method | HV (↑) | IGD+ (↓)

    The best-performing method per benchmark per metric is \\textbf-wrapped.
    """
    n_methods = len(methods)

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        (r'\caption{WFG1\textendash{}9: final HV and '
         r'IGD\textsuperscript{+} at $100d$ evaluations '
         r'(mean\,\textpm\,std over 30 seeds).}'),
        r'\label{tab:wfg_convergence}',
        r'\begin{tabular}{llcrr}',
        r'\toprule',
        (r'Benchmark & $m$ & Method & '
         r'HV\,($\uparrow$) & IGD\textsuperscript{+}\,($\downarrow$) \\'),
        r'\midrule',
    ]

    for prob_idx, problem in enumerate(problems):
        root = _results_root(experiment_name, problem, n_gen, pop_size)

        # collect stats for all methods
        stats = {}
        for method in methods:
            stats[method] = _load_final_indicators(method, root)

        # find best per metric (higher HV = better, lower IGD+ = better)
        valid_hv  = {m: s['hv'][0]       for m, s in stats.items() if s}
        valid_igd = {m: s['igd_plus'][0]  for m, s in stats.items() if s}
        best_hv   = max(valid_hv,  key=valid_hv.get)  if valid_hv  else None
        best_igd  = min(valid_igd, key=valid_igd.get) if valid_igd else None

        bench_label = problem.upper()

        for mi, method in enumerate(methods):
            s = stats.get(method)
            hv_str  = _fmt(*s['hv'],       method == best_hv)  if s else '--'
            igd_str = _fmt(*s['igd_plus'],  method == best_igd) if s else '--'
            mlbl    = _method_latex_label(method)

            if mi == 0:
                bench_cell = (
                    f'\\multirow{{{n_methods}}}{{*}}{{{bench_label}}}'
                )
                obj_cell = (
                    f'\\multirow{{{n_methods}}}{{*}}{{{n_obj}}}'
                )
                lines.append(
                    f'{bench_cell} & {obj_cell} & {mlbl} & {hv_str} & {igd_str} \\\\'
                )
            else:
                lines.append(
                    f' & & {mlbl} & {hv_str} & {igd_str} \\\\'
                )

        if prob_idx < len(problems) - 1:
            lines.append(r'\midrule')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  LaTeX table saved -> {out_path}')


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    all_problems = [f'wfg{i}' for i in range(1, 10)]

    for problem in args.problems:
        root = _results_root(args.experiment_name, problem, args.n_gen, args.pop_size)
        print(f'\n[ANALYZE] {problem}  root={root}')

        # build reference objects
        n_var = (
            args.n_var
            if args.n_var is not None
            else (2 * (args.n_obj - 1) + 10 if problem.startswith('wfg') else None)
        )
        prob_obj  = _build_problem(problem, args.n_obj, n_var)
        pf        = _get_pareto_front(prob_obj, prob_obj.n_obj)
        ref_point = _default_ref_point(problem, prob_obj.n_obj)
        hv_ceiling = float(HV(ref_point=ref_point)(pf))
        print(f'  PF pts={len(pf)}  hv_ceiling={hv_ceiling:.6f}')

        # ── convergence plot ──────────────────────────────────────────────────
        conv_out = os.path.join(root, 'moo_hv_igd.png')
        plot_results(
            methods=args.methods,
            n_gen=args.n_gen,
            pop_size=args.pop_size,
            hv_ceiling=hv_ceiling,
            out_path=conv_out,
            results_root=root,
            title=(
                f'{problem.upper()}  —  HV / IGD+ convergence\n'
                f'(pop={args.pop_size}, {args.n_gen} gens'
                f' = {args.pop_size * args.n_gen} evals, mean \u00b1 std)'
            ),
        )

        # ── extended grid plot (written alongside the standard plot) ──────────
        if args.extended_plot:
            grid_out = os.path.join(root, 'moo_hv_igd_extended.png')
            plot_results_grid(
                methods=args.methods,
                n_gen=args.n_gen,
                pop_size=args.pop_size,
                hv_ceiling=hv_ceiling,
                out_path=grid_out,
                results_root=root,
                title=(
                    f'{problem.upper()}  —  Extended comparison: HV / IGD+\n'
                    f'(pop={args.pop_size}, {args.n_gen} gens'
                    f' = {args.pop_size * args.n_gen} evals, mean \u00b1 std)'
                ),
            )

        # ── Pareto snapshot plot ──────────────────────────────────────────────
        snap_out = os.path.join(root, 'pareto_snapshots.png')
        plot_pareto_snapshots(
            methods=args.methods,
            n_gen=args.n_gen,
            pop_size=args.pop_size,
            n_var=prob_obj.n_var,
            pf=pf,
            out_path=snap_out,
            results_root=root,
            checkpoints_d=args.checkpoints_d,
            title=(
                f'{problem.upper()}  —  50\u202f% attainment surfaces\n'
                f'(n_var={prob_obj.n_var}, checkpoints at '
                f'{", ".join(str(c) + "d" for c in args.checkpoints_d)} evals)'
            ),
        )

    # ── combined LaTeX table ──────────────────────────────────────────────────
    table_out = os.path.join('results', args.experiment_name,
                             'wfg_convergence_table.tex')
    generate_latex_table(
        problems=args.problems,
        methods=args.methods,
        n_obj=args.n_obj,
        experiment_name=args.experiment_name,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        out_path=table_out,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Post-run analysis for WFG pymoo benchmark experiments'
    )
    parser.add_argument('--experiment_name', type=str,
                        default='pymoo_benchmark/2_obj')
    parser.add_argument('--problems', type=str, nargs='+',
                        default=[f'wfg{i}' for i in range(1, 10)])
    parser.add_argument('--n_obj',    type=int, default=2)
    parser.add_argument('--n_var',    type=int, default=None,
                        help='Override n_var (default: 2*(n_obj-1)+10 for WFG)')
    parser.add_argument('--methods',  type=str, nargs='+',
                        default=['random', 'nsga2',
                                 'samos-xgb-i200-g20', 'samos-rfr-i200-g20',
                                 'mosmac'])
    parser.add_argument('--pop_size', type=int, default=20)
    parser.add_argument('--n_gen',    type=int, default=60)
    parser.add_argument('--checkpoints_d', type=int, nargs='+',
                        default=[10, 25, 50, 100],
                        help='Budget checkpoints as multiples of n_var for Pareto snapshots')
    parser.add_argument('--extended_plot', action='store_true',
                        help='Also produce moo_hv_igd_extended.png with group-column layout')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
