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
from scipy.stats import ranksums

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
    build_problem,
    get_pareto_front,
    default_ref_point,
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
            try:
                data = pickle.load(f)
            except Exception as e:
                print(f"  [WARN] Failed to load {pkl_file}: {e}")
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


def _load_final_indicators_seeds(method: str, results_root: str) -> dict:
    """Return {'hv': np.ndarray, 'igd_plus': np.ndarray} of per-seed values."""
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None
    hvs, igds = [], []
    for pkl_file in sorted(os.listdir(seed_dir)):
        if not pkl_file.endswith('.pkl'):
            continue
        with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(f"  [WARN] Failed to load {pkl_file}: {e}")
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


# ─── LaTeX table ─────────────────────────────────────────────────────────────
_WILCOXON_REF   = 'samos-xgb-i200-g20'
_WILCOXON_ALPHA = 0.05

_DEFAULT_METHOD_ORDER = [
    'random', 'nsga2', 'parego', 'mosmac', 'gpsaf-default',
    'samos-rfr-i200-g20', 'samos-xgb-i200-g20', 'samos-ssa-i200-g20',
]
_METHOD_ALIASES = {
    'samos-xgb':  'samos-xgb-i200-g20',
    'samos-rfr':  'samos-rfr-i200-g20',
    'samos-ssa':  'samos-ssa-i200-g20',
    'gpsaf':      'gpsaf-default',
}

_METHOD_LATEX_LABELS = {
    'random':             'Random',
    'nsga2':              'NSGA-II',
    'samos-rfr':          'SAMOS-RFR',
    'samos-xgb':          'SAMOS-XGB',
    'samos-rfr-i200-g20': 'SAMOS-RFR',
    'samos-xgb-i200-g20': 'SAMOS-XGB',
    'samos-ssa-i200-g20': 'SAMOS-SSA',
    'mosmac':             'MOSMAC',
    'parego':             'ParEGO',
    'cobra':              'COBRA',
    'gpsaf-default':      'GPSAF-Default',
    'gpsaf-rfr':          'GPSAF-RFR',
    'gpsaf-xgb':          'GPSAF-XGB',
    'ssa-nsga2-default':  'SSA-Default',
    'ssa-nsga2-rfr':      'SSA-RFR',
    'ssa-nsga2-xgb':      'SSA-XGB',
}


def _method_latex_label(method: str) -> str:
    """Map a method key to a short LaTeX display name."""
    if method in _METHOD_LATEX_LABELS:
        return _METHOD_LATEX_LABELS[method]
    # prefix match (fallback for unlisted variants)
    for key, lbl in _METHOD_LATEX_LABELS.items():
        if method.startswith(key + '-'):
            suffix = method[len(key):]
            return lbl + r'\textsubscript{' + suffix.strip('-') + '}'
    return method


def _wilcoxon_marker(ref_vals, other_vals, higher_is_better: bool) -> str:
    """Wilcoxon rank-sum significance marker vs. the reference method.

    Returns ``$(+)$`` when the method is significantly better than the reference,
    ``$(-)$`` when significantly worse, or ``$(\\approx)$`` when not significant.
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
        return r'$(+)$' if other_med > ref_med else r'$(-)$'
    else:
        return r'$(+)$' if other_med < ref_med else r'$(-)$'


def _fmt(mean: float, std: float, bold: bool, marker: str = '') -> str:
    s = f'{mean:.4f}\\,\\textpm\\,{std:.4f}'
    cell = f'\\textbf{{{s}}}' if bold else s
    return f'{cell}{marker}'


def generate_latex_table(
    problems: list,
    methods: list,
    experiment_name: str,
    pop_size: int,
    n_gen: int,
    out_path: str,
    n_obj: int = 2,
    n_var: int = None,
) -> None:
    """Write a booktabs LaTeX table: rows = methods, columns = problems × {HV, IGD+}.

    Problems are displayed as block headers (``\\makecell``) instead of a
    dedicated Benchmark column.  Wilcoxon rank-sum tests (p<0.05) are computed
    per problem using ``_WILCOXON_REF`` as the reference method.  Each cell of
    a non-reference method is annotated with ``$(+)$`` when significantly
    better, ``$(-)$`` when significantly worse, or ``$(\\approx)$`` when not
    significant.

    Methods are always displayed in the canonical order defined by
    ``_DEFAULT_METHOD_ORDER``; any methods not in that list are appended in the
    order they are received.
    """
    # Resolve n_var default (WFG convention: 2*(n_obj-1)+10)
    _n_var = n_var if n_var is not None else 2 * (n_obj - 1) + 10

    # Enforce canonical method order
    order   = {m: i for i, m in enumerate(_DEFAULT_METHOD_ORDER)}
    methods = sorted(methods, key=lambda m: order.get(m, len(_DEFAULT_METHOD_ORDER)))

    # Collect per-seed data and (mean, std) stats for every problem
    all_seeds: dict = {}
    all_stats: dict = {}
    best_hv:   dict = {}
    best_igd:  dict = {}
    for problem in problems:
        root       = _results_root(experiment_name, problem, n_gen, pop_size)
        prob_seeds, prob_stats = {}, {}
        for m in methods:
            s = _load_final_indicators_seeds(m, root)
            prob_seeds[m] = s
            prob_stats[m] = {
                'hv':       (float(np.mean(s['hv'])),       float(np.std(s['hv']))),
                'igd_plus': (float(np.mean(s['igd_plus'])), float(np.std(s['igd_plus']))),
            } if s is not None else None
        all_seeds[problem] = prob_seeds
        all_stats[problem] = prob_stats
        valid_hv  = {m: v['hv'][0]      for m, v in prob_stats.items() if v}
        valid_igd = {m: v['igd_plus'][0] for m, v in prob_stats.items() if v}
        best_hv[problem]  = max(valid_hv,  key=valid_hv.get)  if valid_hv  else None
        best_igd[problem] = min(valid_igd, key=valid_igd.get) if valid_igd else None

    hv_col  = r'HV\,($\uparrow$)'
    igd_col = r'IGD\textsuperscript{+}\,($\downarrow$)'

    # Two-column layout: split problems left/right
    mid           = (len(problems) + 1) // 2
    left_problems = problems[:mid]
    right_problems = problems[mid:]

    def _prob_header(prob: str) -> str:
        label = prob.upper()
        return (
            r'\makecell[c]{\textbf{' + label + r'}'
            + r' \\ (' + str(n_obj) + r'\,obj, ' + str(_n_var) + r'\,vars)}'
        )

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        (r'\caption{WFG1\textendash{}9 ($m=' + str(n_obj) + r'$): final HV and '
         r'IGD\textsuperscript{+} (mean\,\textpm\,std). '
         r'\textbf{Bold}: best per problem. '
         r'Wilcoxon rank-sum vs.\ SAMOS\,(XGBoost) ($p{<}0.05$): '
         r'$(+)$\,better, $(-)$\,worse, $(\approx)$\,no significant difference.}'),
        r'\label{tab:wfg_convergence}',
        r'\resizebox{\linewidth}{!}{',
        r'\begin{tabular}{l r r r r}',
        r'\toprule',
        f'Method & {hv_col} & {igd_col} & {hv_col} & {igd_col} \\\\',
    ]

    for li, l_prob in enumerate(left_problems):
        r_prob = right_problems[li] if li < len(right_problems) else None

        # Block header row
        l_hdr = _prob_header(l_prob)
        lines.append(r'\midrule')
        if r_prob is not None:
            r_hdr = _prob_header(r_prob)
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
        lref      = all_seeds[l_prob].get(_WILCOXON_REF)
        l_ref_hv  = lref['hv']       if lref else None
        l_ref_igd = lref['igd_plus'] if lref else None
        if r_prob is not None:
            rref      = all_seeds[r_prob].get(_WILCOXON_REF)
            r_ref_hv  = rref['hv']       if rref else None
            r_ref_igd = rref['igd_plus'] if rref else None

        for method in methods:
            mlbl   = _method_latex_label(method)
            is_ref = (method == _WILCOXON_REF)

            ls = all_stats[l_prob].get(method)
            if ls and not is_ref:
                lseed  = all_seeds[l_prob].get(method)
                lhv_m  = _wilcoxon_marker(l_ref_hv,  lseed['hv']       if lseed else None, True)
                ligd_m = _wilcoxon_marker(l_ref_igd, lseed['igd_plus'] if lseed else None, False)
            else:
                lhv_m = ligd_m = ''
            lhv  = _fmt(*ls['hv'],       method == best_hv[l_prob],  lhv_m)  if ls else '--'
            ligd = _fmt(*ls['igd_plus'],  method == best_igd[l_prob], ligd_m) if ls else '--'

            if r_prob is not None:
                rs = all_stats[r_prob].get(method)
                if rs and not is_ref:
                    rseed  = all_seeds[r_prob].get(method)
                    rhv_m  = _wilcoxon_marker(r_ref_hv,  rseed['hv']       if rseed else None, True)
                    rigd_m = _wilcoxon_marker(r_ref_igd, rseed['igd_plus'] if rseed else None, False)
                else:
                    rhv_m = rigd_m = ''
                rhv  = _fmt(*rs['hv'],       method == best_hv[r_prob],  rhv_m)  if rs else '--'
                rigd = _fmt(*rs['igd_plus'],  method == best_igd[r_prob], rigd_m) if rs else '--'
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

def main(args):
    # Expand short aliases to canonical directory names
    args.methods = [_METHOD_ALIASES.get(m, m) for m in args.methods]

    all_problems = [f'wfg{i}' for i in range(1, 10)]

    # ── combined LaTeX table ──────────────────────────────────────────────────
    table_out = os.path.join('results', args.experiment_name,
                             'wfg_convergence_table_full.tex')
    generate_latex_table(
        problems=args.problems,
        methods=args.methods,
        experiment_name=args.experiment_name,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        out_path=table_out,
        n_obj=args.n_obj,
        n_var=args.n_var,
    )

    if args.table_only:
        return

    for problem in args.problems:
        root = _results_root(args.experiment_name, problem, args.n_gen, args.pop_size)
        print(f'\n[ANALYZE] {problem}  root={root}')

        # build reference objects
        n_var = (
            args.n_var
            if args.n_var is not None
            else (2 * (args.n_obj - 1) + 10 if problem.startswith('wfg') else None)
        )
        prob_obj  = build_problem(problem, args.n_obj, n_var)
        pf        = get_pareto_front(prob_obj, prob_obj.n_obj)
        ref_point = default_ref_point(problem, prob_obj.n_obj)
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

        # # ── Pareto snapshot plot ──────────────────────────────────────────────
        # snap_out = os.path.join(root, 'pareto_snapshots.png')
        # plot_pareto_snapshots(
        #     methods=args.methods,
        #     n_gen=args.n_gen,
        #     pop_size=args.pop_size,
        #     n_var=prob_obj.n_var,
        #     pf=pf,
        #     out_path=snap_out,
        #     results_root=root,
        #     checkpoints_d=args.checkpoints_d,
        #     title=(
        #         f'{problem.upper()}  —  50\u202f% attainment surfaces\n'
        #         f'(n_var={prob_obj.n_var}, checkpoints at '
        #         f'{", ".join(str(c) + "d" for c in args.checkpoints_d)} evals)'
        #     ),
        # )



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
    parser.add_argument('--table_only', '--table-only', action='store_true', dest='table_only',
                        help='Skip convergence plots, generate LaTeX table only')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
