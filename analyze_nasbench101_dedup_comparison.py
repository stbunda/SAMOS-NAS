"""analyze_nasbench101_dedup_comparison.py

Produces a two-section booktabs LaTeX table comparing two duplicate-elimination
strategies on NASBench-101 (C10/MOP 1 and 2):

  Section A – Standard dedup (IntegerVectorDuplicateElimination, evoxbench run)
              results/evoxbench/c10mop/pid{pid}/B{budget}_P{pop}/

  Section B – Arch-str dedup (EvoxNASBench101DuplicateElimination)
              results/nasbench101/c10mop/pid{pid}/B{budget}_P{pop}/

Usage:
  python analyze_nasbench101_dedup_comparison.py
  python analyze_nasbench101_dedup_comparison.py --pop_size 20 --n_gen 60 \\
      --methods random nsga2 samos-xgb gpsaf-default parego mosmac \\
      --out results/nasbench101/dedup_comparison_table.tex
"""

import argparse
import os

import numpy as np

# Re-use helpers from the evoxbench analyser (same loading / cell-formatting
# logic, avoids duplication).
from analyze_evoxbench import (
    _WILCOXON_REF,
    _DEFAULT_METHODS,
    _load_final_indicators_seeds,
    _wilcoxon_marker,
    _fmt,
    _method_latex_label,
)


# ─── path helpers ─────────────────────────────────────────────────────────────

def _evoxbench_root(pid: int, pop_size: int, n_gen: int) -> str:
    """Path written by main_evoxbench.py (standard dedup)."""
    return os.path.join(
        'results', 'evoxbench', 'c10mop',
        f'pid{pid}', f'B{n_gen * pop_size}_P{pop_size}',
    )


def _nb101_root(pid: int, pop_size: int, n_gen: int) -> str:
    """Path written by main_nasbench101.py (arch-str dedup)."""
    return os.path.join(
        'results', 'nasbench101', 'c10mop',
        f'pid{pid}', f'B{n_gen * pop_size}_P{pop_size}',
    )


# ─── data collection ──────────────────────────────────────────────────────────

# Maximum plausible HV for a 2-objective problem normalised to [0,1] with
# reference point [1.05, 1.05] is 1.05**2 ≈ 1.1025.  Seeds whose stored HV
# exceeds this threshold were recorded before normalisation was applied and
# must be excluded so they do not corrupt the mean.
_MAX_VALID_HV = 2.0


_N_SEEDS = 3   # only use the first N valid seeds for comparisons


def _filter_valid(s: dict | None) -> dict | None:
    """Drop per-seed entries whose HV is clearly out of the normalised range,
    then keep only the first _N_SEEDS valid seeds."""
    if s is None:
        return None
    hv  = np.asarray(s['hv'],       dtype=float)
    igd = np.asarray(s['igd_plus'],  dtype=float)
    mask = np.isfinite(hv) & (hv <= _MAX_VALID_HV)
    if not mask.any():
        return None
    return {'hv': hv[mask][:_N_SEEDS], 'igd_plus': igd[mask][:_N_SEEDS]}


def _collect(methods: list, pids: list, root_fn) -> tuple[dict, dict, dict, dict]:
    """Load per-seed indicators for all (pid, method) combinations.

    Returns:
        all_seeds  – {pid: {method: {'hv': np.ndarray, 'igd_plus': np.ndarray}}}
        all_stats  – {pid: {method: {'hv': (mean, std), 'igd_plus': (mean, std)}}}
        best_hv    – {pid: method_name_with_best_mean_hv}
        best_igd   – {pid: method_name_with_best_mean_igd}
    """
    all_seeds: dict = {}
    all_stats: dict = {}
    best_hv:   dict = {}
    best_igd:  dict = {}

    for pid in pids:
        root = root_fn(pid)
        pid_seeds, pid_stats = {}, {}

        for m in methods:
            s = _filter_valid(_load_final_indicators_seeds(m, root))
            pid_seeds[m] = s
            if s is not None:
                pid_stats[m] = {
                    'hv':       (float(np.mean(s['hv'])),       float(np.std(s['hv']))),
                    'igd_plus': (float(np.mean(s['igd_plus'])), float(np.std(s['igd_plus']))),
                }
            else:
                pid_stats[m] = None

        all_seeds[pid] = pid_seeds
        all_stats[pid] = pid_stats

        valid_hv  = {m: v['hv'][0]       for m, v in pid_stats.items() if v}
        valid_igd = {m: v['igd_plus'][0]  for m, v in pid_stats.items() if v}
        best_hv[pid]  = max(valid_hv,  key=valid_hv.get)  if valid_hv  else None
        best_igd[pid] = min(valid_igd, key=valid_igd.get) if valid_igd else None

    return all_seeds, all_stats, best_hv, best_igd


# ─── table generation ─────────────────────────────────────────────────────────

def _section_rows(
    methods: list,
    pids: list,
    all_seeds: dict,
    all_stats: dict,
    best_hv: dict,
    best_igd: dict,
) -> list[str]:
    """Return the LaTeX row strings for one table section."""
    lines = []
    pid1, pid2 = pids[0], pids[1]

    ref1 = all_seeds[pid1].get(_WILCOXON_REF)
    ref1_hv  = ref1['hv']       if ref1 else None
    ref1_igd = ref1['igd_plus'] if ref1 else None

    ref2 = all_seeds[pid2].get(_WILCOXON_REF)
    ref2_hv  = ref2['hv']       if ref2 else None
    ref2_igd = ref2['igd_plus'] if ref2 else None

    for method in methods:
        mlbl   = _method_latex_label(method)
        is_ref = (method == _WILCOXON_REF)

        # ── PID 1 ──
        s1 = all_stats[pid1].get(method)
        if s1 and not is_ref:
            sd1  = all_seeds[pid1].get(method)
            hv1m = _wilcoxon_marker(ref1_hv,  sd1['hv']       if sd1 else None, True)
            ig1m = _wilcoxon_marker(ref1_igd, sd1['igd_plus'] if sd1 else None, False)
        else:
            hv1m = ig1m = ''
        cell_hv1  = _fmt(*s1['hv'],       method == best_hv[pid1],  hv1m) if s1 else '--'
        cell_igd1 = _fmt(*s1['igd_plus'], method == best_igd[pid1], ig1m) if s1 else '--'

        # ── PID 2 ──
        s2 = all_stats[pid2].get(method)
        if s2 and not is_ref:
            sd2  = all_seeds[pid2].get(method)
            hv2m = _wilcoxon_marker(ref2_hv,  sd2['hv']       if sd2 else None, True)
            ig2m = _wilcoxon_marker(ref2_igd, sd2['igd_plus'] if sd2 else None, False)
        else:
            hv2m = ig2m = ''
        cell_hv2  = _fmt(*s2['hv'],       method == best_hv[pid2],  hv2m) if s2 else '--'
        cell_igd2 = _fmt(*s2['igd_plus'], method == best_igd[pid2], ig2m) if s2 else '--'

        lines.append(f'{mlbl} & {cell_hv1} & {cell_igd1} & {cell_hv2} & {cell_igd2} \\\\')

    return lines


def generate_comparison_table(
    pids: list,
    methods: list,
    pop_size: int,
    n_gen: int,
    out_path: str,
) -> None:
    """Write the two-section dedup comparison LaTeX table to *out_path*."""

    # Enforce canonical method order
    order   = {m: i for i, m in enumerate(_DEFAULT_METHODS)}
    methods = sorted(methods, key=lambda m: order.get(m, len(_DEFAULT_METHODS)))

    pid1, pid2 = pids[0], pids[1]

    # ── collect data ──────────────────────────────────────────────────────────
    seeds_a, stats_a, besthv_a, bestigd_a = _collect(
        methods, pids,
        lambda pid: _evoxbench_root(pid, pop_size, n_gen),
    )
    seeds_b, stats_b, besthv_b, bestigd_b = _collect(
        methods, pids,
        lambda pid: _nb101_root(pid, pop_size, n_gen),
    )

    hv_col  = r'HV\,($\uparrow$)'
    igd_col = r'IGD\textsuperscript{+}\,($\downarrow$)'

    # ── assemble LaTeX ────────────────────────────────────────────────────────
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        (r'\caption{NASBench-101 C10/MOP PID\,1 and 2: effect of duplicate elimination.'
         r' Final HV and IGD\textsuperscript{+} (mean\,\textpm\,std over seeds).'
         r' \textbf{Bold}: best result per column within each section.'
         r' Wilcoxon rank-sum vs.\ SAMOS\,(XGBoost) ($p{<}0.05$):'
         r' $(+)$\,better, $(-)$\,worse, $(\approx)$\,no significant difference.}'),
        r'\label{tab:nb101_dedup_comparison}',
        r'\resizebox{\linewidth}{!}{',
        r'\begin{tabular}{l r r r r}',
        r'\toprule',
        (f' & \\multicolumn{{2}}{{c}}{{C10/MOP~{pid1}}}'
         f' & \\multicolumn{{2}}{{c}}{{C10/MOP~{pid2}}} \\\\'),
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        f'Method & {hv_col} & {igd_col} & {hv_col} & {igd_col} \\\\',
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{(a)~Standard duplicate elimination (EvoXBench integer-vector)}} \\',
        r'\midrule',
    ]

    lines += _section_rows(methods, pids, seeds_a, stats_a, besthv_a, bestigd_a)

    lines += [
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{(b)~Architecture-string duplicate elimination (NASBench-101 canonical)}} \\',
        r'\midrule',
    ]

    lines += _section_rows(methods, pids, seeds_b, stats_b, besthv_b, bestigd_b)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'}',
        r'\end{table}',
    ]

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'LaTeX table saved -> {out_path}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pids',     type=int,   nargs='+', default=[1, 2],
                   help='C10/MOP pids to include (default: 1 2)')
    p.add_argument('--methods',  type=str,   nargs='+',
                   default=['random', 'nsga2', 'samos-xgb',
                             'gpsaf-default', 'parego', 'mosmac'],
                   help='Methods to include in the table')
    p.add_argument('--pop_size', type=int,   default=20)
    p.add_argument('--n_gen',    type=int,   default=60)
    p.add_argument('--out',      type=str,
                   default=os.path.join('results', 'nasbench101',
                                        'dedup_comparison_table.tex'),
                   help='Output .tex file path')
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    if len(args.pids) < 2:
        raise ValueError('--pids must specify at least two PIDs (e.g. --pids 1 2)')
    generate_comparison_table(
        pids=args.pids[:2],   # table is fixed at two pids (left / right columns)
        methods=args.methods,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        out_path=args.out,
    )


if __name__ == '__main__':
    main(_parse_args())
