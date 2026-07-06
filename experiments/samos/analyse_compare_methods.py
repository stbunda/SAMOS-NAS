"""experiments/samos/analyse_compare_methods.py --- Ablation-study analysis
driver for experiments/samos/compare_methods.py.

Consumes the per-seed pickles written by ``compare_methods.py``
(``results/samos/compare/{suite}/pid{pid}/B{budget}_P{pop}/{method}/seed_*.pkl``,
each a ``results.algorithm.callback.data`` dict from ``EvoxBenchCallback`` --
same schema as the existing obj_ga / obj_ga_pred analysis pipeline) and
produces, per (suite, pid):

1. A shared Pareto approximation across all 6 methods (random, samos-xgb,
   samos2-baseline, samos2-encoding, samos2-isomorphism, samos2-uncertainty),
   cached to ``{pid_root}/pareto_approx.pkl`` (via analysis.convergence).
2. HV / IGD+ convergence plots (mean +/- std over seeds, all generations).
3. A 50% attainment-surface overlay for 2-objective problems.
4. Per-seed HV / IGD+ matrices, feeding a combined LaTeX table (via
   analysis.latex_table_generator.generate_obj_ga_table) with Holm-corrected
   Wilcoxon significance markers against --baseline_method (default:
   samos2-baseline) -- this table IS the ablation study: every
   samos2-encoding / -isomorphism / -uncertainty cell is marked '*' where it
   differs significantly from the samos2-baseline cell in the same row.
5. A console summary of each contribution's final-checkpoint HV delta vs.
   samos2-baseline, per (suite, pid).

samos2-encoding is automatically dropped for search spaces with no
categorical-op structure (DARTS, ResNet-50D, Transformer, MobileNetV3),
mirroring compare_methods.py's own skip logic.

Examples
--------
  python experiments/samos/analyse_compare_methods.py
  python experiments/samos/analyse_compare_methods.py --suites c10mop --pids 5 8
  python experiments/samos/analyse_compare_methods.py --skip_igd --force
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from analysis import convergence
from analysis import plotter
from analysis.latex_table_generator import _holm_correct, _SCIPY_AVAILABLE, generate_obj_ga_table
from problem.evoxbench.benchmark_meta import BENCHMARK_META, get_op_var_group

if _SCIPY_AVAILABLE:
    from scipy.stats import ranksums

METHODS = [
    'random', 'samos-xgb',
    'samos2-baseline', 'samos2-rfr', 'samos2-xgb10',
    'samos2-encoding', 'samos2-isomorphism',
    'samos2-uncertainty', 'samos2-uncertainty-lin_annealing',
    'samos2-uncertainty-obj', 'samos2-xgb10-uncertainty',
]

# (contribution, matched control) pairs for the ablation summary. Each
# uncertainty variant is compared against the control sharing its surrogate
# family, so surrogate-swap and selection-policy effects are not conflated
# (samos2-uncertainty vs samos2-baseline would differ in BOTH surrogate
# family and selection policy).
PAIRS = [
    ('samos2-rfr',                       'samos2-baseline'),
    ('samos2-xgb10',                     'samos2-baseline'),
    ('samos2-encoding',                  'samos2-baseline'),
    ('samos2-isomorphism',               'samos2-baseline'),
    ('samos2-uncertainty',               'samos2-rfr'),
    ('samos2-uncertainty-lin_annealing', 'samos2-rfr'),
    ('samos2-uncertainty-obj',           'samos2-rfr'),
    ('samos2-xgb10-uncertainty',         'samos2-xgb10'),
]

LABELS = {
    'random':                           'Random',
    'samos-xgb':                        'SAMOS (XGBoost, all-surrogate)',
    'samos2-baseline':                  'SAMOS2 (baseline, XGB)',
    'samos2-rfr':                       'SAMOS2 (RFR control)',
    'samos2-xgb10':                     'SAMOS2 (XGB x10 ensemble)',
    'samos2-encoding':                  'SAMOS2 + encoding (C1)',
    'samos2-isomorphism':               'SAMOS2 + isomorphism (C2)',
    'samos2-uncertainty':               'SAMOS2 + uncertainty (C3, RFR)',
    'samos2-uncertainty-lin_annealing': 'SAMOS2 + uncert. ($\\kappa$ anneal)',
    'samos2-uncertainty-obj':           'SAMOS2 + uncert. ($\\sigma$ objective)',
    'samos2-xgb10-uncertainty':         'SAMOS2 + uncert. (XGB x10)',
}
COLOURS = {
    'random':                           '#4e79a7',
    'samos-xgb':                        '#b07aa1',
    'samos2-baseline':                  '#8b0000',
    'samos2-rfr':                       '#17becf',
    'samos2-xgb10':                     '#8c564b',
    'samos2-encoding':                  '#e15759',
    'samos2-isomorphism':               '#f28e2b',
    'samos2-uncertainty':               '#59a14f',
    'samos2-uncertainty-lin_annealing': '#76b7b2',
    'samos2-uncertainty-obj':           '#bcbd22',
    'samos2-xgb10-uncertainty':         '#e377c2',
}


# --------- path / method helpers ------------------------------------------------------------------------------------------------------

def _pid_root(results_root: str, suite: str, pid: int, budget_folder: str) -> str:
    return os.path.join(results_root, suite, f'pid{pid}', budget_folder)


def _methods_for_pid(methods: list, search_space: str) -> list:
    """Drop samos2-encoding on search spaces with no categorical-op structure
    -- mirrors compare_methods.py's own get_op_var_group() skip."""
    if get_op_var_group(search_space) is None:
        return [m for m in methods if m != 'samos2-encoding']
    return list(methods)


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


# --------- per-seed loading (no experiment sub-dir -- compare_methods.py --------------
# writes seed_*.pkl directly under {method}/) -------------------------------------------

def _seed_matrix(pid_root: str, method: str, n_gen: int,
                  ref_point: np.ndarray, pareto_approx: np.ndarray):
    """Per-seed (n_seeds, n_gen) HV / IGD+ matrices, recomputed against the
    shared Pareto approximation. Returns (None, None) if no data."""
    from pymoo.indicators.igd_plus import IGDPlus

    seed_dir = os.path.join(pid_root, method)
    if not os.path.isdir(seed_dir):
        return None, None

    skip_igd = os.environ.get('SAMOS_SKIP_IGD') == '1'
    igd_ind = None if skip_igd else IGDPlus(pareto_approx)

    hv_rows, igd_rows = [], []
    for fname in sorted(os.listdir(seed_dir)):
        if not fname.endswith('.pkl'):
            continue
        try:
            with open(os.path.join(seed_dir, fname), 'rb') as fh:
                data = pickle.load(fh)
        except Exception as exc:
            print(f'  [analyse] WARN: failed to load {seed_dir}/{fname}: {exc}')
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
            F = np.asarray(F, dtype=float)
            F = np.where(np.isfinite(F), F, 1.0)
            hv_series.append(convergence.fast_hv(F, ref_point))
            igd_series.append(np.nan if skip_igd else float(igd_ind(F)))
        hv_rows.append(_fit_len(hv_series, n_gen))
        igd_rows.append(_fit_len(igd_series, n_gen))

    if not hv_rows:
        return None, None
    return np.asarray(hv_rows, dtype=float), np.asarray(igd_rows, dtype=float)


def _final_archives_per_method(pid_root: str, methods: list) -> dict:
    """{method: list[(n_i, 2) ndarray]} from the final-gen test_obj_archive of
    every seed (first two objectives only) -- for attainment-surface plots."""
    out = {}
    for method in methods:
        seed_dir = os.path.join(pid_root, method)
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
            F = np.asarray(archive[-1], dtype=float)
            if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] < 2:
                continue
            finite = np.isfinite(F[:, :2]).all(axis=1)
            F = F[finite][:, :2]
            if len(F) > 0:
                fronts.append(F)
        if fronts:
            out[method] = fronts
    return out


# --------- per-(suite, pid) analysis --------------------------------------------------------------------------------------------------

def analyse_pid(suite: str, pid: int, methods: list, args, output_dir: str):
    meta = BENCHMARK_META.get(suite, {}).get(pid)
    if meta is None:
        print(f'  [analyse] {suite}/pid{pid} not in BENCHMARK_META; skipping.')
        return None

    search_space = meta['search_space']
    n_obj        = meta['n_obj']
    key          = f'{suite}_pid{pid}'
    label        = meta.get('label', key)
    methods_here = _methods_for_pid(methods, search_space)

    budget_folder = f'B{args.n_gen * args.pop_size}_P{args.pop_size}'
    pid_root = _pid_root(args.results_root, suite, pid, budget_folder)

    approx = convergence.build_pareto_approximation(
        suite, pid, methods_here, pop_size=args.pop_size, n_gen=args.n_gen,
        force_rebuild=args.force, results_root=pid_root, experiment_subdir='')
    if approx is None:
        print(f'  [analyse] no data for {key}; skipping.')
        return None

    traj, table_data = {}, {}
    for method in methods_here:
        t = convergence.recompute_indicator_trajectories(
            method, args.n_gen, pid_root, approx['ref_point'], approx['pareto_approx'],
            experiment_subdir='')
        if t is not None:
            traj[method] = t
        hv_mat, igd_mat = _seed_matrix(
            pid_root, method, args.n_gen, approx['ref_point'], approx['pareto_approx'])
        if hv_mat is not None:
            table_data[method] = {'hv': hv_mat, 'igd_plus': igd_mat}

    if traj:
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        out_hv = os.path.join(plot_dir, f'{key}_hv.png')
        plotter.plot_convergence_obj_ga(
            traj, methods_here, LABELS, label, 'hv', out_hv, colours=COLOURS)

        if not args.skip_igd:
            out_igd = os.path.join(plot_dir, f'{key}_igd_plus.png')
            plotter.plot_convergence_obj_ga(
                traj, methods_here, LABELS, label, 'igd_plus', out_igd, colours=COLOURS)

        if n_obj == 2:
            archives = _final_archives_per_method(pid_root, methods_here)
            if archives:
                out_att = os.path.join(plot_dir, f'{key}_attainment.png')
                plotter.plot_attainment_surface(
                    archives, LABELS, label, out_att, colours=COLOURS)
    else:
        print(f'  [analyse] no trajectories for {key} (methods may lack full data).')

    if not table_data:
        return None
    return {'key': key, 'label': label, 'methods': methods_here, 'table_data': table_data}


# --------- ablation-delta console summary ---------------------------------------------------------------------------------------------

def _holm_significant(base_samples: np.ndarray, candidates: dict, alpha: float = 0.05) -> dict:
    """Holm-corrected two-sided Wilcoxon rank-sum test of each candidate's
    samples against base_samples. Same test + correction as
    generate_obj_ga_table's per-cell significance markers, so the console
    summary and the LaTeX table never disagree.

    candidates : {name: 1-D array of per-seed values}
    Returns {name: bool} -- True where the candidate differs significantly
    from the baseline. Names with too few finite samples are False.
    """
    result = {name: False for name in candidates}
    if not _SCIPY_AVAILABLE:
        return result
    base_samples = base_samples[np.isfinite(base_samples)]
    if len(base_samples) < 3:
        return result
    pvals, order = [], []
    for name, samp in candidates.items():
        samp = np.asarray(samp, dtype=float)
        samp = samp[np.isfinite(samp)]
        if len(samp) < 3:
            continue
        try:
            _, p = ranksums(base_samples, samp)
        except Exception:
            p = 1.0
        pvals.append(float(p))
        order.append(name)
    if not pvals:
        return result
    reject = _holm_correct(pvals, alpha)
    for name, r in zip(order, reject):
        result[name] = r
    return result


def print_ablation_summary(sections: list, baseline_method: str):
    """Final-generation HV delta of each contribution vs. its *matched
    control* (PAIRS), per (suite, pid), with Holm-corrected Wilcoxon
    significance markers ('*') -- same test as the LaTeX table, so the two
    never need to be cross-checked by hand.

    The Holm family is (problem x control): all contributions sharing a
    control on one problem are corrected together, mirroring how the LaTeX
    table corrects all methods against its single baseline per row.
    *baseline_method* is unused here (PAIRS defines each contribution's
    control) but kept for CLI symmetry with the LaTeX table.
    """
    controls = []   # ordered unique controls
    for _, ctrl in PAIRS:
        if ctrl not in controls:
            controls.append(ctrl)

    print('\n' + '=' * 88)
    print('Ablation summary -- final-generation HV, contribution vs. matched control')
    print('(* = significant, Holm-corrected Wilcoxon rank-sum, alpha=0.05, '
          'family = problem x control)')
    print('=' * 88)
    for sec in sections:
        pdata = sec['table_data']
        print(f'{sec["label"]:>28s} :')
        for ctrl in controls:
            contribs = [c for c, ct in PAIRS if ct == ctrl]
            base = pdata.get(ctrl, {}).get('hv')
            candidates = {
                c: pdata[c]['hv'][:, -1] for c in contribs
                if pdata.get(c, {}).get('hv') is not None
            }
            if not candidates:
                continue
            if base is None:
                print(f'{"":>28s}   vs {ctrl}: no control data '
                      f'({", ".join(candidates)} present but unmatched)')
                continue
            base_mean = float(np.nanmean(base[:, -1]))
            sig = _holm_significant(base[:, -1], candidates)
            print(f'{"":>28s}   vs {ctrl} = {base_mean:.4f}')
            for contrib, samp in candidates.items():
                mean = float(np.nanmean(samp))
                delta = mean - base_mean
                sign = '+' if delta >= 0 else ''
                marker = '*' if sig.get(contrib) else ' '
                print(f'{"":>28s}     {contrib:<34s}: {mean:.4f}  '
                      f'(delta {sign}{delta:.4f}){marker}')
    print('=' * 88)


# --------- main -------------------------------------------------------------------------------------------------------------------------

def main(args):
    if args.skip_igd:
        os.environ['SAMOS_SKIP_IGD'] = '1'
        print('[analyse] SKIP_IGD enabled: HV only (no IGD+ compute/tables/plots).')

    os.makedirs(args.output_dir, exist_ok=True)

    sections = []
    for suite in args.suites:
        pids = args.pids if args.pids is not None else sorted(BENCHMARK_META.get(suite, {}))
        for pid in pids:
            print(f'\n[analyse] {suite}/pid{pid}')
            sec = analyse_pid(suite, pid, args.methods, args, args.output_dir)
            if sec is not None:
                sections.append(sec)

    if not sections:
        print('[analyse] No data found anywhere; nothing to report.')
        return 0

    print_ablation_summary(sections, args.baseline_method)

    table_dir = os.path.join(args.output_dir, 'tables')
    data_dict = {sec['key']: sec['table_data'] for sec in sections}
    problems = [sec['key'] for sec in sections]
    problem_labels = {sec['key']: sec['label'] for sec in sections}

    metrics = ('hv',) if args.skip_igd else ('hv', 'igd_plus')
    for metric in metrics:
        out_tex = os.path.join(table_dir, f'samos2_ablation_{metric}_table.tex')
        generate_obj_ga_table(
            data_dict, args.methods, LABELS, problems, problem_labels,
            metric, args.checkpoints, out_tex, baseline_method=args.baseline_method)

    print('\n[analyse] Done.')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suites', nargs='+', default=['c10mop', 'in1kmop'], choices=['c10mop', 'in1kmop'])
    p.add_argument('--pids', type=int, nargs='+', default=None,
                    help='Pid subset per suite (default: all pids in BENCHMARK_META).')
    p.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    p.add_argument('--pop_size', type=int, default=20)
    p.add_argument('--n_gen', type=int, default=60)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[15, 30, 45, 60],
                    help='1-based generation checkpoints for the LaTeX table.')
    p.add_argument('--results_root', default='results/samos/compare',
                    help='Root written by compare_methods.py.')
    p.add_argument('--output_dir', default='results/samos/analysis')
    p.add_argument('--baseline_method', default='samos2-baseline', choices=METHODS,
                    help='Reference method for Wilcoxon significance markers and '
                         'the ablation-delta summary.')
    p.add_argument('--force', action='store_true',
                    help='Rebuild the Pareto approximation cache.')
    p.add_argument('--skip_igd', action='store_true',
                    help='Skip IGD+ (pymoo) computation, tables and plots -- HV only.')
    sys.exit(main(p.parse_args()))
