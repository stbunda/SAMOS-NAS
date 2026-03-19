"""
Analysis-only script for NASBench-101 baseline results.

Loads existing result files from results/nasbench101_baseline/ and
generates HV/IGD+ plots, exploration coverage plots, and a LaTeX table.
No experiments are run.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

from problem.nasbench101_utils import MIN_PARAMS, MAX_PARAMS
from analysis.plotter import plot_results, plot_exploration_coverage
from analysis.latex_table_generator import generate_latex_table_nasbench101

RESULTS_ROOT = 'results/nasbench101_baseline'
DATA_FILE    = 'problem/data/data_nasbench101.pkl'


def _load_bench_db() -> dict:
    with open(DATA_FILE, 'rb') as f:
        return pickle.load(f)


def _load_test_pareto_ref() -> np.ndarray:
    cache = 'results/nasbench101_cgp_p_val_acc_12_r_n_params/G150_I20_C200_D20/cache/nasbench101_test_acc_pareto_v1.pkl'
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            c = pickle.load(f)
        return np.array(c['pareto_front'])

    print('  Building test-acc Pareto front from scratch ...')
    db = _load_bench_db()
    F_all = np.array([
        [1.0 - v['test_acc_108'],
         (v['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)]
        for v in db.values()
        if 'test_acc_108' in v
    ])
    order = np.argsort(F_all[:, 0], kind='stable')
    F_sorted = F_all[order]
    nd_mask = np.zeros(len(F_sorted), dtype=bool)
    best_obj1 = np.inf
    for i in range(len(F_sorted)):
        if F_sorted[i, 1] < best_obj1:
            nd_mask[i] = True
            best_obj1 = F_sorted[i, 1]
    return F_sorted[nd_mask]


def main(args):
    methods = []
    if args.random:
        methods.append('random')
    if args.random_ga:
        methods.append('random_ga')
    if args.nsga2:
        methods.append('nsga2')
    if args.nsga2_single:
        methods.append('nsga2-single')
    if args.samos_rfr:
        methods.append('samos-rfr')
    if args.samos_xgb:
        methods.append('samos-xgb')
    if not methods:
        methods = ['random', 'random_ga', 'nsga2', 'nsga2-single', 'samos-rfr', 'samos-xgb']

    print(f'Loading benchmark data from {DATA_FILE} ...')
    bench_db = _load_bench_db()
    print(f'  {len(bench_db):,} architectures')

    print('Loading test-acc Pareto reference front ...')
    pareto_ref = _load_test_pareto_ref()
    print(f'  {len(pareto_ref)} non-dominated points')

    hv_ceiling = float(HV(ref_point=np.array([1.05, 1.05]))(pareto_ref))
    print(f'  Reference front: {len(pareto_ref)} pts  '
          f'test_err=[{pareto_ref[:,0].min():.4f}, {pareto_ref[:,0].max():.4f}]  '
          f'n_params_norm=[{pareto_ref[:,1].min():.4f}, {pareto_ref[:,1].max():.4f}]  '
          f'hv_ceiling={hv_ceiling:.6f}')

    plot_out = os.path.join(RESULTS_ROOT, 'baseline_hv_igd.png')
    print(f'\nGenerating HV / IGD+ plot ...')
    plot_results(methods, args.n_gen, args.pop_size, hv_ceiling, plot_out,
                 results_root=RESULTS_ROOT)

    coverage_out = os.path.join(RESULTS_ROOT, 'baseline_coverage.png')
    print(f'\nGenerating exploration coverage plot ...')
    plot_exploration_coverage(
        methods=methods,
        results_root=RESULTS_ROOT,
        bench_db=bench_db,
        pareto_ref=pareto_ref,
        acc_key='test_acc_108',
        eval_checkpoints=(200, 500, 1000),
        pop_size=args.pop_size,
        out_path=coverage_out,
    )

    generate_latex_table_nasbench101(
        methods=methods,
        n_gen=args.n_gen,
        results_root=RESULTS_ROOT,
        save_dir=Path(RESULTS_ROOT),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis-only: plot and summarise NASBench-101 baseline results'
    )
    parser.add_argument('--random',      action='store_true')
    parser.add_argument('--random_ga',   action='store_true')
    parser.add_argument('--nsga2',       action='store_true')
    parser.add_argument('--nsga2_single', action='store_true')
    parser.add_argument('--samos_rfr',   action='store_true')
    parser.add_argument('--samos_xgb',   action='store_true')
    parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size used in the runs (default: 20)')
    parser.add_argument('--n_gen',    type=int, default=50,
                        help='Number of generations used in the runs (default: 50)')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
