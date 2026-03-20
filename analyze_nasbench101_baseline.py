"""
Analysis-only script for NASBench-101 baseline results.

Loads existing result files from results/nasbench101_baseline/<budget_folder>/ and
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

DATA_FILE = 'problem/data/data_nasbench101.pkl'


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


def get_method_style(method_name: str) -> dict:
    """Get plotting style for a method."""
    style = {}
    
    # Random (black)
    if method_name == 'random_ga':
        style['color'] = 'black'
        style['linestyle'] = '-'
    
    # NSGA-II variants (dashed)
    elif method_name.startswith('nsga2'):
        style['linestyle'] = '--'
        if method_name == 'nsga2-uniform':
            style['color'] = 'blue'
        elif method_name == 'nsga2-xo-single':
            style['color'] = 'green'
        elif method_name == 'nsga2-no-xo-single':
            style['color'] = 'orange'
    
    # SAMOS variants (solid)
    elif method_name.startswith('samos'):
        style['linestyle'] = '-'
        # Different colors for different infill strategies
        if '200' in method_name:
            style['color'] = 'red'
        elif '50' in method_name:
            style['color'] = 'purple'
        elif '1000' in method_name:
            style['color'] = 'brown'
    
    return style


def main(args):
    # Build results root with budget folder
    n_infill = args.n_infill if args.n_infill is not None else args.pop_size
    n_doe = args.n_doe if args.n_doe is not None else args.pop_size
    budget_folder = f"G{args.n_gen}_GI{args.n_gen_inner}_P{args.pop_size}_I{n_infill}_D{n_doe}_ELIM-{args.elim_dupes}"
    
    results_root = os.path.join('results', args.experiment_name, budget_folder)
    
    if not os.path.exists(results_root):
        print(f"Results directory not found: {results_root}")
        return

    # Define all available methods
    all_methods = [
        'random_ga',
        'nsga2-uniform',
        'nsga2-xo-single',
        'nsga2-no-xo-single',
        'samos-xgb-xo-uniform-200',
        'samos-xgb-xo-single-200',
        'samos-xgb-no-xo-single-200',
        'samos-xgb-xo-uniform-50',
        'samos-xgb-xo-single-50',
        'samos-xgb-no-xo-single-50',
        'samos-xgb-xo-uniform-1000',
        'samos-xgb-xo-single-1000',
        'samos-xgb-no-xo-single-1000',
        # Legacy support
        'nsga2',
        'nsga2-single',
        'samos-rfr',
        'samos-xgb',
    ]

    # Find which methods have results
    methods = []
    for method in all_methods:
        method_dir = os.path.join(results_root, method)
        if os.path.exists(method_dir) and any(f.startswith('seed_') for f in os.listdir(method_dir)):
            methods.append(method)
    
    if not methods:
        print(f"No result files found in {results_root}")
        return

    print(f'Results root: {results_root}')
    print(f'Methods found: {methods}')

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

    plot_out = os.path.join(results_root, 'baseline_hv_igd.png')
    print(f'\nGenerating HV / IGD+ plot ...')
    plot_results(methods, args.n_gen, args.pop_size, hv_ceiling, plot_out,
                 results_root=results_root)

    coverage_out = os.path.join(results_root, 'baseline_coverage.png')
    print(f'\nGenerating exploration coverage plot ...')
    plot_exploration_coverage(
        methods=methods,
        results_root=results_root,
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
        results_root=results_root,
        save_dir=Path(results_root),
    )
    
    print(f'\nAnalysis complete. Plots saved to {results_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis-only: plot and summarise NASBench-101 baseline results'
    )
    parser.add_argument('--experiment_name', type=str, default='nasbench101_baseline',
                        help='Experiment name (default: nasbench101_baseline)')
    parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size used in the runs (default: 20)')
    parser.add_argument('--n_gen', type=int, default=50,
                        help='Number of generations used in the runs (default: 50)')
    parser.add_argument('--n_gen_inner', type=int, default=20,
                        help='SAMOS: inner NSGA-II generations (default: 20)')
    parser.add_argument('--n_infill', type=int, default=None,
                        help='SAMOS: real evaluations per outer generation (default: pop_size)')
    parser.add_argument('--n_doe', type=int, default=None,
                        help='SAMOS: initial DOE size (default: pop_size)')
    parser.add_argument('--elim_dupes', choices=['arch_str', 'pymoo_default'],
                        default='arch_str',
                        help='Duplicate elimination strategy (default: arch_str)')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
