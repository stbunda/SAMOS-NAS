"""
Analysis-only script for NASBench-201 baseline results.

Loads existing result files from
  results/<experiment_name>/<dataset>/<budget_folder>/
and generates HV/IGD+ plots and a LaTeX table.
No experiments are run.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

from problem.nasbench201_utils import (
    VALID_DATASETS, DATASET_INFO, compute_flops_range, build_test_pareto_ref,
)
from analysis.plotter import plot_results
from analysis.latex_table_generator import generate_latex_table_nasbench201

DATA_FILE = 'problem/data/data_nasbench201.pkl'


def _load_bench_db() -> dict:
    with open(DATA_FILE, 'rb') as f:
        return pickle.load(f)



def main(args):
    dataset_info = DATASET_INFO[args.dataset]
    val_key  = dataset_info['val_key']
    test_key = dataset_info['test_key']

    # Build results root — must match the folder layout written by main_nasbench201_baseline.py
    n_infill = args.n_infill if args.n_infill is not None else args.pop_size
    n_doe    = args.n_doe    if args.n_doe    is not None else args.pop_size
    budget_folder = (
        f"G{args.n_gen}_GI{args.n_gen_inner}_P{args.pop_size}_I{n_infill}_D{n_doe}"
    )

    results_root = os.path.join(
        'results', args.experiment_name, args.dataset, budget_folder
    )

    if not os.path.exists(results_root):
        print(f'Results directory not found: {results_root}')
        return

    # ── discover which methods have results ───────────────────────────────────
    all_methods = [
        'random',
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
        # Legacy names
        'nsga2',
        'nsga2-single',
        'samos-rfr',
        'samos-xgb',
    ]

    methods = [
        m for m in all_methods
        if os.path.isdir(os.path.join(results_root, m))
        and any(
            f.startswith('seed_') and f.endswith('.pkl')
            for f in os.listdir(os.path.join(results_root, m))
        )
    ]

    if not methods:
        print(f'No result files found in {results_root}')
        return

    print(f'Results root:  {results_root}')
    print(f'Dataset:       {args.dataset}')
    print(f'Methods found: {methods}')

    # ── load benchmark data ───────────────────────────────────────────────────
    print(f'\nLoading benchmark data from {DATA_FILE} ...')
    bench_db = _load_bench_db()
    print(f'  {len(bench_db):,} architectures')

    print(f'Computing FLOPs range for val key {val_key} ...')
    val_min_flops, val_max_flops = compute_flops_range(bench_db, val_key)
    print(f'  val_flops: [{val_min_flops:.5f}, {val_max_flops:.5f}]')

    print(f'Computing FLOPs range for test key {test_key} ...')
    test_min_flops, test_max_flops = compute_flops_range(bench_db, test_key)
    print(f'  test_flops: [{test_min_flops:.5f}, {test_max_flops:.5f}]')

    # ── build Pareto reference fronts ─────────────────────────────────────────
    print('\nBuilding test-acc Pareto reference front ...')
    pareto_ref = build_test_pareto_ref(
        bench_db, test_key, test_min_flops, test_max_flops
    )
    hv_ceiling = float(HV(ref_point=np.array([1.05, 1.05]))(pareto_ref))
    print(
        f'  {len(pareto_ref)} non-dominated points  '
        f'test_err=[{pareto_ref[:, 0].min():.4f}, {pareto_ref[:, 0].max():.4f}]  '
        f'flops_norm=[{pareto_ref[:, 1].min():.4f}, {pareto_ref[:, 1].max():.4f}]  '
        f'hv_ceiling={hv_ceiling:.6f}'
    )

    # ── HV / IGD+ trajectory plot ─────────────────────────────────────────────
    plot_out = os.path.join(results_root, 'baseline_hv_igd.png')
    print(f'\nGenerating HV / IGD+ plot ...')
    plot_results(
        methods, args.n_gen, args.pop_size, hv_ceiling, plot_out,
        results_root=results_root,
    )

    # ── LaTeX table ───────────────────────────────────────────────────────────
    generate_latex_table_nasbench201(
        methods=methods,
        n_gen=args.n_gen,
        results_root=results_root,
        dataset=args.dataset,
        save_dir=Path(results_root),
    )

    print(f'\nAnalysis complete. Results saved to {results_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis-only: plot and summarise NASBench-201 baseline results'
    )
    parser.add_argument(
        '--dataset', choices=VALID_DATASETS, default='cifar10-valid',
        help='Dataset to analyse (default: cifar10-valid)',
    )
    parser.add_argument('--experiment_name', type=str, default='nasbench201_baseline',
                        help='Experiment name (default: nasbench201_baseline)')
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

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
