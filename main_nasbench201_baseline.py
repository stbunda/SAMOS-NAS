"""
NASBench-201 baseline search: Random and NSGA-II.

Results are saved to:
  results/nasbench201_baseline/<dataset>/<budget_folder>/<method>/seed_<seed>.pkl

The pkl format matches the PDNS result format so that analysis scripts can
consume them directly.
"""

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize

from strategy.algorithm.algorithms import RandomGA
from strategy.surrogate.models import RFR, XGBoost
from strategy.surrogate.samos_minimal import SAMOSMinimal as SAMOS

from problem.nasbench201.utils import (
    N_VAR, DATASET_INFO, VALID_DATASETS,
    compute_flops_range, build_test_pareto_ref, _vec_to_arch_str,
)
from strategy.genetics.duplicate import NASBench201DuplicateElimination
from problem.nasbench201.baseline_problem import NASBench201Problem
from problem.nasbench201.surrogate_problem import SurrogateProblem201
from strategy.callbacks import NASArchiveCallback
from problem.nasbench201.utils import _update_archive, _test_archive as _test_archive_201
from strategy.sampler import ValidRandomSampling201
from strategy.operations.crossover import NoCrossover, UniformCrossover201
from strategy.operations.mutation import UniformMutation201, SinglePointMutation201
from analysis.plotter import plot_results
from analysis.latex_table_generator import generate_latex_table_nasbench201

# ─── file-level constants ─────────────────────────────────────────────────────

DATA_FILE = 'problem/data/data_nasbench201.pkl'


# ─── main run logic ───────────────────────────────────────────────────────────

def _load_bench_db() -> dict:
    with open(DATA_FILE, 'rb') as f:
        return pickle.load(f)



def run_single(
    method: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    bench_db: dict,
    pareto_ref: np.ndarray,
    dataset: str,
    val_min_flops: float,
    val_max_flops: float,
    test_key: str,
    test_min_flops: float,
    test_max_flops: float,
    n_doe=None,
    n_infill=None,
    n_gen_inner=20,
    inner_pop_size=None,
    warm_start_ratio=0.75,
    predict_obj=None,
    real_obj=None,
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    problem  = NASBench201Problem(
        bench_db,
        dataset=dataset,
        min_flops=val_min_flops,
        max_flops=val_max_flops,
    )
    callback = NASArchiveCallback(
        bench_db, pareto_ref, _update_archive, _test_archive_201,
        test_key=test_key,
        min_flops=test_min_flops,
        max_flops=test_max_flops,
    )
    sampling   = ValidRandomSampling201()
    elim_dupes = NASBench201DuplicateElimination()

    if method == 'random_ga':
        algorithm = RandomGA(
            pop_size=pop_size,
            sampling=sampling,
            eliminate_duplicates=elim_dupes,
        )

    # ─── NSGA-II variants ────────────────────────────────────────────────────
    elif method == 'nsga2-uniform':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=UniformCrossover201(prob=0.9),
            mutation=UniformMutation201(prob=1.0 / N_VAR),
            eliminate_duplicates=elim_dupes,
        )
    elif method == 'nsga2-xo-single':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=UniformCrossover201(prob=0.9),
            mutation=SinglePointMutation201(),
            eliminate_duplicates=elim_dupes,
        )
    elif method == 'nsga2-no-xo-single':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=NoCrossover(),
            mutation=SinglePointMutation201(),
            eliminate_duplicates=elim_dupes,
        )

    # Legacy names (for backward compatibility)
    elif method == 'nsga2':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=UniformCrossover201(prob=0.9),
            mutation=UniformMutation201(prob=1.0 / N_VAR),
            eliminate_duplicates=elim_dupes,
        )
    elif method == 'nsga2-single':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=NoCrossover(),
            mutation=SinglePointMutation201(),
            eliminate_duplicates=elim_dupes,
        )

    # ─── SAMOS variants ──────────────────────────────────────────────────────
    elif method.startswith('samos-'):
        # Format: samos-{type}-{xo_type}-{mut_type}-{inner_infill}
        # e.g., samos-xgb-xo-uniform-200, samos-xgb-no-xo-single-50
        parts = method.split('-')

        samos_type = parts[1]   # 'xgb' or 'rfr'

        inner_infill  = None
        crossover_op  = NoCrossover()
        mutation_op   = SinglePointMutation201()

        if 'xo' in parts:
            if 'xo-uniform' in method:
                crossover_op = UniformCrossover201(prob=0.9)
                mutation_op  = UniformMutation201(prob=1.0 / N_VAR)
            elif 'xo-single' in method:
                crossover_op = UniformCrossover201(prob=0.9)
                mutation_op  = SinglePointMutation201()

        last_part = parts[-1]
        try:
            inner_infill = int(last_part)
        except ValueError:
            pass

        n_doe    = n_doe    if n_doe    is not None else pop_size
        n_infill = n_infill if n_infill is not None else pop_size

        if inner_infill is not None:
            inner_pop_size = inner_infill
        else:
            inner_pop_size = inner_pop_size if inner_pop_size is not None else pop_size * 10

        predict_obj = predict_obj if predict_obj is not None else ['val_err']
        real_obj    = real_obj    if real_obj    is not None else ['flops']
        print(
            f'  [SAMOS] type={samos_type}  predict={predict_obj}  real={real_obj}  '
            f'n_infill={n_infill} (real evals)  inner_pop_size={inner_pop_size} (surrogate evals)'
        )

        rng = np.random.RandomState(seed)
        surrogates = [
            (RFR(20, seed=rng.randint(0, 2**31 - 1))
             if samos_type == 'rfr' else
             XGBoost(100, seed=rng.randint(0, 2**31 - 1)))
            for _ in range(len(predict_obj))
        ]

        _ds  = dataset
        _mf  = val_min_flops
        _Mf  = val_max_flops
        factory = lambda surrs, _ro=real_obj, _db=bench_db, _d=_ds, _mi=_mf, _ma=_Mf: (
            SurrogateProblem201(surrs, _ro, _db, _d, _mi, _ma)
        )

        algorithm = SAMOS(
            sampling=sampling,
            surrogates=surrogates,
            surrogate_problem_factory=factory,
            crossover=crossover_op,
            mutation=mutation_op,
            n_doe=n_doe,
            n_infill=n_infill,
            n_gen_inner=n_gen_inner,
            ga_pop_size=inner_pop_size,
            warm_start_ratio=warm_start_ratio,
            use_subset_selection=False,
            eliminate_duplicates=elim_dupes,
            dedup_key_fn=elim_dupes.key,
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=('n_gen', n_gen),
        seed=seed,
        callback=callback,
        save_history=False,
        verbose=True,
    )

    data              = results.algorithm.callback.data
    data['time']      = problem.time
    data['log_archs'] = problem.log_archs

    return data


def main(args):
    dataset_info = DATASET_INFO[args.dataset]
    val_key  = dataset_info['val_key']
    test_key = dataset_info['test_key']

    # Budget folder: G<n_gen>_GI<n_gen_inner>_P<pop_size>_I<infill>_D<n_doe>
    n_infill = args.n_infill if args.n_infill is not None else args.pop_size
    n_doe    = args.n_doe    if args.n_doe    is not None else args.pop_size
    budget_folder = (
        f"G{args.n_gen}_GI{args.n_gen_inner}_P{args.pop_size}_I{n_infill}_D{n_doe}"
    )

    results_root = os.path.join(
        'results', args.experiment_name, args.dataset, budget_folder
    )

    print(f'Loading benchmark data from {DATA_FILE} ...')
    bench_db = _load_bench_db()
    print(f'  {len(bench_db):,} architectures')

    print(f'Computing FLOPs range for dataset {val_key} ...')
    val_min_flops, val_max_flops = compute_flops_range(bench_db, val_key)
    print(f'  val_flops: [{val_min_flops:.5f}, {val_max_flops:.5f}]')

    print(f'Computing FLOPs range for test key {test_key} ...')
    test_min_flops, test_max_flops = compute_flops_range(bench_db, test_key)
    print(f'  test_flops: [{test_min_flops:.5f}, {test_max_flops:.5f}]')

    print('Building test-acc Pareto reference front ...')
    pareto_ref = build_test_pareto_ref(
        bench_db, test_key, test_min_flops, test_max_flops
    )
    print(f'  {len(pareto_ref)} non-dominated points')

    # ── define all available methods ──────────────────────────────────────────
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
        # Legacy names
        'nsga2',
        'nsga2-single',
        'samos-rfr',
        'samos-xgb',
    ]

    methods = []
    if args.random_ga:       methods.append('random_ga')
    if args.nsga2_uniform:   methods.append('nsga2-uniform')
    if args.nsga2_xo_single: methods.append('nsga2-xo-single')
    if args.nsga2_no_xo_single: methods.append('nsga2-no-xo-single')
    if args.samos_xgb_uniform_200:    methods.append('samos-xgb-xo-uniform-200')
    if args.samos_xgb_single_200:     methods.append('samos-xgb-xo-single-200')
    if args.samos_xgb_no_xo_single_200: methods.append('samos-xgb-no-xo-single-200')
    if args.samos_xgb_uniform_50:     methods.append('samos-xgb-xo-uniform-50')
    if args.samos_xgb_single_50:      methods.append('samos-xgb-xo-single-50')
    if args.samos_xgb_no_xo_single_50: methods.append('samos-xgb-no-xo-single-50')
    if args.samos_xgb_uniform_1000:   methods.append('samos-xgb-xo-uniform-1000')
    if args.samos_xgb_single_1000:    methods.append('samos-xgb-xo-single-1000')
    if args.samos_xgb_no_xo_single_1000: methods.append('samos-xgb-no-xo-single-1000')
    # Legacy
    if args.nsga2:       methods.append('nsga2')
    if args.nsga2_single: methods.append('nsga2-single')
    if args.samos_rfr:   methods.append('samos-rfr')
    if args.samos_xgb:   methods.append('samos-xgb')

    if not methods:
        methods = all_methods

    # ── run each method / seed ────────────────────────────────────────────────
    for method in methods:
        save_dir = os.path.join(results_root, method)
        os.makedirs(save_dir, exist_ok=True)

        for seed in args.seeds:
            out_path = os.path.join(save_dir, f'seed_{seed}.pkl')
            if os.path.exists(out_path) and not args.overwrite:
                print(f'[SKIP] {method}/seed_{seed} already exists')
                continue

            print(
                f'\n[RUN] method={method}  seed={seed}  pop={args.pop_size}'
                f'  n_gen={args.n_gen}  dataset={args.dataset}'
            )
            data = run_single(
                method, seed,
                args.pop_size, args.n_gen,
                bench_db, pareto_ref,
                dataset=args.dataset,
                val_min_flops=val_min_flops,
                val_max_flops=val_max_flops,
                test_key=test_key,
                test_min_flops=test_min_flops,
                test_max_flops=test_max_flops,
                n_doe=args.n_doe,
                n_infill=args.n_infill,
                n_gen_inner=args.n_gen_inner,
                inner_pop_size=args.inner_pop_size,
                warm_start_ratio=args.warm_start_ratio,
                predict_obj=args.predict_obj,
                real_obj=args.real_obj,
            )

            with open(out_path, 'wb') as f:
                pickle.dump(data, f)
            print(f'  Saved -> {out_path}')

    # ── plot HV and IGD+ trajectories ─────────────────────────────────────────
    hv_ceiling = float(HV(ref_point=np.array([1.05, 1.05]))(pareto_ref))
    print(
        f'  Reference front: {len(pareto_ref)} pts  '
        f'test_err=[{pareto_ref[:, 0].min():.4f}, {pareto_ref[:, 0].max():.4f}]  '
        f'flops_norm=[{pareto_ref[:, 1].min():.4f}, {pareto_ref[:, 1].max():.4f}]  '
        f'hv_ceiling={hv_ceiling:.6f}'
    )

    plot_out = os.path.join(results_root, 'baseline_hv_igd.png')
    print('\nGenerating HV / IGD+ plot ...')
    plot_results(
        methods, args.n_gen, args.pop_size, hv_ceiling, plot_out,
        results_root=results_root,
    )

    # Note: plot_exploration_coverage is not called here because it expects a
    # flat bench_db structure (bench_db[arch_str][metric]) used in NASBench-101.
    # NASBench-201 uses bench_db[arch_str][dataset][metric] — adapt the plotter
    # separately if exploration coverage plots are needed.

    generate_latex_table_nasbench201(
        methods=methods,
        n_gen=args.n_gen,
        results_root=results_root,
        dataset=args.dataset,
        save_dir=Path(results_root),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NASBench-201 baselines: Random + NSGA-II + SAMOS'
    )

    # ─── Dataset ─────────────────────────────────────────────────────────────
    parser.add_argument(
        '--dataset', choices=VALID_DATASETS, default='cifar10-valid',
        help='Benchmark dataset to optimise (default: cifar10-valid)',
    )

    # ─── Basic methods ────────────────────────────────────────────────────────
    parser.add_argument('--random_ga', action='store_true',
                        help='Run generational random search')

    # ─── NSGA-II variants ────────────────────────────────────────────────────
    parser.add_argument('--nsga2_uniform', action='store_true',
                        help='Run NSGA-II with uniform crossover + uniform mutation')
    parser.add_argument('--nsga2_xo_single', action='store_true',
                        help='Run NSGA-II with uniform crossover + single-point mutation')
    parser.add_argument('--nsga2_no_xo_single', action='store_true',
                        help='Run NSGA-II with no crossover + single-point mutation')

    # ─── SAMOS-XGB variants with infill=200 ──────────────────────────────────
    parser.add_argument('--samos_xgb_uniform_200', action='store_true',
                        help='Run SAMOS-XGB with unif. XO + uniform mutation, infill=200')
    parser.add_argument('--samos_xgb_single_200', action='store_true',
                        help='Run SAMOS-XGB with unif. XO + single-pt mutation, infill=200')
    parser.add_argument('--samos_xgb_no_xo_single_200', action='store_true',
                        help='Run SAMOS-XGB with no XO + single-pt mutation, infill=200')

    # ─── SAMOS-XGB variants with infill=50 ───────────────────────────────────
    parser.add_argument('--samos_xgb_uniform_50', action='store_true',
                        help='Run SAMOS-XGB with unif. XO + uniform mutation, infill=50')
    parser.add_argument('--samos_xgb_single_50', action='store_true',
                        help='Run SAMOS-XGB with unif. XO + single-pt mutation, infill=50')
    parser.add_argument('--samos_xgb_no_xo_single_50', action='store_true',
                        help='Run SAMOS-XGB with no XO + single-pt mutation, infill=50')

    # ─── SAMOS-XGB variants with infill=1000 ─────────────────────────────────
    parser.add_argument('--samos_xgb_uniform_1000', action='store_true',
                        help='Run SAMOS-XGB with unif. XO + uniform mutation, infill=1000')
    parser.add_argument('--samos_xgb_single_1000', action='store_true',
                        help='Run SAMOS-XGB with unif. XO + single-pt mutation, infill=1000')
    parser.add_argument('--samos_xgb_no_xo_single_1000', action='store_true',
                        help='Run SAMOS-XGB with no XO + single-pt mutation, infill=1000')

    # ─── Legacy method flags (for backward compatibility) ────────────────────
    parser.add_argument('--nsga2',  action='store_true',
                        help='(Legacy) Run NSGA-II with uniform XO + uniform mutation')
    parser.add_argument('--nsga2_single', action='store_true',
                        help='(Legacy) Run NSGA-II with no XO + single-point mutation')
    parser.add_argument('--samos_rfr', action='store_true',
                        help='(Legacy) Run SAMOS with Random Forest surrogate')
    parser.add_argument('--samos_xgb', action='store_true',
                        help='(Legacy) Run SAMOS with XGBoost surrogate')

    # ─── Search budget parameters ─────────────────────────────────────────────
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)),
                        help='Seeds to run (default: 0-9)')
    parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size (default: 20)')
    parser.add_argument('--n_gen', type=int, default=50,
                        help='Number of generations (default: 50, → 1000 evals)')
    parser.add_argument('--n_doe', type=int, default=None,
                        help='SAMOS: initial DOE size (default: pop_size)')
    parser.add_argument('--n_infill', type=int, default=None,
                        help='SAMOS: real evaluations per outer generation (default: pop_size)')
    parser.add_argument('--n_gen_inner', type=int, default=20,
                        help='SAMOS: inner NSGA-II generations (default: 20)')
    parser.add_argument('--inner_pop_size', type=int, default=None,
                        help='SAMOS: inner NSGA-II population size (default: pop_size * 10)')
    parser.add_argument('--warm_start_ratio', type=float, default=1.0,
                        help='SAMOS: warm-start fraction from best archive (default: 1.0)')
    parser.add_argument('--predict_obj', type=str, nargs='+', default=None,
                        help='SAMOS: objectives approximated by surrogates (default: [val_err])')
    parser.add_argument('--real_obj', type=str, nargs='*', default=None,
                        help='SAMOS: objectives evaluated exactly in inner loop (default: [flops])')
    parser.add_argument('--experiment_name', type=str, default='nasbench201_baseline',
                        help='Experiment name; results saved to results/<experiment_name>/')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if result file already exists')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
