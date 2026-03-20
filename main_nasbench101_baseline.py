"""
NASBench-101 baseline search: Random and NSGA-II.

Results are saved to:
  results/nasbench101_baseline/<method>/seed_<seed>.pkl

The pkl format matches the PDNS result format so that
plot_nasbench101_comparison.py can consume them directly.
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

from problem.nasbench101_utils import N_VAR, MIN_PARAMS, MAX_PARAMS
from strategy.genetics.duplicate import NASBench101DuplicateElimination
from problem.nasbench101_baseline_problem import NASBench101Problem
from problem.nasbench101_surrogate_problem import SurrogateProblem101
from strategy.nasbench101_callback import PDNSStyleCallback
from strategy.sampler import ValidRandomSampling101
from strategy.operations.crossover import NoCrossover101, TwoPointCrossover101
from strategy.operations.mutation import UniformMutation101, SinglePointMutation101
from analysis.plotter import plot_results, plot_exploration_coverage
from analysis.latex_table_generator import generate_latex_table_nasbench101

# ─── file-level constants ─────────────────────────────────────────────────────

DATA_FILE = 'problem/data/data_nasbench101.pkl'


# ─── main run logic ───────────────────────────────────────────────────────────

def _load_bench_db() -> dict:
    with open(DATA_FILE, 'rb') as f:
        raw = pickle.load(f)

    # data_nasbench101.pkl stores entries keyed by arch_str with val_acc_12, etc.
    return raw


def _load_test_pareto_ref() -> np.ndarray:
    """
    Load the cached test-acc Pareto front (42 points, 2-column: test_err, params_norm).
    Falls back to computing it if not cached.
    """
    cache = 'results/nasbench101_cgp_p_val_acc_12_r_n_params/G150_I20_C200_D20/cache/nasbench101_test_acc_pareto_v1.pkl'
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            c = pickle.load(f)
        return np.array(c['pareto_front'])

    # build from scratch
    print('  Building test-acc Pareto front from scratch ...')
    db = _load_bench_db()
    F_all = np.array([
        [1.0 - v['test_acc_108'],
         (v['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)]
        for v in db.values()
        if 'test_acc_108' in v
    ])
    # 2-objective Pareto front: sort by obj0, sweep for decreasing obj1.
    # O(n log n) time, O(n) memory — avoids the O(n²).
    order = np.argsort(F_all[:, 0], kind='stable')
    F_sorted = F_all[order]
    nd_mask = np.zeros(len(F_sorted), dtype=bool)
    best_obj1 = np.inf
    for i in range(len(F_sorted)):
        if F_sorted[i, 1] < best_obj1:
            nd_mask[i] = True
            best_obj1 = F_sorted[i, 1]
    return F_sorted[nd_mask]


def run_single(method: str, seed: int, pop_size: int, n_gen: int, bench_db: dict, pareto_ref: np.ndarray,
               n_doe=None, n_infill=None, n_gen_inner=20, inner_pop_size=None,
               warm_start_ratio=0.75, predict_obj=None, real_obj=None, elim_dupes_mode='arch_str'):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    problem  = NASBench101Problem(bench_db)
    callback = PDNSStyleCallback(bench_db, pareto_ref)
    sampling = ValidRandomSampling101()

    if elim_dupes_mode == 'arch_str':
        elim_dupes   = NASBench101DuplicateElimination(bench_db)
        dedup_key_fn = elim_dupes.key
    else:  # pymoo_default
        elim_dupes   = True
        dedup_key_fn = None

    n_gen_minimize = n_gen   # may be overridden for 'random'

    if method == 'random':
        # Sample all architectures in one shot — no generational structure.
        n_total = pop_size * n_gen
        algorithm = RandomGA(pop_size=n_total,
                             sampling=sampling,
                             eliminate_duplicates=elim_dupes,)
        n_gen_minimize = 1

    elif method == 'random_ga':
        algorithm = RandomGA(pop_size=pop_size,
                             sampling=sampling,
                             eliminate_duplicates=elim_dupes,)

    # ─── NSGA-II variants ────────────────────────────────────────────────────
    elif method == 'nsga2-uniform':
        # NSGA2 with xo (TwoPoint), uniform mutation
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=TwoPointCrossover101(prob=0.9),
            mutation=UniformMutation101(prob=1.0 / N_VAR, eta=1.0),
            eliminate_duplicates=elim_dupes,
        )
    elif method == 'nsga2-xo-single':
        # NSGA2 with xo (TwoPoint), single mutation
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=TwoPointCrossover101(prob=0.9),
            mutation=SinglePointMutation101(),
            eliminate_duplicates=elim_dupes,
        )
    elif method == 'nsga2-no-xo-single':
        # NSGA2 with no crossover, single mutation
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=NoCrossover101(),
            mutation=SinglePointMutation101(),
            eliminate_duplicates=elim_dupes,
        )

    # Legacy names (for backward compatibility)
    elif method == 'nsga2':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=TwoPointCrossover101(prob=0.9),
            mutation=UniformMutation101(prob=1.0 / N_VAR, eta=1.0),
            eliminate_duplicates=elim_dupes,
        )
    elif method == 'nsga2-single':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=NoCrossover101(),
            mutation=SinglePointMutation101(),
            eliminate_duplicates=elim_dupes,
        )

    # ─── SAMOS variants ──────────────────────────────────────────────────────
    elif method.startswith('samos-'):
        # Parse method name to extract crossover, mutation, inner infill (surrogate evaluations), and type
        # Format: samos-{type}-{xo_type}-{mut_type}-{inner_infill}
        # e.g., samos-xgb-xo-uniform-200, samos-xgb-no-xo-single-50, samos-rfr-xo-single-200
        # inner_infill (50, 200, 1000) controls the inner NSGA-II loop population size (surrogate evals)
        # Real evaluation budget (n_infill) stays constant to match random/nsga2
        parts = method.split('-')
        
        samos_type = parts[1]  # 'xgb' or 'rfr'
        
        # Default inner infill and crossover/mutation config
        inner_infill = None
        crossover_op = NoCrossover101()
        mutation_op = SinglePointMutation101()
        
        # Parse crossover configuration
        if 'xo' in parts:
            if 'xo-uniform' in method:
                crossover_op = TwoPointCrossover101(prob=0.9)
                mutation_op = UniformMutation101(prob=1.0 / N_VAR, eta=1.0)
            elif 'xo-single' in method:
                crossover_op = TwoPointCrossover101(prob=0.9)
                mutation_op = SinglePointMutation101()
            # else: no-xo-single (default above)
        
        # Extract inner infill value if present (controls inner NSGA-II pop size on surrogate)
        if len(parts) >= 2:
            last_part = parts[-1]
            try:
                inner_infill = int(last_part)
            except ValueError:
                pass
        
        n_doe    = n_doe    if n_doe    is not None else pop_size
        n_infill = n_infill if n_infill is not None else pop_size  # Real eval budget stays constant
        
        # Set inner pop size based on inner_infill (surrogate evaluation budget per inner loop)
        if inner_infill is not None:
            inner_pop_size = inner_infill
        else:
            inner_pop_size = inner_pop_size if inner_pop_size is not None else pop_size * 10
        
        predict_obj = predict_obj if predict_obj is not None else ['val_err_12']
        real_obj    = real_obj    if real_obj    is not None else ['n_params']
        print(f'  [SAMOS] type={samos_type}  predict={predict_obj}  real={real_obj}  n_infill={n_infill} (real evals)  inner_pop_size={inner_pop_size} (surrogate evals)')

        rng  = np.random.RandomState(seed)
        surrogates = [
            (RFR(20, seed=rng.randint(0, 2**31 - 1))
             if samos_type == 'rfr' else
             XGBoost(100, seed=rng.randint(0, 2**31 - 1)))
            for _ in range(len(predict_obj))
        ]
        factory = lambda surrs, _ro=real_obj, _db=bench_db: SurrogateProblem101(surrs, _ro, _db)
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
            dedup_key_fn=dedup_key_fn,
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=('n_gen', n_gen_minimize),
        seed=seed,
        callback=callback,
        save_history=False,
        verbose=True,
    )

    data = results.algorithm.callback.data
    data['time'] = problem.time          # list of timestamps (PDNS-compatible)
    data['log_archs'] = problem.log_archs

    return data


def main(args):
    # Build budget folder name: G<n_gen>_GI<n_gen_inner>_P<pop_size>_I<infill>_D<n_doe>_ELIM-<elim_dupes>
    n_infill = args.n_infill if args.n_infill is not None else args.pop_size
    n_doe = args.n_doe if args.n_doe is not None else args.pop_size
    budget_folder = f"G{args.n_gen}_GI{args.n_gen_inner}_P{args.pop_size}_I{n_infill}_D{n_doe}_ELIM-{args.elim_dupes}"
    
    results_root = os.path.join('results', args.experiment_name, budget_folder)

    print(f'Loading benchmark data from {DATA_FILE} ...')
    bench_db = _load_bench_db()
    print(f'  {len(bench_db):,} architectures')

    print('Loading test-acc Pareto reference front ...')
    pareto_ref = _load_test_pareto_ref()
    print(f'  {len(pareto_ref)} non-dominated points')

    # Define all available methods
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
        # Legacy names (for backward compatibility)
        'nsga2',
        'nsga2-single',
        'samos-rfr',
        'samos-xgb',
    ]

    methods = []
    if args.random:
        methods.append('random')
    if args.random_ga:
        methods.append('random_ga')
    if args.nsga2_uniform:
        methods.append('nsga2-uniform')
    if args.nsga2_xo_single:
        methods.append('nsga2-xo-single')
    if args.nsga2_no_xo_single:
        methods.append('nsga2-no-xo-single')
    if args.samos_xgb_uniform_200:
        methods.append('samos-xgb-xo-uniform-200')
    if args.samos_xgb_single_200:
        methods.append('samos-xgb-xo-single-200')
    if args.samos_xgb_no_xo_single_200:
        methods.append('samos-xgb-no-xo-single-200')
    if args.samos_xgb_uniform_50:
        methods.append('samos-xgb-xo-uniform-50')
    if args.samos_xgb_single_50:
        methods.append('samos-xgb-xo-single-50')
    if args.samos_xgb_no_xo_single_50:
        methods.append('samos-xgb-no-xo-single-50')
    if args.samos_xgb_uniform_1000:
        methods.append('samos-xgb-xo-uniform-1000')
    if args.samos_xgb_single_1000:
        methods.append('samos-xgb-xo-single-1000')
    if args.samos_xgb_no_xo_single_1000:
        methods.append('samos-xgb-no-xo-single-1000')
    # Legacy support
    if args.nsga2:
        methods.append('nsga2')
    if args.nsga2_single:
        methods.append('nsga2-single')
    if args.samos_rfr:
        methods.append('samos-rfr')
    if args.samos_xgb:
        methods.append('samos-xgb')
    
    if not methods:
        methods = all_methods

    for method in methods:
        save_dir = os.path.join(results_root, method)
        os.makedirs(save_dir, exist_ok=True)

        for seed in args.seeds:
            out_path = os.path.join(save_dir, f'seed_{seed}.pkl')
            if os.path.exists(out_path) and not args.overwrite:
                print(f'[SKIP] {method}/seed_{seed} already exists')
                continue

            print(f'\n[RUN] method={method}  seed={seed}  pop={args.pop_size}  n_gen={args.n_gen}')
            data = run_single(
                method, seed, args.pop_size, args.n_gen, bench_db, pareto_ref,
                n_doe=args.n_doe, n_infill=args.n_infill, n_gen_inner=args.n_gen_inner,
                inner_pop_size=args.inner_pop_size,
                warm_start_ratio=args.warm_start_ratio,
                predict_obj=args.predict_obj, real_obj=args.real_obj,
                elim_dupes_mode=args.elim_dupes,
            )

            with open(out_path, 'wb') as f:
                pickle.dump(data, f)
            print(f'  Saved -> {out_path}')

    # ── plot HV and IGD+ trajectories ─────────────────────────────────────────
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NASBench-101 baselines: Random + NSGA-II + SAMOS (PDNS-style)'
    )
    # ─── Basic methods ────────────────────────────────────────────────────────
    parser.add_argument('--random', action='store_true',
                        help='Run one-shot random search (pop_size * n_gen samples at once)')
    parser.add_argument('--random_ga', action='store_true',
                        help='Run generational random search (pop_size samples per generation)')
    
    # ─── NSGA-II variants ────────────────────────────────────────────────────
    parser.add_argument('--nsga2_uniform', action='store_true',
                        help='Run NSGA-II with 2-point crossover + uniform mutation')
    parser.add_argument('--nsga2_xo_single', action='store_true',
                        help='Run NSGA-II with 2-point crossover + single-point mutation')
    parser.add_argument('--nsga2_no_xo_single', action='store_true',
                        help='Run NSGA-II with no crossover + single-point mutation')
    
    # ─── SAMOS-XGB variants with infill=200 ────────────────────────────────
    parser.add_argument('--samos_xgb_uniform_200', action='store_true',
                        help='Run SAMOS-XGB with xo + uniform mutation, infill=200')
    parser.add_argument('--samos_xgb_single_200', action='store_true',
                        help='Run SAMOS-XGB with xo + single mutation, infill=200')
    parser.add_argument('--samos_xgb_no_xo_single_200', action='store_true',
                        help='Run SAMOS-XGB with no xo + single mutation, infill=200')
    
    # ─── SAMOS-XGB variants with infill=50 ─────────────────────────────────
    parser.add_argument('--samos_xgb_uniform_50', action='store_true',
                        help='Run SAMOS-XGB with xo + uniform mutation, infill=50')
    parser.add_argument('--samos_xgb_single_50', action='store_true',
                        help='Run SAMOS-XGB with xo + single mutation, infill=50')
    parser.add_argument('--samos_xgb_no_xo_single_50', action='store_true',
                        help='Run SAMOS-XGB with no xo + single mutation, infill=50')
    
    # ─── SAMOS-XGB variants with infill=1000 ───────────────────────────────
    parser.add_argument('--samos_xgb_uniform_1000', action='store_true',
                        help='Run SAMOS-XGB with xo + uniform mutation, infill=1000')
    parser.add_argument('--samos_xgb_single_1000', action='store_true',
                        help='Run SAMOS-XGB with xo + single mutation, infill=1000')
    parser.add_argument('--samos_xgb_no_xo_single_1000', action='store_true',
                        help='Run SAMOS-XGB with no xo + single mutation, infill=1000')
    
    # ─── Legacy method flags (for backward compatibility) ───────────────────
    parser.add_argument('--nsga2',  action='store_true',
                        help='(Legacy) Run NSGA-II with 2-point crossover + uniform mutation')
    parser.add_argument('--nsga2_single', action='store_true',
                        help='(Legacy) Run NSGA-II with no crossover + single-point mutation')
    parser.add_argument('--samos_rfr', action='store_true',
                        help='(Legacy) Run simplified SAMOS with Random Forest surrogate')
    parser.add_argument('--samos_xgb', action='store_true',
                        help='(Legacy) Run simplified SAMOS with XGBoost surrogate')
    
    # ─── Search budget parameters ─────────────────────────────────────────
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)),
                        help='Seeds to run (default: 0-9)')
    parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size (default: 20, matching PDNS)')
    parser.add_argument('--n_gen', type=int, default=50,
                        help='Number of generations (default: 50, -> 1000 evals)')
    parser.add_argument('--n_doe', type=int, default=None,
                        help='SAMOS: initial DOE size (default: pop_size)')
    parser.add_argument('--n_infill', type=int, default=None,
                        help='SAMOS: real evaluations per outer generation (default: pop_size)')
    parser.add_argument('--n_gen_inner', type=int, default=20,
                        help='SAMOS: inner NSGA-II generations (default: 20)')
    parser.add_argument('--inner_pop_size', type=int, default=None,
                        help='SAMOS: inner NSGA-II population size (default: pop_size * 10)')
    parser.add_argument('--warm_start_ratio', type=float, default=0.75,
                        help='SAMOS: fraction of inner pop warm-started from best archive (default: 0.75)')
    parser.add_argument('--predict_obj', type=str, nargs='+', default=['val_err_12'],
                        help='SAMOS: objectives approximated by surrogates (default: val_err_12)')
    parser.add_argument('--real_obj', type=str, nargs='*', default=['n_params'],
                        help='SAMOS: objectives evaluated exactly in inner loop (default: n_params)')
    parser.add_argument('--elim_dupes', choices=['arch_str', 'pymoo_default'],
                        default='arch_str',
                        help='Duplicate elimination strategy: arch_str (canonical ModelSpec hash, '
                             'catches phenotypically identical architectures) or '
                             'pymoo_default (raw vector comparison). Default: arch_str')
    parser.add_argument('--experiment_name', type=str, default='nasbench101_baseline',
                        help='Experiment name; results are saved to results/<experiment_name>/')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if result file already exists')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
