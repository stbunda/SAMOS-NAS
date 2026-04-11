"""
NASBench-101 baseline search: Random and NSGA-II.

Results are saved to:
  results/nasbench101_baseline/<method>/seed_<seed>.pkl

The pkl format matches the PDNS result format so that
plot_nasbench101_comparison.py can consume them directly.
"""

import argparse
import copy
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pymoo.util.nds.non_dominated_sorting
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize

from strategy.algorithm.algorithms import RandomGA
from strategy.algorithm.gpsaf import GPSAF, SklearnGPSAF
from strategy.algorithm.ssansga2 import SSANSGA2, SklearnSSANSGA2
from strategy.algorithm.parego import run_parego_nasbench
from strategy.surrogate.models import RFR, XGBoost
from strategy.surrogate.samos_minimal import SAMOSMinimal as SAMOS
from strategy.surrogate.samos_ssa import SAMOSSA

from problem.nasbench101.utils import N_VAR, MIN_PARAMS, MAX_PARAMS, _CANONICAL_OPS_101
from strategy.genetics.duplicate import NASBench101DuplicateElimination
from problem.nasbench101.baseline_problem import NASBench101Problem
from strategy.genetics.nasbench101_lib.model_spec import ModelSpec as _ModelSpec101
from problem.nasbench101.surrogate_problem import SurrogateProblem101
from strategy.callbacks import NASArchiveCallback
from problem.nasbench101.utils import _update_archive, _test_archive as _test_archive_101
from strategy.sampler import ValidRandomSampling101
from strategy.operations.crossover import TwoPointCrossover101
from strategy.operations.mutation import SinglePointMutation101
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

#TODO imports
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
)

from smac import HyperparameterOptimizationFacade, Scenario
from smac.facade.multi_objective_facade import MultiObjectiveFacade as MOFacade

def build_config_space(seed: int = 42) -> ConfigurationSpace:
    """
    Encode a NASBench-101 cell as a flat hyperparameter vector.

    Edges (21 binary choices):  edge_i_j ∈ {0, 1}
    Op nodes (5 interior nodes, nodes 1–5):  op_k ∈ {conv3x3, conv1x1, maxpool}

    Node 0 = input  (fixed)
    Node 6 = output (fixed)
    """
    VERTICES = 7
    OPS = _CANONICAL_OPS_101
    cs = ConfigurationSpace(seed=seed)

    # --- Edge hyperparameters ---
    edge_hps = []
    for i in range(VERTICES):
        for j in range(i + 1, VERTICES):
            hp = Categorical(f"edge_{i}_{j}", [0, 1], default=0)
            edge_hps.append(hp)
    cs.add(edge_hps)

    # --- Operation hyperparameters (interior nodes 1..5) ---
    op_hps = []
    for k in range(1, VERTICES - 1):   # nodes 1, 2, 3, 4, 5
        hp = Categorical(f"op_{k}", OPS, default=OPS[0])
        op_hps.append(hp)
    cs.add(op_hps)

    return cs

def config_to_model_spec(config: Configuration):
    """Convert a SMAC Configuration into a NASBench-101 ModelSpec."""
    # Build adjacency matrix
    VERTICES = 7
    INPUT_NODE = "input"
    OUTPUT_NODE = "output"

    adj = np.zeros((VERTICES, VERTICES), dtype=int)
    for i in range(VERTICES):
        for j in range(i + 1, VERTICES):
            adj[i][j] = int(config[f"edge_{i}_{j}"])

    # Build label list: input, op_1..op_5, output
    labels = [INPUT_NODE]
    for k in range(1, VERTICES - 1):
        labels.append(config[f"op_{k}"])
    labels.append(OUTPUT_NODE)

    return _ModelSpec101(matrix=adj, ops=labels)

def lognorm(x, MIN=1, MAX=10, reverse=False):
    if not reverse:
        return (np.log(x) - np.log(MAX)) / (np.log(MIN) - np.log(MAX))
    else:
        return np.exp(x * np.log(MAX/MIN) + np.log(MIN))

def make_target_fn(bench_db):
    """
    Returns a target function compatible with SMAC's multi-objective interface.

    Returns a dict
    """
    def target_fn(config: Configuration, seed: int = 0) -> dict:
        spec = config_to_model_spec(config)

        # Query the benchmark
        if not spec.valid_spec or spec.hash_spec(_CANONICAL_OPS_101) not in bench_db:
            #invalid
            return {"val_err": 1.5, "n_params": 1.5}

        entry = bench_db[spec.hash_spec(_CANONICAL_OPS_101)]
        val_err = 1.0 - entry.get('val_acc_12', 0.0)
        n_params_norm = lognorm(entry['n_params'], MIN_PARAMS, MAX_PARAMS)

        return {"val_err": val_err, "n_params": n_params_norm}

    return target_fn

def mosmac_run(callback, seed: int, pop_size: int, n_gen: int, bench_db: dict, pareto_ref: np.ndarray,
               n_doe=None, n_infill=None, n_gen_inner=20, inner_pop_size=None,
               warm_start_ratio=0.75, predict_obj=None, real_obj=None, elim_dupes_mode='arch_str'):


    # MOSMAC compatible configspace
    cs = build_config_space(seed)
    # Target algorithm
    target_fn = make_target_fn(bench_db)
    # TODO callback to check isvalid
    # Scenario
    scenario = Scenario(
        configspace=cs,
        objectives=["val_err", "n_params"],  # multi-objective
        n_trials=500, #pop_size*n_gen,
        seed=seed,
        # output_directory=, TODO Fix
        deterministic=True,  # NASBench lookups are deterministic
        n_workers=1,
    )

    #SMAC
    smac = MOFacade(
        scenario=scenario,
        target_function=target_fn,
        overwrite=True,
    )

    incumbents = smac.optimize()

    #Logging
    traj = smac.intensifier.trajectory
    rh = smac.runhistory

    #Compute costs
    val_costs = {}
    test_costs = {}
    for config_id, config in rh.ids_config.items():
        spec = config_to_model_spec(config)
        if not spec.valid_spec or spec.hash_spec(_CANONICAL_OPS_101) not in bench_db:
            val_err = 1.0
            test_err = 1.0
            n_params_norm = 1.0
        else:
            entry = bench_db[spec.hash_spec(_CANONICAL_OPS_101)]
            val_err = 1.0 - entry.get('val_acc_12', 0.0)
            test_err = 1.0 - entry.get('test_acc_108', 0.0)
            n_params      = entry['n_params']
            n_params_norm = (n_params - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)

        val_costs[config_id] = [val_err, n_params_norm]
        test_costs[config_id] = [test_err, n_params_norm]

    def compute_scores(points):

        if len(points) == 0:
            return {"hv": 0.0, "igd_plus": 0.0}

        indicators = {
            'hv': float(callback._hv_ind(points)),
            'igd_plus': float(callback._igd_ind(points)),
        }

        return indicators

    #Compute data
    data = dict()
    #Get populations
    data["var_pop"] = [[rh.ids_config[i] for i in pop.config_ids] for pop in traj] #TODO align with other representation
    data["obj_pop"] = [[val_costs[i] for i in pop.config_ids] for pop in traj]

    #Get archive
    arch = []
    for config_id in rh.ids_config.keys():
        next_arch = [config_id] if len(arch) == 0 else copy.copy(arch[-1]) + [config_id] #Add next config to archive
        ndps = NonDominatedSorting().do(np.array([val_costs[i] for i in next_arch]), only_non_dominated_front=True) #Get non-dominated front
        arch.append([config_id for i, config_id in enumerate(next_arch) if i in ndps]) #Get first front
        #TODO check if arch is changed
    data["var_archive"] = [[rh.ids_config[i] for i in pop] for pop in arch]
    data["obj_archive"] = [[val_costs[i] for i in pop] for pop in arch]

    data["test_var_archive"] = [[rh.ids_config[i] for i in pop] for pop in arch]
    data["test_obj_archive"] = [[test_costs[i] for i in pop] for pop in arch]

    data['indicators'] = [compute_scores(np.array(pop)) for pop in data["test_obj_archive"]]
    data['time'] = list(range(len(data["test_var_archive"]))) #TODO fix

    # data = []
    # data['time'] = problem.time  # list of timestamps (PDNS-compatible)
    # data['log_archs'] = problem.log_archs
    return data

def run_single(method: str, seed: int, pop_size: int, n_gen: int, bench_db: dict, pareto_ref: np.ndarray,
               n_doe=None, n_infill=None, n_gen_inner=20, inner_pop_size=None,
               warm_start_ratio=0.75, predict_obj=None, real_obj=None, elim_dupes_mode='arch_str'):
    np.random.seed(seed)
    random.seed(seed)

    problem  = NASBench101Problem(bench_db)
    callback = NASArchiveCallback(bench_db, pareto_ref, _update_archive, _test_archive_101)
    sampling = ValidRandomSampling101()

    if elim_dupes_mode == 'arch_str':
        elim_dupes   = NASBench101DuplicateElimination(bench_db)
        dedup_key_fn = elim_dupes.key
    else:
        elim_dupes   = True
        dedup_key_fn = None

    # ─── Standalone methods (manage their own loop) ───────────────────────────
    if method == 'mosmac':
        return mosmac_run(callback, seed, pop_size, n_gen, bench_db, pareto_ref,
                          n_doe, n_infill, n_gen_inner, inner_pop_size,
                          warm_start_ratio, predict_obj, real_obj, elim_dupes_mode)

    if method == 'parego':
        return run_parego_nasbench(problem, callback, sampling, seed, pop_size, n_gen,
                                   n_doe=n_doe)

    # ─── pymoo minimize-based methods ─────────────────────────────────────────
    xover = TwoPointCrossover101(prob=0.9)
    mut   = SinglePointMutation101()

    if method == 'random':
        algorithm = RandomGA(pop_size=pop_size, sampling=sampling,
                             eliminate_duplicates=elim_dupes)

    elif method == 'nsga2':
        algorithm = NSGA2(pop_size=pop_size, sampling=sampling,
                          crossover=xover, mutation=mut,
                          eliminate_duplicates=elim_dupes)

    elif method in ('samos-rfr', 'samos-xgb'):
        samos_type  = 'rfr' if method == 'samos-rfr' else 'xgb'
        n_doe_      = n_doe    if n_doe    is not None else pop_size
        n_infill_   = n_infill if n_infill is not None else pop_size
        inner_ps    = inner_pop_size if inner_pop_size is not None else pop_size * 10
        predict_obj = predict_obj if predict_obj is not None else ['val_err_12']
        real_obj    = real_obj    if real_obj    is not None else ['n_params']
        print(f'  [SAMOS] type={samos_type}  predict={predict_obj}  real={real_obj}  '
              f'n_infill={n_infill_} (real evals)  inner_pop_size={inner_ps} (surrogate evals)')
        rng = np.random.RandomState(seed)
        surrogates = [
            RFR(20, seed=rng.randint(0, 2**31 - 1)) if samos_type == 'rfr'
            else XGBoost(100, seed=rng.randint(0, 2**31 - 1))
            for _ in range(len(predict_obj))
        ]
        factory = lambda surrs, _ro=real_obj, _db=bench_db: SurrogateProblem101(surrs, _ro, _db)
        algorithm = SAMOS(
            sampling=sampling, surrogates=surrogates,
            surrogate_problem_factory=factory,
            crossover=xover, mutation=mut,
            n_doe=n_doe_, n_infill=n_infill_, n_gen_inner=n_gen_inner,
            ga_pop_size=inner_ps, warm_start_ratio=warm_start_ratio,
            use_subset_selection=False,
            eliminate_duplicates=elim_dupes, dedup_key_fn=dedup_key_fn,
        )

    elif method == 'samos-ssa':
        n_doe_   = n_doe    if n_doe    is not None else pop_size
        n_infill_= n_infill if n_infill is not None else pop_size
        inner_ps = inner_pop_size if inner_pop_size is not None else pop_size * 10
        print(f'  [SAMOS-SSA] n_doe={n_doe_}  n_infill={n_infill_}  inner_pop_size={inner_ps}')
        algorithm = SAMOSSA(
            sampling=sampling, crossover=xover, mutation=mut,
            n_doe=n_doe_, n_infill=n_infill_, n_gen_inner=n_gen_inner,
            ga_pop_size=inner_ps, warm_start_ratio=warm_start_ratio,
            use_subset_selection=False,
            eliminate_duplicates=elim_dupes, dedup_key_fn=dedup_key_fn,
        )

    elif method.startswith('gpsaf-') or method.startswith('ssa-nsga2-'):
        algo_family    = 'gpsaf' if method.startswith('gpsaf-') else 'ssa-nsga2'
        surrogate_type = method[len('gpsaf-'):] if algo_family == 'gpsaf' else method[len('ssa-nsga2-'):]
        if surrogate_type not in ('default', 'rfr', 'xgb'):
            raise ValueError(f'Unknown surrogate type {surrogate_type!r} in {method!r}')
        n_doe_    = n_doe    if n_doe    is not None else pop_size
        n_infill_ = n_infill if n_infill is not None else pop_size
        inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10
        if surrogate_type in ('rfr', 'xgb'):
            rng = np.random.RandomState(seed)
            sklearn_models = [
                RFR(20, seed=rng.randint(0, 2**31 - 1)) if surrogate_type == 'rfr'
                else XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                for _ in range(2)
            ]
        if algo_family == 'ssa-nsga2':
            algorithm = (SSANSGA2 if surrogate_type == 'default' else SklearnSSANSGA2)(
                **({} if surrogate_type == 'default' else {'sklearn_models': sklearn_models}),
                sampling=sampling, n_infills=n_infill_,
                surr_pop_size=inner_ps, surr_n_gen=n_gen_inner, n_initial_doe=n_doe_,
            )
        else:
            base_algo = NSGA2(pop_size=pop_size, sampling=sampling,
                              crossover=xover, mutation=mut,
                              eliminate_duplicates=elim_dupes)
            kw = {} if surrogate_type == 'default' else {'sklearn_models': sklearn_models}
            algorithm = (GPSAF if surrogate_type == 'default' else SklearnGPSAF)(
                base_algo, **kw,
                n_initial_doe=n_doe_, n_max_infills=n_infill_, beta=n_gen_inner,
            )

    else:
        raise ValueError(f'Unknown method: {method!r}')

    results = minimize(
        problem=problem, algorithm=algorithm,
        termination=('n_gen', n_gen),
        seed=seed, callback=callback,
        save_history=False, verbose=True,
    )

    data = results.algorithm.callback.data
    data['time']      = problem.time
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

    all_methods = ['random', 'nsga2', 'samos-xgb', 
                   'ssa-nsga2-default',
                   'gpsaf-default',
                   'parego',
    ]
    methods = args.methods if args.methods else all_methods

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
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to run (default: all). Choices: '
                             'random, nsga2, samos-rfr, samos-xgb, samos-ssa, mosmac, parego, '
                             'ssa-nsga2-default, ssa-nsga2-rfr, ssa-nsga2-xgb, '
                             'gpsaf-default, gpsaf-rfr, gpsaf-xgb')

    # ─── Search budget parameters ─────────────────────────────────────────
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)),
                        help='Seeds to run (default: 0-9)')
    parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size (default: 20, matching PDNS)')
    parser.add_argument('--n_gen', type=int, default=60,
                        help='Number of generations (default: 60, -> 1000 evals)')
    parser.add_argument('--n_doe', type=int, default=None,
                        help='SAMOS: initial DOE size (default: pop_size)')
    parser.add_argument('--n_infill', type=int, default=None,
                        help='SAMOS: real evaluations per outer generation (default: pop_size)')
    parser.add_argument('--n_gen_inner', type=int, default=20,
                        help='SAMOS: inner NSGA-II generations (default: 20)')
    parser.add_argument('--inner_pop_size', type=int, default=None,
                        help='SAMOS: inner NSGA-II population size (default: pop_size * 10)')
    parser.add_argument('--warm_start_ratio', type=float, default=1.0,
                        help='SAMOS: fraction of inner pop warm-started from best archive (default: 1.0)')
    parser.add_argument('--predict_obj', type=str, nargs='+', default=['val_err_12'],
                        help='SAMOS: objectives approximated by surrogates (default: val_err_12)')
    parser.add_argument('--real_obj', type=str, nargs='*', default=['n_params'],
                        help='SAMOS: objectives evaluated exactly in inner loop (default: n_params)')
    parser.add_argument('--elim_dupes', choices=['arch_str', 'pymoo_default'],
                        default='arch_str',
                        help='Duplicate elimination strategy: arch_str (canonical ModelSpec hash, '
                             'catches phenotypically identical architectures) or '
                             'pymoo_default (raw vector comparison). Default: arch_str')
    parser.add_argument('--experiment_name', type=str, default='nasbench101_extended',
                        help='Experiment name; results are saved to results/<experiment_name>/')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if result file already exists')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
