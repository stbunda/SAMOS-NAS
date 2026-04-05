"""main_evoxbench.py — Run SAMOS and baselines on evoxbench NAS benchmarks.

Supported test suites: c10mop (CIFAR-10), in1kmop (ImageNet-1K), citysegmop (Cityscapes).
Each suite + pid fully determines the search space, n_var, n_obj, and objectives.

Results are saved to:
  results/evoxbench/<suite>/pid<pid>/<budget_folder>/<method>/seed_<seed>.pkl

Run examples:
  python main_evoxbench.py --suite c10mop --pid 2 --methods random nsga2 samos-xgb
  python main_evoxbench.py --suite in1kmop --pid 7 --methods samos-xgb --seeds 0 1 2
  python main_evoxbench.py --suite c10mop --pid 2 --methods samos-xgb --proxy_obj_indices 0
"""

import argparse
import os
import pickle
import random
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from problem.evoxbench.utils import get_benchmark
from problem.evoxbench.baseline_problem import EvoXBenchProblem
from problem.evoxbench.surrogate_problem import SurrogateProblemEvox
from problem.evoxbench.callbacks import EvoxBenchCallback
from strategy.algorithm.algorithms import RandomGA
from strategy.algorithm.gpsaf import GPSAF
from strategy.algorithm.parego import run_parego_evoxbench
from strategy.sampler import EvoxBenchSampler
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.surrogate.models import RFR, XGBoost
from strategy.surrogate.samos_minimal import SAMOSMinimal as SAMOS
from strategy.surrogate.samos2 import SAMOS2


# ─── mosmac standalone runner ────────────────────────────────────────────────

def _run_mosmac_evoxbench(
    benchmark,
    callback: EvoxBenchCallback,
    seed: int,
    pop_size: int,
    n_gen: int,
) -> dict:
    """Run SMAC3 MultiObjectiveFacade on an evoxbench integer search space."""
    from ConfigSpace import ConfigurationSpace, Categorical
    from smac import Scenario
    from smac.facade.multi_objective_facade import MultiObjectiveFacade as MOFacade
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    xl       = np.asarray(benchmark.search_space.lb, dtype=int)
    xu       = np.asarray(benchmark.search_space.ub, dtype=int)
    n_var    = len(xl)
    n_obj    = benchmark.evaluator.n_objs
    n_trials = pop_size * n_gen
    obj_names = [f'obj{i}' for i in range(n_obj)]

    cs = ConfigurationSpace(seed=seed)
    cs.add([
        Categorical(f'x{i}', list(range(int(xl[i]), int(xu[i]) + 1)))
        for i in range(n_var)
    ])

    def target_fn(config, seed=0):
        x = np.array([int(config[f'x{i}']) for i in range(n_var)]).reshape(1, -1)
        F = benchmark.evaluate(x, true_eval=False)
        if not benchmark.normalized_objectives:
            F = benchmark.normalize(F)
        F = np.where(np.isfinite(F), F, 1.0)
        return {name: float(F[0, j]) for j, name in enumerate(obj_names)}

    scenario = Scenario(
        configspace=cs,
        objectives=obj_names,
        n_trials=n_trials,
        seed=seed,
        deterministic=True,
        n_workers=1,
    )
    smac = MOFacade(scenario=scenario, target_function=target_fn, overwrite=True)
    smac.optimize()

    # Reconstruct evaluation order from runhistory
    rh       = smac.runhistory
    _data    = getattr(rh, 'data', None) or getattr(rh, '_data', {})
    _id2cfg  = getattr(rh, 'ids_config', None) or getattr(rh, '_ids_config', {})
    sorted_trials = sorted(_data.items(), key=lambda kv: kv[1].starttime)

    all_X = np.array([
        [int(_id2cfg[k.config_id][f'x{i}']) for i in range(n_var)]
        for k, _ in sorted_trials
    ])
    all_F = np.array([
        list(v.cost) if hasattr(v.cost, '__iter__') else [v.cost]
        for _, v in sorted_trials
    ])

    # Build per-generation snapshots mirroring EvoxBenchCallback.data format
    cur_var_arch: list = []
    cur_obj_arch: list = []

    for step in range(n_gen):
        lo = step * pop_size
        hi = min((step + 1) * pop_size, len(all_X))
        if lo >= len(all_X):
            break

        X_batch = all_X[lo:hi]
        F_batch = all_F[lo:hi]

        for var_ind, obj_ind in zip(X_batch, F_batch):
            cur_var_arch, cur_obj_arch = EvoxBenchCallback._update_archive(
                cur_var_arch, cur_obj_arch, var_ind, obj_ind
            )

        if cur_var_arch:
            X_arch   = np.array([np.round(v).astype(int) for v in cur_var_arch])
            test_obj = benchmark.evaluate(X_arch, true_eval=True)
            if not benchmark.normalized_objectives:
                test_obj = benchmark.normalize(test_obj)
            test_obj    = np.where(np.isfinite(test_obj), test_obj, 1.0)
            nd_idx      = NonDominatedSorting().do(test_obj, only_non_dominated_front=True)
            test_obj_nd = test_obj[nd_idx]
        else:
            test_obj_nd = np.empty((0, n_obj))

        if len(test_obj_nd) > 0:
            ind = {
                'hv':       float(callback._hv_ind(test_obj_nd)),
                'igd_plus': float(callback._igd_ind(test_obj_nd))
                            if callback._igd_ind is not None else float('nan'),
            }
        else:
            ind = {'hv': 0.0, 'igd_plus': float('nan')}

        callback.data['var_pop'].append(X_batch)
        callback.data['obj_pop'].append(F_batch)
        callback.data['var_archive'].append(list(cur_var_arch))
        callback.data['obj_archive'].append(list(cur_obj_arch))
        callback.data['test_obj_archive'].append(test_obj_nd)
        callback.data['indicators'].append(ind)

    callback.data['time'] = None
    return callback.data


# ─── single run ───────────────────────────────────────────────────────────────

def run_single(
    suite: str,
    pid: int,
    method: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    n_doe: int = None,
    n_infill: int = None,
    n_gen_inner: int = 20,
    inner_pop_size: int = None,
    warm_start_ratio: float = 0.75,
    proxy_obj_indices: list = None,
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    # ── benchmark (drives everything) ─────────────────────────────────────────
    benchmark  = get_benchmark(suite, pid)
    problem    = EvoXBenchProblem(benchmark)
    callback   = EvoxBenchCallback(benchmark)

    xl = np.asarray(benchmark.search_space.lb, dtype=int)
    xu = np.asarray(benchmark.search_space.ub, dtype=int)

    sampler   = EvoxBenchSampler(xl, xu)
    crossover = IntegerUniformCrossover(prob=0.9)
    mutation  = IntegerPointMutation(xl, xu)
    elim      = IntegerVectorDuplicateElimination()

    # ── standalone runners (manage their own loop) ─────────────────────────────
    if method == 'mosmac':
        return _run_mosmac_evoxbench(benchmark, callback, seed, pop_size, n_gen)

    if method == 'parego':
        n_doe_ = n_doe if n_doe is not None else pop_size
        return run_parego_evoxbench(
            benchmark, problem, callback, sampler,
            seed=seed, pop_size=pop_size, n_gen=n_gen, n_doe=n_doe_,
        )

    # ── algorithm ─────────────────────────────────────────────────────────────
    if method == 'random':
        algorithm = RandomGA(
            pop_size=pop_size,
            sampling=sampler,
            eliminate_duplicates=elim,
        )

    elif method == 'nsga2':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampler,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=elim,
        )

    elif method.startswith('samos-'):
        samos_type = method.split('-')[1]   # 'xgb' or 'rfr'
        if samos_type not in ('xgb', 'rfr'):
            raise ValueError(f'Unknown surrogate type {samos_type!r} in {method!r}')

        n_doe_    = n_doe    if n_doe    is not None else pop_size
        n_infill_ = n_infill if n_infill is not None else pop_size
        inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

        n_obj     = benchmark.evaluator.n_objs
        proxy_set = (
            set(proxy_obj_indices) if proxy_obj_indices is not None
            else set(range(n_obj))
        )
        real_obj_indices    = [i for i in range(n_obj) if i not in proxy_set]
        predict_obj_indices = [i for i in range(n_obj) if i in proxy_set]

        rng = np.random.RandomState(seed)
        surrogates = [
            RFR(20, seed=rng.randint(0, 2**31 - 1))
            if samos_type == 'rfr' else
            XGBoost(100, seed=rng.randint(0, 2**31 - 1))
            for _ in predict_obj_indices
        ]

        _poi = predict_obj_indices
        _roi = real_obj_indices
        _bm  = benchmark
        factory = lambda surrs, _p=_poi, _r=_roi, _b=_bm: (
            SurrogateProblemEvox(surrs, _p, _r, _b)
        )

        algorithm = SAMOS(
            sampling=sampler,
            surrogates=surrogates,
            surrogate_problem_factory=factory,
            crossover=crossover,
            mutation=mutation,
            n_doe=n_doe_,
            n_infill=n_infill_,
            n_gen_inner=n_gen_inner,
            ga_pop_size=inner_ps,
            warm_start_ratio=warm_start_ratio,
            use_subset_selection=True,
            eliminate_duplicates=elim,
            dedup_key_fn=elim.key,
        )

    elif method == 'samos2':
        n_doe_    = n_doe    if n_doe    is not None else pop_size
        n_infill_ = n_infill if n_infill is not None else pop_size

        n_obj  = benchmark.evaluator.n_objs
        rng    = np.random.RandomState(seed)
        surrogates = [
            XGBoost(100, seed=rng.randint(0, 2**31 - 1))
            for _ in range(n_obj)
        ]
        _poi = list(range(n_obj))
        _roi: list = []
        _bm  = benchmark
        factory = lambda surrs, _p=_poi, _r=_roi, _b=_bm: (
            SurrogateProblemEvox(surrs, _p, _r, _b)
        )
        algorithm = SAMOS2(
            sampling=sampler,
            surrogates=surrogates,
            surrogate_problem_factory=factory,
            crossover=crossover,
            mutation=mutation,
            n_doe=n_doe_,
            n_infill=n_infill_,
            eliminate_duplicates=elim,
            dedup_key_fn=elim.key,
        )

    elif method == 'gpsaf-default':
        n_doe_    = n_doe    if n_doe    is not None else pop_size
        n_infill_ = n_infill if n_infill is not None else pop_size
        base_algo = NSGA2(
            pop_size=pop_size,
            sampling=sampler,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=elim,
        )
        algorithm = GPSAF(
            base_algo,
            n_initial_doe=n_doe_,
            n_max_infills=n_infill_,
            beta=n_gen_inner,
        )

    else:
        raise ValueError(f'Unknown method: {method!r}')

    results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=('n_gen', n_gen),
        seed=seed,
        callback=callback,
        save_history=False,
        verbose=True,
    )
    return results.algorithm.callback.data


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    budget_folder = f"B{args.n_gen * args.pop_size}_P{args.pop_size}"

    for pid in args.pids:
        results_root = os.path.join(
            'results', 'evoxbench', args.suite, f'pid{pid}', budget_folder
        )

        for method in args.methods:
            save_dir = os.path.join(results_root, method)
            os.makedirs(save_dir, exist_ok=True)

            for seed in args.seeds:
                out_path = os.path.join(save_dir, f'seed_{seed}.pkl')
                if os.path.exists(out_path) and not args.overwrite:
                    print(f'[SKIP] pid{pid}/{method}/seed_{seed} already exists')
                    continue

                print(
                    f'\n[RUN] suite={args.suite}  pid={pid}'
                    f'  method={method}  seed={seed}'
                    f'  pop={args.pop_size}  n_gen={args.n_gen}'
                )
                data = run_single(
                    suite=args.suite,
                    pid=pid,
                    method=method,
                    seed=seed,
                    pop_size=args.pop_size,
                    n_gen=args.n_gen,
                    n_doe=args.n_doe,
                    n_infill=args.n_infill,
                    n_gen_inner=args.n_gen_inner,
                    inner_pop_size=args.inner_pop_size,
                    warm_start_ratio=args.warm_start_ratio,
                    proxy_obj_indices=args.proxy_obj_indices,
                )
                with open(out_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f'  Saved -> {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EvoXBench NAS: Random + NSGA-II + SAMOS'
    )

    parser.add_argument('--suite', type=str, required=True,
                        choices=['c10mop', 'in1kmop', 'citysegmop'],
                        help='EvoXBench test suite')
    parser.add_argument('--pids', type=int, nargs='+', required=True,
                        help='Problem ID(s) within the suite (e.g. --pids 1 2 3 or --pids 7)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['random', 'nsga2', 'samos-xgb'],
                        help='Methods: random, nsga2, samos-xgb, samos-rfr, samos2, '
                             'parego, gpsaf-default, mosmac')
    parser.add_argument('--seeds',          type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--pop_size',       type=int, default=20)
    parser.add_argument('--n_gen',          type=int, default=50)
    parser.add_argument('--n_doe',          type=int, default=None,
                        help='SAMOS: initial DOE size (default: pop_size)')
    parser.add_argument('--n_infill',       type=int, default=None,
                        help='SAMOS: real evaluations per outer generation (default: pop_size)')
    parser.add_argument('--n_gen_inner',    type=int, default=20,
                        help='SAMOS: inner NSGA-II generations (default: 20)')
    parser.add_argument('--inner_pop_size', type=int, default=None,
                        help='SAMOS: inner NSGA-II population size (default: pop_size × 10)')
    parser.add_argument('--warm_start_ratio', type=float, default=0.75,
                        help='SAMOS: warm-start ratio for the inner NSGA-II (default: 0.75)')
    parser.add_argument('--proxy_obj_indices', type=int, nargs='+', default=None,
                        help='SAMOS: objective column indices to approximate with surrogates '
                             '(default: all). E.g. --proxy_obj_indices 0')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Override the default budget_folder name in the results path')
    parser.add_argument('--overwrite',      action='store_true',
                        help='Re-run even if result file already exists')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
