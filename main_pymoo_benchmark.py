"""main_pymoo_benchmark.py — Test Random, NSGA-II, SAMOS, and MOSMAC on pymoo MOO benchmarks.

Supported benchmarks: WFG1-9, ZDT1-6, DTLZ1-7 (and any other pymoo get_problem() target).

Results are saved to:
  results/moo_benchmark/<problem>/<budget_folder>/<method>/seed_<seed>.pkl

Run examples:
  python main_pymoo_benchmark.py --problem wfg1
  python main_pymoo_benchmark.py --problem wfg1 --methods random nsga2 samos-xgb mosmac
  python main_pymoo_benchmark.py --problem wfg2 --n_obj 3 --methods nsga2 samos-xgb mosmac
"""

import argparse
import os
import pickle
import random
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from analysis.plotter import plot_results
from problem.pymoo.benchmark_utils import build_problem, get_pareto_front, default_ref_point
from problem.pymoo.surrogate_problem import SurrogateProblemMOO
from strategy.algorithm.algorithms import RandomGA
from strategy.algorithm.gpsaf import GPSAF, SklearnGPSAF
from strategy.algorithm.ssansga2 import SSANSGA2, SklearnSSANSGA2
from strategy.algorithm.cobra import run_cobra
from strategy.algorithm.parego import run_parego
from strategy.callbacks import PymooBenchmarkCallback
from strategy.surrogate.models import RFR, XGBoost
from strategy.surrogate.samos_minimal import SAMOSMinimal as SAMOS
from strategy.surrogate.samos_ssa import SAMOSSA

# ─── MOSMAC on continuous benchmarks ─────────────────────────────────────────

def _mosmac_run(
    problem_name: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    n_obj: int = 2,
    n_var: int = None,
) -> dict:
    """Run SMAC3 MultiObjectiveFacade on a continuous pymoo benchmark.

    Budget is pop_size × n_gen evaluations to match the other methods.
    Returns the standard data dict with indicators / obj_pop / var_pop.
    """
    from ConfigSpace import ConfigurationSpace, Float as CSFloat
    from smac import Scenario
    from smac.facade.multi_objective_facade import MultiObjectiveFacade as MOFacade

    _prob     = build_problem(problem_name, n_obj, n_var)
    pf        = get_pareto_front(_prob, _prob.n_obj)
    ref_point = default_ref_point(problem_name, _prob.n_obj)
    hv_ind    = HV(ref_point=ref_point)
    igd_ind   = IGDPlus(pf)

    n_trials  = pop_size * n_gen
    obj_names = [f'obj{i}' for i in range(_prob.n_obj)]

    # ConfigSpace: one Float hyperparameter per decision variable
    cs = ConfigurationSpace(seed=seed)
    cs.add([
        CSFloat(f'x{i}', (float(_prob.xl[i]), float(_prob.xu[i])))
        for i in range(_prob.n_var)
    ])

    def target_fn(config, seed=0):
        x = np.array([config[f'x{i}'] for i in range(_prob.n_var)])
        out = {}
        _prob._evaluate(x.reshape(1, -1), out)
        F = out['F'][0]
        return {name: float(F[j]) for j, name in enumerate(obj_names)}

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

    # Reconstruct evaluation order from runhistory.
    rh = smac.runhistory
    _internal = getattr(rh, 'data', None) or getattr(rh, '_data', {})
    sorted_trials = sorted(_internal.items(), key=lambda kv: kv[1].starttime)
    # config_id → Configuration mapping: try public then private attribute
    _id2cfg = getattr(rh, 'ids_config', None) or getattr(rh, '_ids_config', {})
    all_X_arr = np.array([
        [_id2cfg[k.config_id][f'x{i}'] for i in range(_prob.n_var)]
        for k, _ in sorted_trials
    ])
    all_F_arr = np.array([
        list(v.cost) if hasattr(v.cost, '__iter__') else [v.cost]
        for _, v in sorted_trials
    ])

    # Build per-"generation" data: one entry per pop_size evaluations
    indicators, obj_pop, var_pop = [], [], []
    for step in range(n_gen):
        lo        = step * pop_size
        hi        = (step + 1) * pop_size
        F_so_far  = all_F_arr[:hi]
        nd_idx    = NonDominatedSorting().do(F_so_far, only_non_dominated_front=True)
        nd_F      = F_so_far[nd_idx]
        indicators.append({
            'hv':       float(hv_ind(nd_F)),
            'igd_plus': float(igd_ind(nd_F)),
        })
        obj_pop.append(all_F_arr[lo:hi].copy())
        var_pop.append(all_X_arr[lo:hi].copy())

    return {'indicators': indicators, 'obj_pop': obj_pop, 'var_pop': var_pop}





# ─── single run ───────────────────────────────────────────────────────────────

def run_single(
    method: str,
    seed: int,
    problem_name: str,
    pop_size: int,
    n_gen: int,
    n_obj: int = 2,
    n_var: int = None,
    n_doe: int = None,
    n_infill: int = None,
    n_gen_inner: int = 20,
    inner_pop_size: int = None,
    warm_start_ratio: float = 1.0,
    proxy_obj_indices: list = None,
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    # Standalone runners manage their own problem construction internally
    if method == 'mosmac':
        return _mosmac_run(problem_name, seed, pop_size, n_gen, n_obj, n_var)
    if method == 'parego':
        return run_parego(problem_name, seed, pop_size, n_gen, n_obj, n_var)
    if method == 'cobra':
        return run_cobra(problem_name, seed, pop_size, n_gen, n_obj, n_var, n_doe, n_infill)

    # ── build problem ─────────────────────────────────────────────────────────
    problem   = build_problem(problem_name, n_obj, n_var)
    pf        = get_pareto_front(problem, problem.n_obj)
    ref_point = default_ref_point(problem_name, problem.n_obj)

    callback  = PymooBenchmarkCallback(pf, ref_point)
    sampling  = FloatRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation  = PM(eta=20)

    # ── algorithm ─────────────────────────────────────────────────────────────
    if method == 'random':
        algorithm = RandomGA(pop_size=pop_size, sampling=sampling)

    elif method == 'nsga2':
        algorithm = NSGA2(
            pop_size=pop_size, sampling=sampling,
            crossover=crossover, mutation=mutation,
        )

    elif method.startswith('samos-ssa'):
        # SAMOS-SSA: SAMOS loop with pysamoo cross-validated surrogates
        # Supports optional i{INT}/g{INT} tokens: samos-ssa-i200-g20
        parts = method.split('-')
        _n_gen_inner = n_gen_inner
        _inner_ps    = inner_pop_size
        for tok in parts[2:]:
            if tok.startswith('i') and tok[1:].isdigit():
                _inner_ps = int(tok[1:])
            elif tok.startswith('g') and tok[1:].isdigit():
                _n_gen_inner = int(tok[1:])
        n_doe_    = n_doe    if n_doe    is not None else pop_size
        n_infill_ = n_infill if n_infill is not None else pop_size
        inner_ps  = _inner_ps if _inner_ps is not None else pop_size * 10

        algorithm = SAMOSSA(
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            n_doe=n_doe_,
            n_infill=n_infill_,
            n_gen_inner=_n_gen_inner,
            ga_pop_size=inner_ps,
            warm_start_ratio=warm_start_ratio,
            use_subset_selection=True,
            eliminate_duplicates=False,
            dedup_key_fn=lambda x: tuple(np.round(x, 4).tolist()),
        )

    elif method.startswith('samos-'):
        parts = method.split('-')
        samos_type   = parts[1]   # 'xgb' or 'rfr'
        # parse optional i{INT} / g{INT} tokens embedded in the method name
        # e.g. 'samos-xgb-i200-g20' → inner_pop_size=200, n_gen_inner=20
        _n_gen_inner = n_gen_inner
        _inner_ps    = inner_pop_size
        for tok in parts[2:]:
            if tok.startswith('i') and tok[1:].isdigit():
                _inner_ps = int(tok[1:])
            elif tok.startswith('g') and tok[1:].isdigit():
                _n_gen_inner = int(tok[1:])
        n_doe_    = n_doe    if n_doe    is not None else pop_size
        n_infill_ = n_infill if n_infill is not None else pop_size
        inner_ps  = _inner_ps if _inner_ps is not None else pop_size * 10

        rng = np.random.RandomState(seed)
        proxy_set = (
            set(proxy_obj_indices) if proxy_obj_indices is not None
            else set(range(problem.n_obj))
        )
        surrogates = [
            (RFR(20, seed=rng.randint(0, 2**31 - 1))
             if samos_type == 'rfr' else
             XGBoost(100, seed=rng.randint(0, 2**31 - 1)))
            if i in proxy_set else None
            for i in range(problem.n_obj)
        ]
        _nv = problem.n_var
        _xl = problem.xl.copy()
        _xu = problem.xu.copy()
        factory = lambda surrs, _n=_nv, _l=_xl, _u=_xu, _rp=problem: (
            SurrogateProblemMOO(surrs, _n, _l, _u, real_problem=_rp)
        )
        algorithm = SAMOS(
            sampling=sampling,
            surrogates=surrogates,
            surrogate_problem_factory=factory,
            crossover=crossover,
            mutation=mutation,
            n_doe=n_doe_,
            n_infill=n_infill_,
            n_gen_inner=_n_gen_inner,
            ga_pop_size=inner_ps,
            warm_start_ratio=warm_start_ratio,
            use_subset_selection=True,
            eliminate_duplicates=False,
            dedup_key_fn=lambda x: tuple(np.round(x, 4).tolist()),
        )

    elif method.startswith('gpsaf-') or method.startswith('ssa-nsga2-'):
        if method.startswith('gpsaf-'):
            algo_family    = 'gpsaf'
            surrogate_type = method[len('gpsaf-'):]
        else:
            algo_family    = 'ssa-nsga2'
            surrogate_type = method[len('ssa-nsga2-'):]

        if surrogate_type not in ('default', 'rfr', 'xgb'):
            raise ValueError(f'Unknown surrogate type {surrogate_type!r} in {method!r}')

        n_doe_    = n_doe    if n_doe    is not None else pop_size
        n_infill_ = n_infill if n_infill is not None else pop_size
        inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

        if surrogate_type in ('rfr', 'xgb'):
            rng = np.random.RandomState(seed)
            sklearn_models = [
                (RFR(20, seed=rng.randint(0, 2**31 - 1))
                 if surrogate_type == 'rfr' else
                 XGBoost(100, seed=rng.randint(0, 2**31 - 1)))
                for _ in range(problem.n_obj)
            ]

        if algo_family == 'ssa-nsga2':
            if surrogate_type == 'default':
                algorithm = SSANSGA2(
                    n_infills=n_infill_,
                    surr_pop_size=inner_ps,
                    surr_n_gen=n_gen_inner,
                    n_initial_doe=n_doe_,
                )
            else:
                algorithm = SklearnSSANSGA2(
                    sklearn_models=sklearn_models,
                    n_infills=n_infill_,
                    surr_pop_size=inner_ps,
                    surr_n_gen=n_gen_inner,
                    n_initial_doe=n_doe_,
                )
        else:  # gpsaf
            base_algo = NSGA2(pop_size=pop_size, crossover=crossover, mutation=mutation)
            if surrogate_type == 'default':
                algorithm = GPSAF(
                    base_algo,
                    n_initial_doe=n_doe_,
                    n_max_infills=n_infill_,
                    beta=n_gen_inner,
                )
            else:
                algorithm = SklearnGPSAF(
                    base_algo,
                    sklearn_models=sklearn_models,
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
    for prob in args.problem:
        print(f'\n{"="*80}\nRunning benchmark on problem: {prob}\n{"="*80}')
        budget_folder = f"B{args.n_gen * args.pop_size}_P{args.pop_size}"
        results_root = os.path.join(
            'results', args.experiment_name, prob, budget_folder
        )

        for method in args.methods:
            save_dir = os.path.join(results_root, method)
            os.makedirs(save_dir, exist_ok=True)

            for seed in args.seeds:
                out_path = os.path.join(save_dir, f'seed_{seed}.pkl')
                if os.path.exists(out_path) and not args.overwrite:
                    print(f'[SKIP] {method}/seed_{seed} already exists')
                    continue

                print(
                    f'\n[RUN] method={method}  seed={seed}  pop={args.pop_size}'
                    f'  n_gen={args.n_gen}  problem={prob}'
                )
                data = run_single(
                    method=method,
                    seed=seed,
                    problem_name=prob,
                    pop_size=args.pop_size,
                    n_gen=args.n_gen,
                    n_obj=args.n_obj,
                    n_var=args.n_var,
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

        # ── plot ──────────────────────────────────────────────────────────────────
        if not args.no_plot:
            _prob      = build_problem(prob, args.n_obj, args.n_var)
            _pf        = get_pareto_front(_prob, _prob.n_obj)
            _rp        = default_ref_point(prob, _prob.n_obj)
            hv_ceiling = float(HV(ref_point=_rp)(_pf))
            print(f'\n  Reference front: {len(_pf)} pts  hv_ceiling={hv_ceiling:.6f}')

            plot_out = os.path.join(results_root, 'moo_hv_igd.png')
            print('Generating HV / IGD+ plot ...')
            plot_results(
                methods=args.methods,
                n_gen=args.n_gen,
                pop_size=args.pop_size,
                hv_ceiling=hv_ceiling,
                out_path=plot_out,
                results_root=results_root,
                title=(
                    f'{prob.upper()}  —  HV / IGD+ trajectories\n'
                    f'(pop={args.pop_size}, {args.n_gen} gens'
                    f' = {args.pop_size * args.n_gen} evals, mean ± std)'
                ),
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MOO benchmark: Random + NSGA-II + SAMOS + MOSMAC on pymoo problems'
    )

    parser.add_argument('--problem', type=str, nargs='+', default=['wfg1'],
                        help='pymoo problem name, e.g. wfg1, wfg2, zdt1, dtlz2 (default: wfg1)')
    parser.add_argument('--n_obj',   type=int, default=2,
                        help='Number of objectives for WFG/DTLZ (ignored for ZDT, default: 2)')
    parser.add_argument('--n_var',   type=int, default=None,
                        help='Override the default number of decision variables')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['random', 'nsga2', 'samos-xgb',],
                        help='Methods: random, nsga2, samos-rfr, samos-xgb, samos-ssa, '
                             'mosmac, parego, cobra, gpsaf-default, gpsaf-rfr, gpsaf-xgb, '
                             'ssa-nsga2-default, ssa-nsga2-rfr, ssa-nsga2-xgb')
    parser.add_argument('--seeds',   type=int, nargs='+', default=list(range(2)))
    parser.add_argument('--pop_size',        type=int,   default=20)
    parser.add_argument('--n_gen',           type=int,   default=50)
    parser.add_argument('--n_doe',           type=int,   default=None,
                        help='SAMOS: initial DOE size (default: pop_size)')
    parser.add_argument('--n_infill',        type=int,   default=None,
                        help='SAMOS: real evaluations per outer generation (default: pop_size)')
    parser.add_argument('--n_gen_inner',     type=int,   default=20,
                        help='SAMOS: inner NSGA-II generations (default: 20)')
    parser.add_argument('--inner_pop_size',  type=int,   default=None,
                        help='SAMOS: inner NSGA-II population size (default: pop_size × 10)')
    parser.add_argument('--warm_start_ratio', type=float, default=1.0,
                        help='SAMOS: warm start ratio (default: 1.0)')
    parser.add_argument('--proxy_obj_indices', type=int, nargs='+', default=None,
                        help='SAMOS: indices of objectives to approximate with a surrogate '
                             '(default: all objectives). E.g. --proxy_obj_indices 0 1')
    parser.add_argument('--experiment_name', type=str,   default='pymoo_benchmark')
    parser.add_argument('--overwrite',       action='store_true')
    parser.add_argument('--no_plot',         action='store_true',
                        help='Skip plot generation (useful for cluster runs)')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
