"""main_pymoo_benchmark.py — Test Random, NSGA-II, SAMOS, and MOSMAC on pymoo MOO benchmarks.

Supported benchmarks: ZDT1-6, DTLZ1-7 (and any other pymoo get_problem() target).

Results are saved to:
  results/moo_benchmark/<problem>/<budget_folder>/<method>/seed_<seed>.pkl

Run examples:
  python main_pymoo_benchmark.py --problem zdt1
  python main_pymoo_benchmark.py --problem zdt1 --methods random nsga2 samos-xgb mosmac
  python main_pymoo_benchmark.py --problem dtlz2 --n_obj 3 --methods nsga2 samos-xgb mosmac
"""

import argparse
import os
import pickle
import random

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from analysis.plotter import plot_results
from problem.pymoo.surrogate_problem import SurrogateProblemMOO
from strategy.algorithm.algorithms import RandomGA
from strategy.callbacks import MOOBenchmarkCallback
from strategy.surrogate.models import RFR, XGBoost
from strategy.surrogate.samos_minimal import SAMOSMinimal as SAMOS


# ─── helpers ──────────────────────────────────────────────────────────────────

def _get_pareto_front(problem, n_obj: int) -> np.ndarray:
    """Retrieve the reference Pareto front from a pymoo problem."""
    try:
        pf = problem.pareto_front(n_points=1000)
        if pf is not None:
            return pf
    except TypeError:
        pass

    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=12)
    try:
        pf = problem.pareto_front(ref_dirs)
        if pf is not None:
            return pf
    except Exception:
        pass

    pf = problem.pareto_front()
    if pf is None:
        raise RuntimeError(f'Cannot obtain Pareto front for {problem}.')
    return pf


def _default_ref_point(problem_name: str, n_obj: int) -> np.ndarray:
    """Heuristic reference point for HV (slightly dominates the full Pareto front)."""
    if problem_name.lower().startswith('dtlz1'):
        return np.full(n_obj, 0.6)
    return np.full(n_obj, 1.1)


def _build_problem(problem_name: str, n_obj: int, n_var: int):
    kwargs = {}
    if n_var is not None:
        kwargs['n_var'] = n_var
    if not problem_name.lower().startswith('zdt'):
        kwargs['n_obj'] = n_obj
    return get_problem(problem_name, **kwargs)


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

    _prob     = _build_problem(problem_name, n_obj, n_var)
    pf        = _get_pareto_front(_prob, _prob.n_obj)
    ref_point = _default_ref_point(problem_name, _prob.n_obj)
    hv_ind    = HV(ref_point=ref_point)
    igd_ind   = IGDPlus(pf)

    n_trials  = pop_size * n_gen
    obj_names = [f'obj{i}' for i in range(_prob.n_obj)]

    # ConfigSpace: one Float hyperparameter per decision variable
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        CSFloat(f'x{i}', (float(_prob.xl[i]), float(_prob.xu[i])))
        for i in range(_prob.n_var)
    ])

    # Target function — records every evaluation in order
    all_X: list = []
    all_F: list = []

    def target_fn(config, seed=0):
        x = np.array([config[f'x{i}'] for i in range(_prob.n_var)])
        out = {}
        _prob._evaluate(x.reshape(1, -1), out)
        F = out['F'][0]
        all_X.append(x.copy())
        all_F.append(F.copy())
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

    all_X_arr = np.array(all_X)   # (n_trials, n_var)
    all_F_arr = np.array(all_F)   # (n_trials, n_obj)

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
    warm_start_ratio: float = 0.75,
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    # MOSMAC manages its own problem construction internally
    if method == 'mosmac':
        return _mosmac_run(problem_name, seed, pop_size, n_gen, n_obj, n_var)

    # ── build problem ─────────────────────────────────────────────────────────
    problem   = _build_problem(problem_name, n_obj, n_var)
    pf        = _get_pareto_front(problem, problem.n_obj)
    ref_point = _default_ref_point(problem_name, problem.n_obj)

    callback  = MOOBenchmarkCallback(pf, ref_point)
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

    elif method.startswith('samos-'):
        samos_type = method.split('-', 1)[1]
        n_doe_    = n_doe          if n_doe          is not None else pop_size
        n_infill_ = n_infill       if n_infill       is not None else pop_size
        inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

        rng = np.random.RandomState(seed)
        surrogates = [
            (RFR(20, seed=rng.randint(0, 2**31 - 1))
             if samos_type == 'rfr' else
             XGBoost(100, seed=rng.randint(0, 2**31 - 1)))
            for _ in range(problem.n_obj)
        ]
        _nv = problem.n_var
        _xl = problem.xl.copy()
        _xu = problem.xu.copy()
        factory = lambda surrs, _n=_nv, _l=_xl, _u=_xu: (
            SurrogateProblemMOO(surrs, _n, _l, _u)
        )
        algorithm = SAMOS(
            sampling=sampling,
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
            eliminate_duplicates=False,
            dedup_key_fn=lambda x: tuple(np.round(x, 4).tolist()),
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
    n_infill = args.n_infill if args.n_infill is not None else args.pop_size
    n_doe    = args.n_doe    if args.n_doe    is not None else args.pop_size
    budget_folder = (
        f"G{args.n_gen}_GI{args.n_gen_inner}_P{args.pop_size}_I{n_infill}_D{n_doe}"
    )
    results_root = os.path.join(
        'results', args.experiment_name, args.problem, budget_folder
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
                f'  n_gen={args.n_gen}  problem={args.problem}'
            )
            data = run_single(
                method=method,
                seed=seed,
                problem_name=args.problem,
                pop_size=args.pop_size,
                n_gen=args.n_gen,
                n_obj=args.n_obj,
                n_var=args.n_var,
                n_doe=args.n_doe,
                n_infill=args.n_infill,
                n_gen_inner=args.n_gen_inner,
                inner_pop_size=args.inner_pop_size,
                warm_start_ratio=args.warm_start_ratio,
            )
            with open(out_path, 'wb') as f:
                pickle.dump(data, f)
            print(f'  Saved -> {out_path}')

    # ── plot ──────────────────────────────────────────────────────────────────
    _prob      = _build_problem(args.problem, args.n_obj, args.n_var)
    _pf        = _get_pareto_front(_prob, _prob.n_obj)
    _rp        = _default_ref_point(args.problem, _prob.n_obj)
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
            f'{args.problem.upper()}  —  HV / IGD+ trajectories\n'
            f'(pop={args.pop_size}, {args.n_gen} gens'
            f' = {args.pop_size * args.n_gen} evals, mean ± std)'
        ),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MOO benchmark: Random + NSGA-II + SAMOS + MOSMAC on pymoo problems'
    )

    parser.add_argument('--problem', type=str, default='zdt1',
                        help='pymoo problem name, e.g. zdt1, zdt2, dtlz2 (default: zdt1)')
    parser.add_argument('--n_obj',   type=int, default=2,
                        help='Number of objectives for DTLZ (ignored for ZDT, default: 2)')
    parser.add_argument('--n_var',   type=int, default=None,
                        help='Override the default number of decision variables')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['random', 'nsga2', 'samos-xgb', 'mosmac'],
                        help='Methods: random, nsga2, samos-rfr, samos-xgb, mosmac')
    parser.add_argument('--seeds',   type=int, nargs='+', default=list(range(5)))
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
    parser.add_argument('--warm_start_ratio', type=float, default=0.75)
    parser.add_argument('--experiment_name', type=str,   default='moo_benchmark')
    parser.add_argument('--overwrite',       action='store_true')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
