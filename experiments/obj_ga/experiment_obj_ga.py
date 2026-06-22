"""experiment_obj_ga.py --- One (benchmark, problem/pid, method, seed) GA run.

Experiment 1: compare 8 MOO algorithms across WFG1-9 and EvoXBench C-10 / IN-1K
problems, grouped into 2/3/4+ objective sub-experiments. Each invocation runs a
single (method, seed) on a single problem and saves one pickle.

Results directory layout (``<experiment>`` segment is the literal ``ga_obj``):
  WFG:        results/obj_ga/wfg/{n_obj}_obj/{problem}/{method}/ga_obj/seed_{seed}.pkl
  EvoXBench:  results/obj_ga/evoxbench/{suite}/pid{pid}/{method}/ga_obj/seed_{seed}.pkl

Run examples:
  python experiment_obj_ga.py --benchmark wfg --problem wfg1 --n_obj 2 --method nsga2 --seed 0
  python experiment_obj_ga.py --benchmark c10mop --pid 1 --n_obj 2 --method nsga2 --seed 0
  python experiment_obj_ga.py --benchmark in1kmop --pid 9 --n_obj 4 --method rvea --seed 3 --overwrite
"""

import argparse
import os
import pickle
import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

# Allow running from any CWD: put the repo root (two levels up) on sys.path so the
# root packages (strategy/, problem/, analysis/) import regardless of where this
# script lives or is launched from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import yaml
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from problem.evoxbench.baseline_problem import EvoXBenchProblem
from problem.evoxbench.utils import get_benchmark
from problem.pymoo.benchmark_utils import build_problem, get_pareto_front, default_ref_point
from strategy.algorithm.algorithms import RandomGA
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.sampler import EvoxBenchSampler

EVOXBENCH_SUITES = ('c10mop', 'in1kmop')
REF_DIR_METHODS = ('nsga3', 'moead', 'rvea')

# Subdirectory segment for this experiment (kept literal so the layout is fixed).
EXPERIMENT_SUBDIR = 'ga_obj'


# --------- ref_dirs sizing ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _choose_ref_dirs(n_obj: int, target: int, floor: int):
    """Das-Dennis ref_dirs whose count is >= floor and as close as possible to target.

    Returns the chosen reference direction array. ``len(ref_dirs)`` becomes the
    effective population size for NSGA-III / MOEA/D / RVEA.
    """
    best = None  # (distance_to_target, n_partitions, ref_dirs)
    # Search a generous range of partition counts; for high n_obj len() grows fast.
    for p in range(1, 60):
        ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=p)
        n = len(ref_dirs)
        if n < floor:
            continue
        dist = abs(n - target)
        cand = (dist, p, ref_dirs)
        if best is None or cand[:2] < best[:2]:
            best = cand
        # Once we have a valid candidate and counts only grow, we can stop early
        # after we've passed the target by a comfortable margin.
        if n >= target and best is not None and n > target + target:
            break

    if best is None:
        # Fallback: smallest partitioning that yields at least one direction.
        ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=1)
        return ref_dirs
    return best[2]


# --------- algorithm factory ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_algorithm(method, n_obj, n_var, pop_size, pop_size_floor, sampling,
                    crossover, mutation, elim):
    """Map a method name to a configured pymoo algorithm instance.

    Operators (sampling/crossover/mutation/elim) are pre-built by the caller to be
    problem-appropriate (discrete for EvoXBench, continuous for WFG). ``elim`` is
    the duplicate-elimination operator (or False for continuous problems).
    """
    eliminate_duplicates = elim if elim is not None else False

    if method == 'random':
        return RandomGA(
            pop_size=pop_size,
            sampling=sampling,
            eliminate_duplicates=elim,
        ), pop_size

    if method == 'nsga2':
        return NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates,
        ), pop_size

    if method == 'sms-emoa':
        return SMSEMOA(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates,
        ), pop_size

    if method == 'age-moea':
        return AGEMOEA(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates,
        ), pop_size

    if method == 'age-moea2':
        return AGEMOEA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates,
        ), pop_size

    if method in REF_DIR_METHODS:
        ref_dirs = _choose_ref_dirs(n_obj, target=n_var, floor=pop_size_floor)
        eff_pop_size = len(ref_dirs)

        if method == 'nsga3':
            algo = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=eff_pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=eliminate_duplicates,
            )
        elif method == 'moead':
            # MOEA/D has no eliminate_duplicates / pop_size args; its population
            # size equals len(ref_dirs).
            algo = MOEAD(
                ref_dirs=ref_dirs,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
            )
        else:  # rvea
            algo = RVEA(
                ref_dirs=ref_dirs,
                pop_size=eff_pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=eliminate_duplicates,
            )
        return algo, eff_pop_size

    raise ValueError(f'Unknown method: {method!r}')


# --------- callback ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ObjGACallback(Callback):
    """Per-generation callback maintaining a cumulative non-dominated archive.

    For WFG (continuous) problems, only the (surrogate-free) objective archive is
    relevant; ``test_obj_archive`` mirrors ``obj_archive`` and indicators are
    computed live against the analytic Pareto front + reference point.

    For EvoXBench (discrete) problems, the archive's true (test) objectives are
    recomputed each generation via ``benchmark.evaluate(..., true_eval=True)`` and
    stored separately in ``test_obj_archive``. Indicators are left empty ({}) and
    filled post-hoc by the analysis pipeline against a pooled Pareto approximation.

    One entry is appended to every archive / indicators list at every generation,
    so each list has length == n_gen.
    """

    def __init__(self, is_evoxbench, benchmark=None, no_norm=False,
                 pareto_front=None, ref_point=None, n_obj=None):
        super().__init__()
        self.is_evoxbench = is_evoxbench
        self.benchmark = benchmark
        self.no_norm = no_norm
        self.n_obj = n_obj

        if not is_evoxbench:
            self._hv_ind = HV(ref_point=ref_point)
            self._igd_ind = IGDPlus(pareto_front)

        self.data['var_archive'] = []
        self.data['obj_archive'] = []
        self.data['test_obj_archive'] = []
        self.data['indicators'] = []

    @staticmethod
    def _update_archive(var_arch, obj_arch, var_new, obj_new):
        """Add a candidate to the non-dominated archive (returns new lists)."""
        var_arch = list(var_arch)
        obj_arch = list(obj_arch)

        for obj_existing in obj_arch:
            if all(obj_existing[k] <= obj_new[k] for k in range(len(obj_new))):
                return var_arch, obj_arch  # new solution is dominated -> discard

        dominated = [
            idx for idx, obj_existing in enumerate(obj_arch)
            if all(obj_new[k] <= obj_existing[k] for k in range(len(obj_new)))
            and not all(obj_new[k] == obj_existing[k] for k in range(len(obj_new)))
        ]
        for idx in sorted(dominated, reverse=True):
            var_arch.pop(idx)
            obj_arch.pop(idx)

        var_arch.append(var_new)
        obj_arch.append(obj_new)
        return var_arch, obj_arch

    def notify(self, algorithm):
        var_pop = algorithm.pop.get('X')
        obj_pop = algorithm.pop.get('F')

        # Carry the cumulative archive forward across generations.
        var_arch = list(self.data['var_archive'][-1]) if self.data['var_archive'] else []
        obj_arch = list(self.data['obj_archive'][-1]) if self.data['obj_archive'] else []

        for var_ind, obj_ind in zip(var_pop, obj_pop):
            var_arch, obj_arch = self._update_archive(var_arch, obj_arch, var_ind, obj_ind)

        var_arch_arr = np.array(var_arch) if var_arch else np.empty((0, algorithm.problem.n_var))
        obj_arch_arr = np.array(obj_arch) if obj_arch else np.empty((0, self.n_obj))

        if self.is_evoxbench:
            test_obj_nd, indicators = self._evoxbench_test(var_arch)
        else:
            # WFG: true objectives == surrogate objectives; indicators computed live.
            test_obj_nd = obj_arch_arr
            if len(obj_arch_arr) > 0:
                indicators = {
                    'hv': float(self._hv_ind(obj_arch_arr)),
                    'igd_plus': float(self._igd_ind(obj_arch_arr)),
                }
            else:
                indicators = {'hv': 0.0, 'igd_plus': float('inf')}

        self.data['var_archive'].append(var_arch_arr)
        self.data['obj_archive'].append(obj_arch_arr)
        self.data['test_obj_archive'].append(test_obj_nd)
        self.data['indicators'].append(indicators)

    def _evoxbench_test(self, var_arch):
        """Recompute true objectives of the current archive (EvoXBench only)."""
        n_obj = self.benchmark.evaluator.n_objs
        if not var_arch:
            return np.empty((0, n_obj)), {}

        X_arch = np.array([np.round(v).astype(int) for v in var_arch])
        test_obj = self.benchmark.evaluate(X_arch, true_eval=True)
        if not self.no_norm and not self.benchmark.normalized_objectives:
            test_obj = self.benchmark.normalize(test_obj)
        test_obj = np.where(np.isfinite(test_obj), test_obj, np.nan)
        finite_mask = np.isfinite(test_obj).all(axis=1)
        test_obj = test_obj[finite_mask]
        if len(test_obj) > 0:
            nd_idx = NonDominatedSorting().do(test_obj, only_non_dominated_front=True)
            test_obj_nd = test_obj[nd_idx]
        else:
            test_obj_nd = np.empty((0, n_obj))
        # Indicators filled post-hoc by analysis against a pooled Pareto approx.
        return test_obj_nd, {}


# --------- path helper ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_out_path(results_root, benchmark, n_obj, problem, pid, method, seed):
    if benchmark == 'wfg':
        return os.path.join(
            results_root, 'wfg', f'{n_obj}_obj', problem, method,
            EXPERIMENT_SUBDIR, f'seed_{seed}.pkl',
        )
    # EvoXBench: benchmark is the suite name (c10mop / in1kmop)
    return os.path.join(
        results_root, 'evoxbench', benchmark, f'pid{pid}', method,
        EXPERIMENT_SUBDIR, f'seed_{seed}.pkl',
    )


# --------- main ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results_root = args.results_root or config['results_root']
    n_gen = config['n_gen']
    pop_size_floor = config['pop_size_floor']

    is_evoxbench = args.benchmark in EVOXBENCH_SUITES

    # ------ resolve output path + skip logic ---------------------------------------------------------------------------------------------------------------------
    out_path = build_out_path(
        results_root, args.benchmark, args.n_obj,
        args.problem, args.pid, args.method, args.seed,
    )
    if os.path.exists(out_path) and not args.overwrite:
        print(f'[SKIP] {out_path} already exists (use --overwrite to replace)')
        return 0

    np.random.seed(args.seed)
    random.seed(args.seed)

    # ------ build problem + operators ------------------------------------------------------------------------------------------------------------------------------------------
    if is_evoxbench:
        if args.pid is None:
            raise ValueError('--pid is required for EvoXBench benchmarks')
        benchmark = get_benchmark(args.benchmark, args.pid)
        problem = EvoXBenchProblem(benchmark)
        n_var = problem.n_var
        n_obj = problem.n_obj

        xl = np.asarray(benchmark.search_space.lb, dtype=int)
        xu = np.asarray(benchmark.search_space.ub, dtype=int)
        sampling = EvoxBenchSampler(xl, xu)
        crossover = IntegerUniformCrossover(prob=0.9)
        mutation = IntegerPointMutation(xl, xu)
        elim = IntegerVectorDuplicateElimination()

        callback = ObjGACallback(
            is_evoxbench=True, benchmark=benchmark, no_norm=False, n_obj=n_obj,
        )
        problem_label = f'{args.benchmark}/pid{args.pid}'
    else:
        if args.problem is None:
            raise ValueError('--problem is required for WFG benchmarks')
        wfg_cfg = config['experiments'][_exp_key_for_n_obj(config, args.n_obj)]['wfg']
        n_var = wfg_cfg['n_var']
        problem = build_problem(args.problem, args.n_obj, n_var)
        n_var = problem.n_var
        n_obj = problem.n_obj

        sampling = FloatRandomSampling()
        crossover = SBX(prob=0.9, eta=15)
        mutation = PM(eta=20)
        elim = None  # no duplicate elimination for continuous problems

        pf = get_pareto_front(problem, n_obj)
        ref_point = default_ref_point(args.problem, n_obj)
        callback = ObjGACallback(
            is_evoxbench=False, pareto_front=pf, ref_point=ref_point, n_obj=n_obj,
        )
        problem_label = args.problem

    # ------ population size + algorithm ------------------------------------------------------------------------------------------------------------------------------------
    pop_size = max(n_var, pop_size_floor)
    algorithm, pop_size = build_algorithm(
        args.method, n_obj, n_var, pop_size, pop_size_floor,
        sampling, crossover, mutation, elim,
    )

    print(
        f'[RUN] benchmark={args.benchmark} problem={problem_label} '
        f'method={args.method} seed={args.seed} '
        f'n_obj={n_obj} n_var={n_var} pop_size={pop_size} n_gen={n_gen}'
    )

    # ------ run ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    t0 = time.time()
    minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        callback=callback,
        seed=args.seed,
        verbose=False,
    )
    elapsed = time.time() - t0

    # ------ save ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    run_config = {
        'method': args.method,
        'seed': args.seed,
        'benchmark': args.benchmark,
        'problem': args.problem if not is_evoxbench else None,
        'pid': args.pid if is_evoxbench else None,
        'n_obj': n_obj,
        'n_var': n_var,
        'pop_size': pop_size,
        'n_gen': n_gen,
    }
    data = {
        'var_archive': callback.data['var_archive'],
        'obj_archive': callback.data['obj_archive'],
        'test_obj_archive': callback.data['test_obj_archive'],
        'indicators': callback.data['indicators'],
        'config': run_config,
        'time': elapsed,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'  Saved -> {out_path}  ({elapsed:.1f}s, {len(data["obj_archive"])} gens)')
    return 0


def _exp_key_for_n_obj(config, n_obj):
    """Find the sub-experiment key whose n_obj matches the requested value."""
    for key, block in config['experiments'].items():
        if block['n_obj'] == n_obj:
            return key
    raise ValueError(f'No sub-experiment in config has n_obj={n_obj}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single GA run for Experiment 1 (objectives comparison).'
    )
    parser.add_argument('--benchmark', type=str, required=True,
                        choices=['wfg', 'c10mop', 'in1kmop'],
                        help='Benchmark: wfg (continuous) or an EvoXBench suite.')
    parser.add_argument('--pid', type=int, default=None,
                        help='EvoXBench problem id (required for c10mop/in1kmop).')
    parser.add_argument('--problem', type=str, default=None,
                        help='WFG problem name (e.g. wfg1..wfg9; required for wfg).')
    parser.add_argument('--n_obj', type=int, required=True,
                        help='Number of objectives (2, 3, or 4).')
    parser.add_argument('--method', type=str, required=True,
                        help='Algorithm key, e.g. nsga2, nsga3, moead, rvea, '
                             'sms-emoa, age-moea, age-moea2, random.')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'config', 'experiment_obj_ga.yaml'))
    parser.add_argument('--results_root', type=str, default=None,
                        help='Override results_root from the config.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if the result pickle already exists.')

    arguments = parser.parse_args()
    sys.exit(main(arguments))
