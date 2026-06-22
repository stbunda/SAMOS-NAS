"""experiment_obj_ga_pred.py --- One (benchmark, problem/pid, predictor, method, seed) run.

Experiment 2: evaluate how well each of the 8 MOO genetic algorithms exploits a
*fixed* surrogate predictor, simulating the inner search of a surrogate-assisted
optimiser. For each run:

  1. Sample ``n_doe`` (=100) random configurations and evaluate their TRUE
     objectives (analytic for WFG, ``true_eval=True`` for EvoXBench).
  2. Fit one surrogate per objective on that DOE for the chosen predictor
     (XGBoost / Random Forest / RBF). The surrogate is trained ONCE and frozen.
  3. Warm-start the GA from the best DOE members and run it for ``n_gen`` (=100)
     generations against the FROZEN surrogate problem (all objectives predicted).
  4. The GA keeps a non-dominated archive in PREDICTED space. Each generation the
     archive's TRUE objectives are recorded; the GA is scored on those true values.

Results directory layout (``<experiment>`` segment is the literal ``ga_obj_pred``):
  WFG:        results/obj_ga_pred/wfg/{n_obj}_obj/{problem}/{predictor}/{method}/ga_obj_pred/seed_{seed}.pkl
  EvoXBench:  results/obj_ga_pred/evoxbench/{suite}/pid{pid}/{predictor}/{method}/ga_obj_pred/seed_{seed}.pkl

Run examples:
  python experiments/obj_ga_pred/experiment_obj_ga_pred.py --benchmark wfg --problem wfg1 --n_obj 2 --predictor rbf --method nsga2 --seed 0
  python experiments/obj_ga_pred/experiment_obj_ga_pred.py --benchmark c10mop --pid 1 --n_obj 2 --predictor rf --method nsga2 --seed 0
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
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from problem.evoxbench.baseline_problem import EvoXBenchProblem
from problem.evoxbench.surrogate_problem import SurrogateProblemEvox
from problem.evoxbench.utils import get_benchmark
from problem.pymoo.benchmark_utils import build_problem, get_pareto_front, default_ref_point
from problem.pymoo.surrogate_problem import SurrogateProblemMOO
from strategy.algorithm.algorithms import RandomGA
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.sampler import EvoxBenchSampler
from strategy.surrogate.models import SURROGATE_MODELS

EVOXBENCH_SUITES = ('c10mop', 'in1kmop')
REF_DIR_METHODS = ('nsga3', 'moead', 'rvea')

# Subdirectory segment for this experiment (kept literal so the layout is fixed).
EXPERIMENT_SUBDIR = 'ga_obj_pred'


# --------- ref_dirs sizing ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _choose_ref_dirs(n_obj: int, target: int, floor: int):
    """Das-Dennis ref_dirs whose count is >= floor and as close as possible to target."""
    best = None  # (distance_to_target, n_partitions, ref_dirs)
    for p in range(1, 60):
        ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=p)
        n = len(ref_dirs)
        if n < floor:
            continue
        dist = abs(n - target)
        cand = (dist, p, ref_dirs)
        if best is None or cand[:2] < best[:2]:
            best = cand
        if n >= target and best is not None and n > target + target:
            break

    if best is None:
        ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=1)
        return ref_dirs
    return best[2]


def effective_pop_size(method, n_obj, n_var, pop_size, pop_size_floor):
    """Population size ``build_algorithm`` will use (ref-dir methods use len(ref_dirs))."""
    if method in REF_DIR_METHODS:
        return len(_choose_ref_dirs(n_obj, target=n_var, floor=pop_size_floor))
    return pop_size


# --------- algorithm factory ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_algorithm(method, n_obj, n_var, pop_size, pop_size_floor, sampling,
                    crossover, mutation, elim):
    """Map a method name to a configured pymoo algorithm instance.

    ``sampling`` may be a pymoo Sampling operator or an (X-only) Population used to
    warm-start the GA. Operators are pre-built by the caller to be problem-appropriate.
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


# --------- surrogate factory ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_surrogate(predictor_cfg, seed):
    """Instantiate one surrogate model from a config entry.

    ``predictor_cfg`` is a dict like
    ``{name, label, builder, params}`` where ``builder`` is a key in
    ``strategy.surrogate.models.SURROGATE_MODELS``. Tree models accept a ``seed``;
    RBF models do not, so we fall back to params-only construction.
    """
    builder = predictor_cfg['builder']
    params = dict(predictor_cfg.get('params', {}))
    if builder not in SURROGATE_MODELS or SURROGATE_MODELS[builder] is None:
        raise ValueError(
            f"Unknown / unavailable surrogate builder {builder!r}. "
            f"Available: {[k for k, v in SURROGATE_MODELS.items() if v is not None]}"
        )
    cls = SURROGATE_MODELS[builder]
    try:
        return cls(seed=seed, **params)
    except TypeError:
        # Models without a seed kwarg (e.g. RBF variants).
        return cls(**params)


def _predictor_cfg(config, name):
    for p in config['predictors']:
        if p['name'] == name:
            return p
    raise ValueError(
        f"Predictor {name!r} not in config. "
        f"Available: {[p['name'] for p in config['predictors']]}"
    )


# --------- callback ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ObjGAPredCallback(Callback):
    """Per-generation callback maintaining a cumulative non-dominated archive.

    The archive is maintained in PREDICTED objective space (the surrogate output
    that guides the search). Each generation the archive's TRUE objectives are
    recomputed and stored separately in ``test_obj_archive`` --- these are the values
    the GA is ultimately scored on.

    WFG (continuous): true objectives come from the analytic real problem; HV / IGD+
    are computed live against the analytic Pareto front + reference point.

    EvoXBench (discrete): true objectives come from ``benchmark.evaluate(true_eval=True)``;
    indicators are left empty ({}) and filled post-hoc by the analysis pipeline
    against a pooled Pareto approximation.

    One entry is appended to every archive / indicators list at every generation, so
    each list has length == n_gen.
    """

    def __init__(self, is_evoxbench, benchmark=None, real_problem=None, no_norm=False,
                 pareto_front=None, ref_point=None, n_obj=None):
        super().__init__()
        self.is_evoxbench = is_evoxbench
        self.benchmark = benchmark
        self.real_problem = real_problem
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
        obj_pop = algorithm.pop.get('F')  # PREDICTED objectives (surrogate output)

        # Carry the cumulative (predicted-space) archive forward across generations.
        var_arch = list(self.data['var_archive'][-1]) if self.data['var_archive'] else []
        obj_arch = list(self.data['obj_archive'][-1]) if self.data['obj_archive'] else []

        for var_ind, obj_ind in zip(var_pop, obj_pop):
            var_arch, obj_arch = self._update_archive(var_arch, obj_arch, var_ind, obj_ind)

        n_var = algorithm.problem.n_var
        var_arch_arr = np.array(var_arch) if var_arch else np.empty((0, n_var))
        obj_arch_arr = np.array(obj_arch) if obj_arch else np.empty((0, self.n_obj))

        # True objectives of the predicted-ND archive members -> scoring basis.
        if self.is_evoxbench:
            test_obj, indicators = self._evoxbench_true(var_arch)
        else:
            test_obj, indicators = self._wfg_true(var_arch_arr)

        self.data['var_archive'].append(var_arch_arr)
        self.data['obj_archive'].append(obj_arch_arr)
        self.data['test_obj_archive'].append(test_obj)
        self.data['indicators'].append(indicators)

    def _wfg_true(self, var_arch_arr):
        """True analytic objectives of the archive + live HV / IGD+ on them."""
        if len(var_arch_arr) == 0:
            return np.empty((0, self.n_obj)), {'hv': 0.0, 'igd_plus': float('inf')}
        true_F = np.asarray(
            self.real_problem.evaluate(var_arch_arr, return_values_of=['F'])
        )
        indicators = {
            'hv': float(self._hv_ind(true_F)),
            'igd_plus': float(self._igd_ind(true_F)),
        }
        return true_F, indicators

    def _evoxbench_true(self, var_arch):
        """True objectives of the archive (EvoXBench); indicators filled post-hoc."""
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
        # Indicators filled post-hoc by analysis against a pooled Pareto approx.
        return test_obj, {}


# --------- path helper ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_out_path(results_root, benchmark, n_obj, problem, pid, predictor, method, seed):
    if benchmark == 'wfg':
        return os.path.join(
            results_root, 'wfg', f'{n_obj}_obj', problem, predictor, method,
            EXPERIMENT_SUBDIR, f'seed_{seed}.pkl',
        )
    # EvoXBench: benchmark is the suite name (c10mop / in1kmop)
    return os.path.join(
        results_root, 'evoxbench', benchmark, f'pid{pid}', predictor, method,
        EXPERIMENT_SUBDIR, f'seed_{seed}.pkl',
    )


# --------- DOE + warm-start helpers ------------------------------------------------------------------------------------------------------------------------------------------------------

def fit_surrogates(predictor_cfg, X_doe, Y_doe, n_obj, seed):
    """Fit one frozen surrogate per objective on the DOE; return (models, train_rmse)."""
    surrogates, train_rmse = [], []
    for k in range(n_obj):
        model = build_surrogate(predictor_cfg, seed=seed + k)
        model.fit(X_doe, Y_doe[:, k])
        pred = np.asarray(model.predict(X_doe)).ravel()
        train_rmse.append(float(np.sqrt(np.mean((pred - Y_doe[:, k]) ** 2))))
        surrogates.append(model)
    return surrogates, train_rmse


def warm_start_population(X_doe, Y_doe, surr_problem, pop_size, sampling):
    """Best ``pop_size`` DOE members by rank+crowding (on TRUE objs), X-only.

    Returned as an X-only Population so the GA re-evaluates them on the surrogate
    problem. Pads with fresh random samples if the DOE is smaller than pop_size.
    """
    doe_pop = Population.new('X', X_doe, 'F', Y_doe)
    top = RankAndCrowding().do(surr_problem, doe_pop, n_survive=min(pop_size, len(doe_pop)))
    warm_X = top.get('X')
    if len(warm_X) < pop_size:
        extra = sampling.do(surr_problem, pop_size - len(warm_X)).get('X')
        warm_X = np.vstack([warm_X, extra])
    return Population.new('X', warm_X)


# --------- main ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results_root = args.results_root or config['results_root']
    n_gen = config['n_gen']
    n_doe = config['n_doe']
    pop_size_floor = config['pop_size_floor']
    predictor_cfg = _predictor_cfg(config, args.predictor)

    is_evoxbench = args.benchmark in EVOXBENCH_SUITES

    # ------ resolve output path + skip logic ---------------------------------------------------------------------------------------------------------------------
    out_path = build_out_path(
        results_root, args.benchmark, args.n_obj,
        args.problem, args.pid, args.predictor, args.method, args.seed,
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
        problem_label = args.problem

    # ------ DOE: sample n_doe configs, evaluate TRUE objectives ---------------------------------------------------------------
    X_doe = sampling.do(problem, n_doe).get('X')
    if is_evoxbench:
        X_int = np.round(X_doe).astype(int)
        Y_doe = benchmark.evaluate(X_int, true_eval=True)
        if not benchmark.normalized_objectives:
            Y_doe = benchmark.normalize(Y_doe)
    else:
        Y_doe = np.asarray(problem.evaluate(X_doe, return_values_of=['F']))

    # Keep only DOE rows with finite objectives (invalid NB-101 archs etc.).
    finite = np.isfinite(Y_doe).all(axis=1)
    X_doe, Y_doe = X_doe[finite], Y_doe[finite]
    if len(X_doe) < n_obj + 1:
        raise RuntimeError(
            f'DOE yielded only {len(X_doe)} finite samples --- too few to fit surrogates.'
        )

    # ------ fit one frozen surrogate per objective ---------------------------------------------------------------------------------------------------
    surrogates, train_rmse = fit_surrogates(predictor_cfg, X_doe, Y_doe, n_obj, args.seed)

    # ------ inner surrogate problem (all objectives predicted) ------------------------------------------------------------------
    if is_evoxbench:
        surr_problem = SurrogateProblemEvox(
            surrogates=surrogates,
            predict_obj_indices=list(range(n_obj)),
            real_obj_indices=[],
            benchmark=benchmark,
        )
    else:
        surr_problem = SurrogateProblemMOO(
            surrogates=surrogates, n_var=n_var, xl=problem.xl, xu=problem.xu,
        )

    # ------ population size + warm-started algorithm ------------------------------------------------------------------------------------------------
    # RandomGA re-samples fresh candidates every generation, so it takes the raw
    # sampling operator (warm-starting a pure random search is meaningless); every
    # other GA is warm-started from the best DOE members.
    pop_size = max(n_var, pop_size_floor)
    pop_size = effective_pop_size(args.method, n_obj, n_var, pop_size, pop_size_floor)
    if args.method == 'random':
        ga_init = sampling
    else:
        ga_init = warm_start_population(X_doe, Y_doe, surr_problem, pop_size, sampling)
    algorithm, pop_size = build_algorithm(
        args.method, n_obj, n_var, pop_size, pop_size_floor,
        ga_init, crossover, mutation, elim,
    )

    # ------ callback ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if is_evoxbench:
        callback = ObjGAPredCallback(
            is_evoxbench=True, benchmark=benchmark, no_norm=False, n_obj=n_obj,
        )
    else:
        pf = get_pareto_front(problem, n_obj)
        ref_point = default_ref_point(args.problem, n_obj)
        callback = ObjGAPredCallback(
            is_evoxbench=False, real_problem=problem, pareto_front=pf,
            ref_point=ref_point, n_obj=n_obj,
        )

    print(
        f'[RUN] benchmark={args.benchmark} problem={problem_label} '
        f'predictor={args.predictor} method={args.method} seed={args.seed} '
        f'n_obj={n_obj} n_var={n_var} pop_size={pop_size} n_gen={n_gen} '
        f'n_doe={len(X_doe)} train_rmse={[round(r, 4) for r in train_rmse]}'
    )

    # ------ run GA against the frozen surrogate ---------------------------------------------------------------------------------------------------------------
    t0 = time.time()
    minimize(
        surr_problem,
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
        'predictor': args.predictor,
        'seed': args.seed,
        'benchmark': args.benchmark,
        'problem': args.problem if not is_evoxbench else None,
        'pid': args.pid if is_evoxbench else None,
        'n_obj': n_obj,
        'n_var': n_var,
        'pop_size': pop_size,
        'n_gen': n_gen,
        'n_doe': len(X_doe),
        'predictor_builder': predictor_cfg['builder'],
        'predictor_params': predictor_cfg.get('params', {}),
        'train_rmse': train_rmse,
    }
    data = {
        'var_archive': callback.data['var_archive'],
        'obj_archive': callback.data['obj_archive'],        # PREDICTED objectives
        'test_obj_archive': callback.data['test_obj_archive'],  # TRUE objectives
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
        description='Single GA run for Experiment 2 (predictor-guided search).'
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
    parser.add_argument('--predictor', type=str, required=True,
                        help='Predictor key from the config (e.g. xgboost, rf, rbf).')
    parser.add_argument('--method', type=str, required=True,
                        help='Algorithm key, e.g. nsga2, nsga3, moead, rvea, '
                             'sms-emoa, age-moea, age-moea2, random.')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'config', 'experiment_obj_ga_pred.yaml'))
    parser.add_argument('--results_root', type=str, default=None,
                        help='Override results_root from the config.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if the result pickle already exists.')

    arguments = parser.parse_args()
    sys.exit(main(arguments))
