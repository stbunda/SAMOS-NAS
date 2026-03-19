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
import time

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.mutation.pm import mut_pm
from pymoo.optimize import minimize

from strategy.genetics.nasbench101_lib.model_spec import ModelSpec as _ModelSpec101
from strategy.algorithm.algorithms import RandomGA
from strategy.surrogate.models import RFR, XGBoost
from strategy.surrogate.samos_minimal import SAMOSMinimal as SAMOS

# ─── constants ────────────────────────────────────────────────────────────────
_OP_LIST   = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
N_VAR      = 26          # 5 op genes + 21 edge genes
N_OPS      = 5
N_EDGES    = 21
MIN_PARAMS = 227_274
MAX_PARAMS = 49_979_274

XL = np.array([0] * N_OPS  + [0] * N_EDGES, dtype=int)
XU = np.array([2] * N_OPS  + [1] * N_EDGES, dtype=int)

RESULTS_ROOT = 'results/nasbench101_baseline'
DATA_FILE    = 'problem/data/data_nasbench101.pkl'


# ─── helpers to convert NASBENCH101 genome to integer vector ──────────────────

def _vec_to_arch_str(vec: np.ndarray, bench_db: dict) -> 'str | None':
    """Integer vector -> arch hash (MD5 of canonical ModelSpec)."""
    ops_idx = vec[:N_OPS]
    edges   = vec[N_OPS:]
    ops = ['input'] + [_OP_LIST[int(o)] for o in ops_idx] + ['output']
    mat = np.zeros((7, 7), dtype=int)
    k = 0
    for i in range(7):
        for j in range(i + 1, 7):
            mat[i, j] = int(edges[k])
            k += 1
    spec = _ModelSpec101(matrix=mat, ops=ops)
    if not spec.valid_spec:
        return None
    return spec.hash_spec(_OP_LIST)


def _is_valid_vec(vec: np.ndarray) -> bool:
    ops_idx = vec[:N_OPS]
    edges   = vec[N_OPS:]
    ops = ['input'] + [_OP_LIST[int(o)] for o in ops_idx] + ['output']
    mat = np.zeros((7, 7), dtype=int)
    k = 0
    for i in range(7):
        for j in range(i + 1, 7):
            mat[i, j] = int(edges[k])
            k += 1
    spec = _ModelSpec101(matrix=mat, ops=ops)
    return spec.valid_spec and int(np.sum(edges)) <= 9


# ─── PDNS operators ─────────────────────────────────────────────────────

class NoCrossover101(Crossover):
    """
    No crossover: each parent is copied directly into its offspring slot.
    Mirrors NoCrossoverProgram from strategy.operations.crossover.
    """

    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=1.0, **kwargs)

    def _do(self, problem, X, **kwargs):
        # X.shape = (n_parents, n_matings, n_var)
        Xp = np.empty_like(X)
        for p in range(X.shape[0]):
            for m in range(X.shape[1]):
                Xp[p, m] = X[p, m].copy()
        return Xp


class TwoPointCrossover101(Crossover):
    """
    2-point crossover over the 26-gene integer vector.
    Retries up to 26 times to obtain a valid pair; falls back to parents.
    prob=0.9 (per-mating crossover probability).
    """

    def __init__(self, prob: float = 0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Xp = np.empty_like(X)

        for i in range(n_matings):
            for _ in range(n_var):           # up to 26 retries
                # choose 2 cut points
                cuts = np.sort(
                    np.random.choice(n_var - 1, 2, replace=False) + 1
                )
                a, b = int(cuts[0]), int(cuts[1])

                c0 = X[0, i].copy()
                c1 = X[1, i].copy()
                c0[a:b], c1[a:b] = X[1, i, a:b].copy(), X[0, i, a:b].copy()

                if _is_valid_vec(c0.astype(int)) and _is_valid_vec(c1.astype(int)):
                    Xp[0, i] = c0
                    Xp[1, i] = c1
                    break
            else:
                # fallback: pass parents through unchanged
                Xp[0, i] = X[0, i].copy()
                Xp[1, i] = X[1, i].copy()

        return Xp


class UniformMutation101(Mutation):
    """
    Polynomial mutation with eta=1.0 and prob=1/n_var.
    Retries up to 26 times for validity; falls back to parent.
    Mirrors PDNS CustomPolynomialMutation.
    """

    def __init__(self, prob=None, eta: float = 1.0, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))

    def _do(self, problem, X, params=None, **kwargs):
        X = X.astype(float)
        Xp = np.copy(X)
        eta      = get(self.eta, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))

        for i in range(len(X)):
            for _ in range(N_VAR):          # up to 26 retries
                candidate = mut_pm(
                    X[i].reshape(1, -1),
                    problem.xl, problem.xu,
                    np.array([eta[i]]),
                    np.array([prob_var[i]]),
                    at_least_once=False,
                )
                if _is_valid_vec(np.round(candidate[0]).astype(int)):
                    Xp[i] = candidate[0]
                    break
            # if all retries failed, Xp[i] remains == X[i] (parent)
        return Xp


class SinglePointMutation101(Mutation):
    """
    Single-gene mutation: exactly one gene is changed per individual.
    - Op gene  (0-4):  replaced by a uniformly chosen *different* op (0-2).
    - Edge gene (5-25): bit-flipped.
    Retries up to 26 times for validity; falls back to parent.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        X = X.astype(float)
        Xp = np.copy(X)

        for i in range(len(X)):
            for _ in range(N_VAR):          # up to 26 retries
                candidate = X[i].copy()
                gene = np.random.randint(0, N_VAR)
                if gene < N_OPS:            # op gene
                    current = int(round(candidate[gene]))
                    choices = [v for v in range(3) if v != current]
                    candidate[gene] = float(np.random.choice(choices))
                else:                       # edge gene — flip
                    candidate[gene] = 1.0 - candidate[gene]
                if _is_valid_vec(np.round(candidate).astype(int)):
                    Xp[i] = candidate
                    break
            # if all retries failed, Xp[i] remains == X[i] (parent)
        return Xp


# ─── pymoo Problem wrapper ────────────────────────────────────────────────────

from pymoo.core.problem import Problem


class NASBench101Problem(Problem):
    """
    Pymoo problem over the 26-gene integer NASBench-101 search space.

    Objectives (both minimised):
      - val_acc_12 error  (1 − val_acc@epoch12)
      - n_params normalised to [0, 1]

    Time tracking mirrors PDNS: self.time is a list of
    wall_clock + cumulative_estimated_training_time values, one per
    *unique* architecture evaluation.  self.log_archs stores each
    evaluated vector (to avoid double-counting duplicates).
    """

    TRAIN_TIME_ESTIMATE = 0.30215823150349097   # seconds — PDNS n_params estimate

    def __init__(self, bench_db: dict, **kwargs):
        super().__init__(
            n_var=N_VAR, n_obj=2,
            xl=XL.astype(float), xu=XU.astype(float),
        )
        self.bench_db = bench_db
        self.cum_time  = 0.0
        self.time      = [time.time()]   # start timestamp
        self.log_archs: list = []

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        F = np.zeros((n, 2))

        for i, vec in enumerate(X):
            vec_int = np.round(vec).astype(int)
            vec_int = np.clip(vec_int, XL, XU)

            arch_str = _vec_to_arch_str(vec_int, self.bench_db)
            if arch_str is None or arch_str not in self.bench_db:
                # invalid architecture — penalise
                F[i, 0] = 1.0
                F[i, 1] = 1.0
                continue

            entry    = self.bench_db[arch_str]
            val_err  = 1.0 - entry['val_acc_12']
            n_params = entry['n_params']
            n_params_norm = (n_params - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)

            F[i, 0] = val_err
            F[i, 1] = n_params_norm

            # time tracking (PDNS-compatible)
            if vec_int.tolist() not in self.log_archs:
                train_time = entry.get('train_time_12', self.TRAIN_TIME_ESTIMATE)
                self.cum_time += float(train_time) + self.TRAIN_TIME_ESTIMATE
                self.log_archs.append(vec_int.tolist())

            self.time.append(time.time() + self.cum_time)

        out['F'] = F


# ─── archive / indicator helpers ─────────────────────────────────────────────

def _update_archive(var_arch, obj_arch, var_new, obj_new):
    """Non-dominated archive update (identical to PDNS update_elitist_archive)."""
    var_arch = list(var_arch)
    obj_arch = list(obj_arch)

    dominated_idx = []
    new_dominated  = False

    for idx, obj_existing in enumerate(obj_arch):
        if obj_new[0] >= obj_existing[0] and obj_new[1] >= obj_existing[1]:
            new_dominated = True
            break
        if obj_existing[0] >= obj_new[0] and obj_existing[1] >= obj_new[1]:
            dominated_idx.append(idx)

    if not new_dominated:
        for idx in sorted(dominated_idx, reverse=True):
            var_arch.pop(idx)
            obj_arch.pop(idx)
        var_arch.append(var_new)
        obj_arch.append(obj_new)

    return var_arch, obj_arch


def _test_archive(var_arch, bench_db):
    """Evaluate elitist archive on test_acc_108 and return non-dominated front."""
    test_objs = []
    for vec in var_arch:
        arch_str = _vec_to_arch_str(np.array(vec), bench_db)
        if arch_str is None or arch_str not in bench_db:
            test_objs.append((1.0, 1.0))
            continue
        entry = bench_db[arch_str]
        test_err = 1.0 - entry.get('test_acc_108', 0.0)
        n_params_norm = (entry['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)
        test_objs.append((test_err, n_params_norm))

    # keep only non-dominated
    nd_var, nd_obj = [], []
    for i, (v, o) in enumerate(zip(var_arch, test_objs)):
        dominated = any(
            other[0] <= o[0] and other[1] <= o[1] and other != o
            for other in test_objs
        )
        if not dominated:
            nd_var.append(v)
            nd_obj.append(o)
    return np.array(nd_var) if nd_var else np.empty((0, N_VAR)), \
           np.array(nd_obj)  if nd_obj  else np.empty((0, 2))


# ─── PDNS-style Callback ──────────────────────────────────────────────────────

from pymoo.core.callback import Callback


class PDNSStyleCallback(Callback):
    """
    Records per-evaluation indicators in PDNS format so that
    plot_nasbench101_comparison.py can load these results directly.
    """

    REF_POINT = np.array([1.05, 1.05])

    def __init__(self, bench_db: dict, pareto_ref: np.ndarray) -> None:
        super().__init__()
        self.bench_db   = bench_db
        self.pareto_ref = pareto_ref          # test-acc Pareto front (Nx2)
        self._hv_ind   = HV(ref_point=self.REF_POINT)
        self._igd_ind  = IGDPlus(pareto_ref)

        self.data['var_pop']          = []
        self.data['obj_pop']          = []
        self.data['var_archive']      = []
        self.data['obj_archive']      = []
        self.data['test_var_archive'] = []
        self.data['test_obj_archive'] = []
        self.data['indicators']       = []
        self.data['time']             = None

    def notify(self, algorithm):
        var_pop = algorithm.pop.get('X')
        obj_pop = algorithm.pop.get('F')

        var_arch = self.data['var_archive'][-1].copy() if self.data['var_archive'] else []
        obj_arch = self.data['obj_archive'][-1].copy() if self.data['obj_archive'] else []

        for var_ind, obj_ind in zip(var_pop, obj_pop):
            var_arch, obj_arch = _update_archive(var_arch, obj_arch, var_ind, obj_ind)

        test_var_arch, test_obj_arch = _test_archive(var_arch, self.bench_db)

        if len(test_obj_arch) > 0:
            indicators = {
                'hv':       float(self._hv_ind(test_obj_arch)),
                'igd_plus': float(self._igd_ind(test_obj_arch)),
            }
        else:
            indicators = {'hv': 0.0, 'igd_plus': np.inf}

        self.data['var_pop'].append(var_pop)
        self.data['obj_pop'].append(obj_pop)
        self.data['var_archive'].append(var_arch)
        self.data['obj_archive'].append(obj_arch)
        self.data['test_var_archive'].append(test_var_arch)
        self.data['test_obj_archive'].append(test_obj_arch)
        self.data['indicators'].append(indicators)
        self.data['time'] = algorithm.problem.time


# ─── random sampling that produces valid integer vectors ─────────────────────

from pymoo.core.sampling import Sampling


class ValidRandomSampling101(Sampling):
    """Generates a matrix of valid 26-gene NASBench-101 integer vectors."""

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, N_VAR), dtype=float)
        for i in range(n_samples):
            for _ in range(200):
                ops   = np.random.randint(0, 3, size=N_OPS)
                n_e   = np.random.randint(1, 10)
                edges = np.zeros(N_EDGES, dtype=int)
                edges[np.random.choice(N_EDGES, n_e, replace=False)] = 1
                vec = np.concatenate([ops, edges])
                if _is_valid_vec(vec):
                    X[i] = vec.astype(float)
                    break
        return X


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
    print('  Building test-acc Pareto front from scratch …')
    db = _load_bench_db()
    F_all = np.array([
        [1.0 - v['test_acc_108'],
         (v['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)]
        for v in db.values()
        if 'test_acc_108' in v
    ])
    # non-dominated
    is_nd = np.ones(len(F_all), dtype=bool)
    for i in range(len(F_all)):
        if not is_nd[i]:
            continue
        dominated = np.all(F_all <= F_all[i], axis=1) & np.any(F_all < F_all[i], axis=1)
        is_nd[dominated] = False
        is_nd[i] = True
    return F_all[is_nd]


def run_single(method: str, seed: int, pop_size: int, n_gen: int, bench_db: dict, pareto_ref: np.ndarray,
               n_doe=None, n_infill=None, n_gen_inner=20, inner_pop_size=None,
               predict_obj=None, real_obj=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    problem  = NASBench101Problem(bench_db)
    callback = PDNSStyleCallback(bench_db, pareto_ref)
    sampling = ValidRandomSampling101()

    if method == 'random':
        algorithm = RandomGA(pop_size=pop_size,
                             sampling=sampling,
                             eliminate_duplicates=True,)

    elif method == 'nsga2':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=TwoPointCrossover101(prob=0.9),
            mutation=UniformMutation101(prob=1.0 / N_VAR, eta=1.0),
            eliminate_duplicates=True,
        )
    elif method == 'nsga2-single':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=NoCrossover101(),
            mutation=SinglePointMutation101(),
            eliminate_duplicates=True,
        )
    elif method in ('samos-rfr', 'samos-xgb'):
        n_doe    = n_doe    if n_doe    is not None else pop_size
        n_infill = n_infill if n_infill is not None else pop_size
        inner_pop_size = inner_pop_size if inner_pop_size is not None else pop_size
        predict_obj = predict_obj if predict_obj is not None else ['val_err_12']
        real_obj    = real_obj    if real_obj    is not None else ['n_params']
        print(f'  [SAMOS] predict={predict_obj}  real={real_obj}')

        rng  = np.random.RandomState(seed)
        surrogates = [
            (RFR(20, seed=rng.randint(0, 2**31 - 1))
             if method == 'samos-rfr' else
             XGBoost(100, seed=rng.randint(0, 2**31 - 1)))
            for _ in range(len(predict_obj))
        ]
        factory = lambda surrs, _ro=real_obj, _db=bench_db: SurrogateProblem101(surrs, _ro, _db)
        algorithm = SAMOS(
            sampling=sampling,
            surrogates=surrogates,
            surrogate_problem_factory=factory,
            crossover=TwoPointCrossover101(prob=0.9),
            mutation=UniformMutation101(prob=1.0 / N_VAR, eta=1.0),
            n_doe=n_doe,
            n_infill=n_infill,
            n_gen_inner=n_gen_inner,
            ga_pop_size=inner_pop_size,
            use_subset_selection=True,
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

    data = results.algorithm.callback.data
    data['time'] = problem.time          # list of timestamps (PDNS-compatible)
    data['log_archs'] = problem.log_archs

    return data


# ─── Inner surrogate Problem (used by SAMOS methods) ────────────────────────────────

class SurrogateProblem101(Problem):
    """
    Inner pymoo problem for the SAMOS surrogate loop.

    Objectives (in order): predicted ones first, real ones second.
    - Predicted objectives: approximated by the fitted surrogates.
    - Real objectives (e.g. n_params): evaluated exactly via bench_db lookup
      at zero additional training cost.
    """

    # Mapping from real-objective name -> how to compute it from a bench_db entry
    _REAL_OBJ_FNS = {
        'n_params': lambda e: (e['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS),
    }

    def __init__(self, surrogates, real_objectives: list, bench_db: dict, **kwargs):
        n_obj = len(surrogates) + len(real_objectives)
        super().__init__(
            n_var=N_VAR, n_obj=n_obj,
            xl=XL.astype(float), xu=XU.astype(float),
        )
        self.surrogates      = surrogates
        self.real_objectives = real_objectives
        self.bench_db        = bench_db

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        n_pred = len(self.surrogates)
        n_real = len(self.real_objectives)
        F = np.zeros((n, n_pred + n_real))

        # Predicted objectives via surrogates
        X_float = np.round(X).astype(float)
        for i, surrogate in enumerate(self.surrogates):
            preds = surrogate.predict(X_float)
            F[:, i] = np.clip(preds.squeeze(), 0.0, 1.0)

        # Real objectives via bench_db lookup
        for j, obj_name in enumerate(self.real_objectives):
            fn = self._REAL_OBJ_FNS.get(obj_name)
            if fn is None:
                raise ValueError(f'Unknown real objective: {obj_name}')
            for k, x in enumerate(X):
                arch_str = _vec_to_arch_str(np.round(x).astype(int), self.bench_db)
                if arch_str is not None and arch_str in self.bench_db:
                    F[k, n_pred + j] = fn(self.bench_db[arch_str])
                else:
                    F[k, n_pred + j] = 1.0   # penalise invalid

        out['F'] = F


def main(args):
    print(f'Loading benchmark data from {DATA_FILE} …')
    bench_db = _load_bench_db()
    print(f'  {len(bench_db):,} architectures')

    print('Loading test-acc Pareto reference front …')
    pareto_ref = _load_test_pareto_ref()
    print(f'  {len(pareto_ref)} non-dominated points')

    methods = []
    if args.random:
        methods.append('random')
    if args.nsga2:
        methods.append('nsga2')
    if args.nsga2_single:
        methods.append('nsga2-single')
    if args.samos_rfr:
        methods.append('samos-rfr')
    if args.samos_xgb:
        methods.append('samos-xgb')
    if not methods:
        methods = ['random', 'nsga2', 'nsga2-single', 'samos-rfr', 'samos-xgb']

    for method in methods:
        save_dir = os.path.join(RESULTS_ROOT, method)
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
                predict_obj=args.predict_obj, real_obj=args.real_obj,
            )

            with open(out_path, 'wb') as f:
                pickle.dump(data, f)
            print(f'  Saved -> {out_path}')

    # ── plot HV and IGD+ trajectories ─────────────────────────────────────────
    hv_ceiling = float(HV(ref_point=np.array([1.05, 1.05]))(pareto_ref))
    plot_out = os.path.join(RESULTS_ROOT, 'baseline_hv_igd.png')
    print(f'\nGenerating HV / IGD+ plot …')
    plot_results(methods, args.n_gen, args.pop_size, hv_ceiling, plot_out)


# ─── plotting ─────────────────────────────────────────────────────────────────

COLOURS = {
    'random':       '#4e79a7',
    'nsga2':        '#f28e2b',
    'nsga2-single': '#59a14f',
    'samos-rfr':    '#e15759',
    'samos-xgb':    '#b07aa1',
}
LABELS = {
    'random':       'Random',
    'nsga2':        'NSGA-II (2-pt XO, unif. mut)',
    'nsga2-single': 'NSGA-II (no XO, single-pt mut)',
    'samos-rfr':    'SAMOS (RFR surrogate)',
    'samos-xgb':    'SAMOS (XGBoost surrogate)',
}


def _load_indicator_trajectories(method: str, n_gen: int):
    """Return (hv_mean, hv_std, igd_mean, igd_std) arrays of length n_gen."""
    seed_dir = os.path.join(RESULTS_ROOT, method)
    if not os.path.isdir(seed_dir):
        return None

    hv_runs, igd_runs = [], []
    for pkl_file in sorted(os.listdir(seed_dir)):
        if not pkl_file.endswith('.pkl'):
            continue
        with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
            data = pickle.load(f)

        indicators = data.get('indicators', [])
        hv_series  = [ind.get('hv',       0.0) for ind in indicators]
        igd_series = [ind.get('igd_plus', np.nan) for ind in indicators]

        # Each generation appends pop_size entries; keep one per generation
        # by sampling every pop_size-th entry (last entry of each gen).
        # If the list length == n_gen already (one per gen), use as-is.
        if len(hv_series) == n_gen:
            hv_runs.append(hv_series)
            igd_runs.append(igd_series)
        elif len(hv_series) >= n_gen:
            # One entry per individual: sample the last entry of each generation
            step = len(hv_series) // n_gen
            hv_runs.append( [hv_series[min((g + 1) * step - 1, len(hv_series) - 1)]  for g in range(n_gen)])
            igd_runs.append([igd_series[min((g + 1) * step - 1, len(igd_series) - 1)] for g in range(n_gen)])
        else:
            hv_runs.append(hv_series)
            igd_runs.append(igd_series)

    if not hv_runs:
        return None

    # Pad shorter runs to the same length with their last value
    max_len = max(len(r) for r in hv_runs)
    for r in hv_runs:
        while len(r) < max_len:
            r.append(r[-1])
    for r in igd_runs:
        while len(r) < max_len:
            r.append(r[-1])

    hv_arr  = np.array(hv_runs,  dtype=float)
    igd_arr = np.array(igd_runs, dtype=float)
    return (
        hv_arr.mean(axis=0),  hv_arr.std(axis=0),
        igd_arr.mean(axis=0), igd_arr.std(axis=0),
    )


def plot_results(methods: list, n_gen: int, pop_size: int, hv_ceiling: float, out_path: str):
    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for method in methods:
        result = _load_indicator_trajectories(method, n_gen)
        if result is None:
            print(f'  [plot] No data found for method={method}, skipping.')
            continue
        hv_mean, hv_std, igd_mean, igd_std = result
        x = np.arange(1, len(hv_mean) + 1) * pop_size   # evaluations
        colour = COLOURS.get(method, None)
        label  = LABELS.get(method, method)

        # HV (left)
        axes[0].plot(x, hv_mean, label=label, color=colour, linewidth=1.8)
        axes[0].fill_between(x, hv_mean - hv_std, hv_mean + hv_std,
                             alpha=0.15, color=colour)

        # IGD+ (right)
        axes[1].plot(x, igd_mean, label=label, color=colour, linewidth=1.8)
        axes[1].fill_between(x, igd_mean - igd_std, igd_mean + igd_std,
                             alpha=0.15, color=colour)

    # HV ceiling reference line
    axes[0].axhline(hv_ceiling, color='black', linestyle='--', linewidth=1.0,
                    label=f'Optimal HV ({hv_ceiling:.4f})')

    axes[0].set_title('Hypervolume (Higher is better)')
    axes[0].set_xlabel('Evaluations')
    axes[0].set_ylabel('Hypervolume')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('IGD+ (Lower is better)')
    axes[1].set_xlabel('Evaluations')
    axes[1].set_ylabel('IGD+')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f'NASBench-101  --  test_acc@108  x  n_params\n'
        f'(pop={pop_size}, {n_gen} generations = {pop_size * n_gen} evals, mean +/- std over seeds)',
        y=1.01,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'  Plot saved -> {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NASBench-101 baselines: Random + NSGA-II (PDNS-style)'
    )
    parser.add_argument('--random', action='store_true',
                        help='Run random search')
    parser.add_argument('--nsga2',  action='store_true',
                        help='Run NSGA-II with 2-point crossover + uniform mutation')
    parser.add_argument('--nsga2_single', action='store_true',
                        help='Run NSGA-II with no crossover + single-point mutation')
    parser.add_argument('--samos_rfr', action='store_true',
                        help='Run simplified SAMOS with Random Forest surrogate')
    parser.add_argument('--samos_xgb', action='store_true',
                        help='Run simplified SAMOS with XGBoost surrogate')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(30)),
                        help='Seeds to run (default: 0-29)')
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
                        help='SAMOS: inner NSGA-II population size (default: same as --pop_size)')
    parser.add_argument('--predict_obj', type=str, nargs='+', default=['val_err_12'],
                        help='SAMOS: objectives approximated by surrogates (default: val_err_12)')
    parser.add_argument('--real_obj', type=str, nargs='*', default=['n_params'],
                        help='SAMOS: objectives evaluated exactly in inner loop (default: n_params)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if result file already exists')

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
