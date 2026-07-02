"""experiments/samos/_common.py --- shared runner for SAMOS2 contribution ablations.

Each experiment_*.py under experiments/samos/ tests exactly one SAMOS2
contribution (C1 encoding, C2 canonicalisation, C3 uncertainty-aware infill)
in isolation: every variant uses this same runner, same seeds, same
pop/inner-GA budget, and differs ONLY in the samos2_kwargs / mutation passed
for that one contribution. The 'baseline' variant always leaves the new
toggle at its SAMOSMinimal-equivalent default, so any HV/IGD delta between
variants is attributable to that single contribution.

Metrics are computed on the *true (test)* objectives -- not the surrogate's
predictions -- every outer generation:
  hv  : hypervolume vs. benchmark.hv_ref_point (always available)
  igd : inverted generational distance vs. benchmark.pareto_front (only for
        benchmarks that expose a known true Pareto front; NaN otherwise --
        e.g. DARTS/c10mop MOP8-9 have no known front)

Normalisation follows the convention already used by
experiments/samos_cheap/gens_to_dominate.py's DominationCallback: call
benchmark.normalize() unconditionally on true-objective evaluations before
computing indicators.
"""

import argparse
import contextlib
import inspect
import json
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize

INNER_GAS = {'nsga2': NSGA2, 'sms-emoa': SMSEMOA}

from problem.evoxbench.baseline_problem import EvoXBenchProblem
from problem.evoxbench.benchmark_meta import BENCHMARK_META
from problem.evoxbench.surrogate_problem import SurrogateProblemEvox
from problem.evoxbench.utils import get_benchmark
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.sampler import EvoxBenchSampler
from strategy.surrogate.models import get_surrogate_model
from strategy.surrogate.samos2 import SAMOS2


def _make_surrogate(surr_cls, seed, n_estimators=100):
    """Construct one surrogate instance, adapting to whichever constructor
    kwargs the model class accepts (RFR/ETR/EnsembleSurrogate need
    n_estimators with no default; XGBoost defaults it; GPR variants take
    only seed). Models needing other kwargs (KNN's n_neighbors, CART's
    n_tree/rnd, kriging's length_scale) are out of scope for this simple
    CLI-driven --surrogate path -- construct those directly instead."""
    params = inspect.signature(surr_cls.__init__).parameters
    kwargs = {}
    if 'seed' in params:
        kwargs['seed'] = seed
    if 'n_estimators' in params:
        kwargs['n_estimators'] = n_estimators
    return surr_cls(**kwargs)


def infer_obj_split(suite, pid, n_obj):
    """(predict_obj_indices, real_obj_indices) from BENCHMARK_META cheap indices."""
    meta = BENCHMARK_META.get(suite, {}).get(pid, {})
    cheap = list(meta.get('cheap_obj_indices', []))
    real_idx = cheap
    predict_idx = [i for i in range(n_obj) if i not in cheap]
    return predict_idx, real_idx


class TrueMetricsCallback(Callback):
    """After each outer generation, evaluate the whole archive's true (test)
    objectives and record hypervolume (always) and IGD (if a known Pareto
    front is available)."""

    def __init__(self, benchmark, seed=None, out=None):
        super().__init__()
        self.benchmark = benchmark
        self.seed = seed
        self._out = out if out is not None else sys.stdout
        self.hv: list = []
        self.igd: list = []
        self.n_eval: list = []

        ref_raw = getattr(benchmark, 'hv_ref_point', None)
        self._hv = (
            HV(ref_point=benchmark.normalize(np.asarray(ref_raw, dtype=float)[None, :])[0])
            if ref_raw is not None else None
        )

        pf_raw = getattr(benchmark, 'pareto_front', None)
        self._igd = (
            IGD(benchmark.normalize(np.asarray(pf_raw, dtype=float)))
            if pf_raw is not None else None
        )

    def notify(self, algorithm):
        X = algorithm._archive.get('X')
        if X is None or len(X) == 0:
            return
        X_int = np.round(X).astype(int)
        F_test = self.benchmark.evaluate(X_int, true_eval=True)
        F_test = F_test[np.isfinite(F_test).all(axis=1)]
        if len(F_test) == 0:
            return
        F_norm = self.benchmark.normalize(F_test)

        hv = float(self._hv(F_norm)) if self._hv is not None else float('nan')
        igd = float(self._igd(F_norm)) if self._igd is not None else float('nan')
        self.hv.append(hv)
        self.igd.append(igd)
        self.n_eval.append(len(X))

        tag = f'[seed {self.seed}] ' if self.seed is not None else ''
        print(f'{tag}gen {algorithm.n_gen:3d} | n_eval={len(X):5d} | '
              f'hv={hv:.4f} | igd={igd:.4f}', file=self._out, flush=True)


def run_seed(seed, suite, pid, pop, inner_pop, inner_gen, n_doe, n_gen_max,
             surrogate_name, inner_ga='nsga2', mutation=None, samos2_kwargs=None):
    """Run one SAMOS2 seed on suite/pid; samos2_kwargs is merged into the
    SAMOS2 constructor on top of the shared defaults below (this is where a
    single experiment_*.py injects its one contribution's toggle)."""
    np.random.seed(seed)
    random.seed(seed)

    benchmark = get_benchmark(suite, pid)
    problem = EvoXBenchProblem(benchmark)
    n_obj = problem.n_obj

    predict_idx, real_idx = infer_obj_split(suite, pid, n_obj)
    if not predict_idx:
        raise ValueError(
            f'{suite}/pid{pid} has no expensive objective to predict '
            f'(cheap_obj_indices covers all {n_obj} objectives).')

    xl = np.asarray(benchmark.search_space.lb, dtype=int)
    xu = np.asarray(benchmark.search_space.ub, dtype=int)
    sampling = EvoxBenchSampler(xl, xu)
    crossover = IntegerUniformCrossover(prob=0.9)
    mutation = mutation if mutation is not None else IntegerPointMutation(xl, xu)
    elim = IntegerVectorDuplicateElimination()

    surr_cls = get_surrogate_model(surrogate_name)
    surrogates = [_make_surrogate(surr_cls, seed) for _ in predict_idx]

    def factory(surrs):
        return SurrogateProblemEvox(
            surrogates=surrs,
            predict_obj_indices=predict_idx,
            real_obj_indices=real_idx,
            benchmark=benchmark,
            no_norm=False,
        )

    kwargs = dict(samos2_kwargs or {})
    kwargs.setdefault('eliminate_duplicates', elim)

    algorithm = SAMOS2(
        sampling=sampling,
        surrogates=surrogates,
        surrogate_problem_factory=factory,
        predict_obj_indices=predict_idx,
        crossover=crossover,
        mutation=mutation,
        n_doe=n_doe,
        n_infill=pop,
        n_gen_inner=inner_gen,
        ga_pop_size=inner_pop,
        use_subset_selection=True,
        inner_algorithm=INNER_GAS[inner_ga],
        **kwargs,
    )

    cb = TrueMetricsCallback(benchmark, seed=seed, out=sys.stdout)
    t0 = time.time()
    with open(os.devnull, 'w') as _devnull, contextlib.redirect_stdout(_devnull):
        minimize(problem, algorithm, termination=('n_gen', n_gen_max),
                 callback=cb, seed=seed, verbose=False)
    return {
        'seed': seed,
        'hv': cb.hv,
        'igd': cb.igd,
        'n_eval': cb.n_eval,
        'time': time.time() - t0,
    }


def add_common_args(p: argparse.ArgumentParser, default_output: str):
    """Each experiment_*.py should call p.set_defaults(pid=...) afterwards to
    pick a sensible default PID for that contribution."""
    p.add_argument('--suite', default='c10mop', choices=['c10mop', 'in1kmop'])
    p.add_argument('--pid', type=int, default=None, help='Required (set via experiment default or --pid).')
    p.add_argument('--pop', type=int, default=20, help='Real evals per outer generation (n_infill).')
    p.add_argument('--n_doe', type=int, default=40, help='Initial DOE size (generation 1).')
    p.add_argument('--inner_pop', type=int, default=200, help='Inner surrogate-GA population.')
    p.add_argument('--inner_gen', type=int, default=20, help='Inner surrogate-GA generations.')
    p.add_argument('--surrogate', default='RFR', help='Surrogate model key (see strategy/surrogate/models).')
    p.add_argument('--inner_ga', default='nsga2', choices=list(INNER_GAS))
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--start_seed', type=int, default=0)
    p.add_argument('--n_gen_max', type=int, default=15, help='Outer-generation budget.')
    p.add_argument('--output', default=default_output)
    return p


def run_comparison(args, contribution_name: str, variants: dict):
    """variants : dict[name -> {'samos2_kwargs': dict, 'mutation_factory': fn(xl, xu) -> Mutation | None}]

    Runs every variant over the same seed range with identical pop/inner-GA
    budget, prints a per-variant summary, and (if --output given) saves
    per-seed hv/igd trajectories for offline plotting.
    """
    print(f'[SAMOS2-{contribution_name}] {args.suite}/pid{args.pid} '
          f'pop={args.pop} inner={args.inner_ga}:{args.inner_pop}x{args.inner_gen} '
          f'n_gen_max={args.n_gen_max} seeds={args.start_seed}..{args.start_seed + args.seeds - 1} '
          f'variants={list(variants)}')

    all_results = {}
    for name, spec in variants.items():
        mutation_factory = spec.get('mutation_factory')
        mutation = None
        if mutation_factory is not None:
            benchmark = get_benchmark(args.suite, args.pid)
            xl = np.asarray(benchmark.search_space.lb, dtype=int)
            xu = np.asarray(benchmark.search_space.ub, dtype=int)
            mutation = mutation_factory(xl, xu)

        results = []
        for s in range(args.start_seed, args.start_seed + args.seeds):
            r = run_seed(s, args.suite, args.pid, args.pop, args.inner_pop, args.inner_gen,
                         args.n_doe, args.n_gen_max, args.surrogate, inner_ga=args.inner_ga,
                         mutation=mutation, samos2_kwargs=spec.get('samos2_kwargs'))
            final_hv = r['hv'][-1] if r['hv'] else float('nan')
            final_igd = r['igd'][-1] if r['igd'] else float('nan')
            print(f'  [{name} | seed {s:2d}] final_hv={final_hv:.4f}  '
                  f'final_igd={final_igd:.4f}  ({r["time"]:.0f}s)')
            results.append(r)
        all_results[name] = results

    summary = {
        'suite': args.suite, 'pid': args.pid, 'contribution': contribution_name,
        'pop': args.pop, 'inner_pop': args.inner_pop, 'inner_gen': args.inner_gen,
        'n_doe': args.n_doe, 'n_gen_max': args.n_gen_max, 'surrogate': args.surrogate,
        'inner_ga': args.inner_ga, 'n_seeds': args.seeds,
    }

    print('\n' + '=' * 72)
    print(f'SAMOS2-{contribution_name} on {args.suite}/pid{args.pid}')
    per_variant_summary = {}
    for name, results in all_results.items():
        finals_hv  = np.array([r['hv'][-1] if r['hv'] else np.nan for r in results])
        finals_igd = np.array([r['igd'][-1] if r['igd'] else np.nan for r in results])
        print(f'  {name:>14s} : hv={np.nanmean(finals_hv):.4f} +/- {np.nanstd(finals_hv):.4f}  '
              f'igd={np.nanmean(finals_igd):.4f} +/- {np.nanstd(finals_igd):.4f}  (n={len(results)})')
        per_variant_summary[name] = {
            'hv_mean': float(np.nanmean(finals_hv)), 'hv_std': float(np.nanstd(finals_hv)),
            'igd_mean': float(np.nanmean(finals_igd)), 'igd_std': float(np.nanstd(finals_igd)),
        }
    print('=' * 72)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump({'summary': summary, 'results': all_results}, f)
        with open(os.path.splitext(args.output)[0] + '.json', 'w') as f:
            json.dump({
                **summary,
                'per_variant_summary': per_variant_summary,
                'per_variant_trajectories': {
                    name: {'hv': [r['hv'] for r in res], 'igd': [r['igd'] for r in res]}
                    for name, res in all_results.items()
                },
            }, f, indent=2)
        print(f'saved -> {args.output}')

    return summary
