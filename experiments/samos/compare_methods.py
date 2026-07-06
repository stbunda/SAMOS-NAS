"""experiments/samos/compare_methods.py --- Compare each SAMOS2 contribution
(C1 encoding, C2 isomorphism, C3 uncertainty) in isolation against the
established baselines random / samos-xgb, across every EvoXBench suite/pid.
Modeled on depreciated/main_evoxbench.py's run_single/main structure (same
budget defaults: pop_size=20, n_gen=60, n_gen_inner=20, n_doe/n_infill=
pop_size, inner_pop_size=pop_size*10) and reuses its output format
(EvoxBenchCallback -> results.algorithm.callback.data), so results are
drop-in compatible with that project's existing indicator/plotting
conventions.

Methods
-------
  random              : RandomGA (no surrogate).
  samos-xgb           : SAMOSMinimal + XGBoost, ALL objectives surrogate-
                         predicted (no cheap/real split).
  samos2-baseline     : SAMOS2, all three toggles off, cheap-real split
                         (only the expensive objective(s) -- BENCHMARK_META
                         cheap_obj_indices -- are surrogate-predicted). This
                         is the SAMOS2 contributions' reference point; it
                         replaces the old samos-cheapreal method, which used
                         the unpatched SAMOSMinimal fit/predict path (see
                         strategy/surrogate/samos2.py docstring for the
                         predict_obj_indices fix).
  samos2-encoding     : samos2-baseline + C1 (categorical one-hot / ordinal
                         local-step operator). Requires a NB101/NB201/NATS pid
                         -- automatically skipped on pids whose search space
                         has no categorical-op structure (DARTS, ResNet-50D,
                         Transformer, MobileNetV3).
  samos2-isomorphism  : samos2-baseline + C2 (canonical-phenotype dedup +
                         training-set collapse).
  samos2-uncertainty  : samos2-baseline + C3 (acquisition-based infill
                         selection). Forces an ensemble/quantile surrogate
                         (RFR by default) since XGBoost has no predict_std.
  samos2-rfr          : samos2-baseline with the RFR surrogate. De-confounds
                         C3: samos2-uncertainty differs from samos2-baseline
                         in surrogate family AND selection policy; this is
                         the matched control (RFR + diversity selection).
  samos2-xgb10        : samos2-baseline with an EnsembleSurrogate of 10
                         XGBoosts (different random_state -> subsample
                         disagreement gives predict_std). Diversity
                         selection -- the XGB-family control arm.
  samos2-xgb10-uncertainty : samos2-xgb10 + acquisition-based infill
                         selection. C3 within the XGBoost family, matched
                         against samos2-xgb10.
  samos2-uncertainty-lin_annealing : samos2-uncertainty with kappa decayed
                         linearly from --kappa at the first infill to 0 at
                         the final generation (explore early, exploit late).
  samos2-uncertainty-obj : C3 as an extra inner-GA objective instead of a
                         selection rule: the inner surrogate problem gets an
                         (n_obj+1)-th objective -sigma(x), so the candidate
                         pool mixes exploit/explore; infill selection stays
                         the diversity default. RFR surrogate.

The loop nests seed (outermost) > suite > pid > method, and each run is
skipped if its output pkl already exists (unless --overwrite), so an
interrupted batch can simply be re-launched.

Examples
--------
  python experiments/samos/compare_methods.py                                   # all suites/pids/methods, seeds 0-19
  python experiments/samos/compare_methods.py --suites c10mop --pids 5 8
  python experiments/samos/compare_methods.py --methods samos2-baseline samos2-uncertainty --acq_kind hvi
"""

import argparse
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from pymoo.optimize import minimize

from _common import _make_surrogate

from problem.evoxbench.baseline_problem import EvoXBenchProblem
from problem.evoxbench.benchmark_meta import BENCHMARK_META, get_op_var_group
from problem.evoxbench.callbacks import EvoxBenchCallback
from problem.evoxbench.surrogate_problem import (
    SurrogateProblemEvox,
    SurrogateProblemEvoxUncertainty,
)
from problem.evoxbench.utils import get_benchmark
from strategy.algorithm.algorithms import RandomGA
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import EncodingAwareMutation, IntegerPointMutation
from strategy.sampler import EvoxBenchSampler
from strategy.surrogate.canonical import Canonicalizer
from strategy.surrogate.encoding import EncodingSpec
from strategy.surrogate.infill import AcquisitionSelector
from strategy.surrogate.models import EnsembleSurrogate, XGBoost, get_surrogate_model
from strategy.surrogate.samos2 import SAMOS2
from strategy.surrogate.samos_minimal import SAMOSMinimal

METHODS = [
    'random', 'samos-xgb',
    'samos2-baseline', 'samos2-rfr', 'samos2-xgb10',
    'samos2-encoding', 'samos2-isomorphism',
    'samos2-uncertainty', 'samos2-uncertainty-lin_annealing',
    'samos2-uncertainty-obj', 'samos2-xgb10-uncertainty',
]
XGB10_MEMBERS = 10


def _cheap_split(suite, pid, n_obj):
    meta = BENCHMARK_META.get(suite, {}).get(pid, {})
    real_idx = list(meta.get('cheap_obj_indices', []))
    predict_idx = [i for i in range(n_obj) if i not in real_idx]
    return predict_idx, real_idx


def _problem_scale_ref_point(benchmark):
    """HV reference point in the normalisation SAMOS2 sees internally
    (mirrors EvoXBenchProblem._evaluate's conditional normalize)."""
    ref = np.asarray(benchmark.hv_ref_point, dtype=float)[None, :]
    if not benchmark.normalized_objectives:
        ref = benchmark.normalize(ref)
    return ref[0]


def build_algorithm(method, benchmark, suite, pid, seed, pop_size,
                     n_doe, n_infill, n_gen_inner, inner_pop_size,
                     surrogate_name, acq_kind, kappa, n_gen):
    xl = np.asarray(benchmark.search_space.lb, dtype=int)
    xu = np.asarray(benchmark.search_space.ub, dtype=int)
    n_obj = benchmark.evaluator.n_objs

    sampler   = EvoxBenchSampler(xl, xu)
    crossover = IntegerUniformCrossover(prob=0.9)
    mutation  = IntegerPointMutation(xl, xu)
    elim      = IntegerVectorDuplicateElimination()

    n_doe_    = n_doe if n_doe is not None else pop_size
    n_infill_ = n_infill if n_infill is not None else pop_size
    inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

    if method == 'random':
        return RandomGA(pop_size=pop_size, sampling=sampler, eliminate_duplicates=elim)

    if method == 'samos-xgb':
        predict_idx, real_idx = list(range(n_obj)), []
        rng = np.random.RandomState(seed)
        surrogates = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_idx]

        def factory(surrs):
            return SurrogateProblemEvox(surrs, predict_idx, real_idx, benchmark)

        return SAMOSMinimal(
            sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
            crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
            n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
            eliminate_duplicates=elim, dedup_key_fn=elim.key,
        )

    if method.startswith('samos2-'):
        predict_idx, real_idx = _cheap_split(suite, pid, n_obj)
        if not predict_idx:
            raise ValueError(f'{suite}/pid{pid} has no expensive objective to predict.')

        surr_name = surrogate_name
        samos2_kwargs = {}
        problem_cls = SurrogateProblemEvox

        if method == 'samos2-encoding':
            search_space = BENCHMARK_META[suite][pid]['search_space']
            if get_op_var_group(search_space) is None:
                raise ValueError(
                    f'{suite}/pid{pid} ({search_space}) has no categorical-op structure -- '
                    f'encoding_spec would be a no-op. Pick a NB101/NB201/NATS pid instead.')
            encoding_spec = EncodingSpec.from_search_space(search_space, benchmark.search_space.n_var, xu)
            samos2_kwargs['encoding_spec'] = encoding_spec
            mutation = EncodingAwareMutation(xl, xu, categorical_cols=encoding_spec.categorical_cols)

        elif method == 'samos2-isomorphism':
            samos2_kwargs['canonicalizer'] = Canonicalizer(benchmark)
            samos2_kwargs['collapse_training_set'] = True

        elif method == 'samos2-rfr':
            surr_name = 'RFR'

        elif method in ('samos2-uncertainty', 'samos2-uncertainty-lin_annealing',
                        'samos2-uncertainty-obj'):
            if surr_name == 'XGBoost':
                surr_name = 'RFR'
                print(f'[{method}] XGBoost has no predict_std -- using RFR instead.')
            if method == 'samos2-uncertainty-obj':
                # sigma joins the inner GA's objectives; selection stays default
                problem_cls = SurrogateProblemEvoxUncertainty
            else:
                schedule = 'linear' if method.endswith('lin_annealing') else None
                ref_point = _problem_scale_ref_point(benchmark) if acq_kind == 'hvi' else None
                samos2_kwargs['infill_selector'] = AcquisitionSelector(
                    predict_idx, kind=acq_kind, kappa=kappa, ref_point=ref_point,
                    kappa_schedule=schedule,
                    total_gens=n_gen if schedule is not None else None)

        elif method == 'samos2-xgb10-uncertainty':
            ref_point = _problem_scale_ref_point(benchmark) if acq_kind == 'hvi' else None
            samos2_kwargs['infill_selector'] = AcquisitionSelector(
                predict_idx, kind=acq_kind, kappa=kappa, ref_point=ref_point)

        elif method not in ('samos2-baseline', 'samos2-xgb10'):
            raise ValueError(f'Unknown method: {method!r}')

        if method in ('samos2-xgb10', 'samos2-xgb10-uncertainty'):
            # 10 XGBoosts per predicted objective; member disagreement from
            # differing random_state (subsample/colsample randomness) gives
            # predict_std while keeping the surrogate in the XGBoost family.
            rng = np.random.RandomState(seed)
            surrogates = [
                EnsembleSurrogate([
                    XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                    for _ in range(XGB10_MEMBERS)
                ])
                for _ in predict_idx
            ]
        else:
            surr_cls = get_surrogate_model(surr_name)
            surrogates = [_make_surrogate(surr_cls, seed) for _ in predict_idx]

        def factory(surrs):
            return problem_cls(surrs, predict_idx, real_idx, benchmark)

        return SAMOS2(
            sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
            predict_obj_indices=predict_idx,
            crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
            n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
            **samos2_kwargs,
        )

    raise ValueError(f'Unknown method: {method!r}')


def run_single(method, suite, pid, seed, pop_size, n_gen, n_doe, n_infill,
               n_gen_inner, inner_pop_size, surrogate_name, acq_kind, kappa,
               no_norm=False, compute_indicators=True):
    np.random.seed(seed)
    random.seed(seed)

    benchmark = get_benchmark(suite, pid)
    problem   = EvoXBenchProblem(benchmark, no_norm=no_norm)
    callback  = EvoxBenchCallback(benchmark, no_norm=no_norm, compute_indicators=compute_indicators)

    algorithm = build_algorithm(
        method, benchmark, suite, pid, seed, pop_size,
        n_doe, n_infill, n_gen_inner, inner_pop_size, surrogate_name, acq_kind, kappa,
        n_gen)

    results = minimize(
        problem=problem, algorithm=algorithm, termination=('n_gen', n_gen),
        seed=seed, callback=callback, save_history=False, verbose=True,
    )
    return results.algorithm.callback.data


def main(args):
    budget_folder = f'B{args.n_gen * args.pop_size}_P{args.pop_size}'

    suite_pids = []
    for suite in args.suites:
        pids = args.pids if args.pids is not None else sorted(BENCHMARK_META.get(suite, {}))
        for pid in pids:
            if pid not in BENCHMARK_META.get(suite, {}):
                print(f'[SKIP] {suite}/pid{pid} not in BENCHMARK_META')
                continue
            suite_pids.append((suite, pid))

    total_runs = len(args.seeds) * len(suite_pids) * len(args.methods)
    run_i = 0
    summary: dict = {}   # (suite, pid, method) -> list of final hv, one per seed

    for seed in args.seeds:
        for suite, pid in suite_pids:
            search_space = BENCHMARK_META[suite][pid]['search_space']
            for method in args.methods:
                if method == 'samos2-encoding' and get_op_var_group(search_space) is None:
                    print(f'[SKIP] samos2-encoding unsupported on {suite}/pid{pid} '
                          f'({search_space}, no categorical-op structure)')
                    continue

                run_i += 1
                save_dir = os.path.join(
                    args.results_root, suite, f'pid{pid}', budget_folder, method)
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f'seed_{seed}.pkl')

                if os.path.exists(out_path) and not args.overwrite:
                    print(f'[SKIP {run_i}/{total_runs}] {suite}/pid{pid}/{method}/seed_{seed} already exists')
                    with open(out_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    print(f'\n[RUN {run_i}/{total_runs}] {suite}/pid{pid}  method={method}  seed={seed}  '
                          f'pop={args.pop_size}  n_gen={args.n_gen}')
                    data = run_single(
                        method, suite, pid, seed, args.pop_size, args.n_gen,
                        args.n_doe, args.n_infill, args.n_gen_inner, args.inner_pop_size,
                        args.surrogate, args.acq_kind, args.kappa,
                        no_norm=args.no_norm, compute_indicators=not args.no_indicators)
                    with open(out_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f'  Saved -> {out_path}')

                final_hv = data['indicators'][-1]['hv'] if data['indicators'] else float('nan')
                summary.setdefault((suite, pid, method), []).append(final_hv)

    print('\n' + '=' * 88)
    print(f'Final HV summary (mean +/- std over up to {len(args.seeds)} seeds)')
    for suite, pid in suite_pids:
        print(f'-- {suite}/pid{pid} --')
        for method in args.methods:
            hvs = summary.get((suite, pid, method))
            if hvs is None:
                continue
            arr = np.array(hvs, dtype=float)
            print(f'  {method:>20s} : {np.nanmean(arr):.4f} +/- {np.nanstd(arr):.4f}  (n={len(arr)})')
    print('=' * 88)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suites', nargs='+', default=['c10mop', 'in1kmop'], choices=['c10mop', 'in1kmop'])
    p.add_argument('--pids', type=int, nargs='+', default=None,
                    help='Pid subset to run for each suite (default: all pids defined '
                         'in BENCHMARK_META for that suite).')
    p.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(20)))
    p.add_argument('--pop_size', type=int, default=20)
    p.add_argument('--n_gen', type=int, default=60)
    p.add_argument('--n_doe', type=int, default=None,
                    help='SAMOS(2): initial DOE size (default: pop_size)')
    p.add_argument('--n_infill', type=int, default=None,
                    help='SAMOS(2): real evaluations per outer generation (default: pop_size)')
    p.add_argument('--n_gen_inner', type=int, default=20,
                    help='SAMOS(2): inner NSGA-II generations')
    p.add_argument('--inner_pop_size', type=int, default=None,
                    help='SAMOS(2): inner NSGA-II population size (default: pop_size x 10)')
    p.add_argument('--surrogate', default='XGBoost',
                    help='Surrogate for samos2-* methods (samos2-uncertainty auto-switches '
                         'to RFR if left at XGBoost). samos-xgb always uses XGBoost.')
    p.add_argument('--acq_kind', default='lcb', choices=['lcb', 'hvi'],
                    help='samos2-uncertainty acquisition kind.')
    p.add_argument('--kappa', type=float, default=2.0,
                    help='samos2-uncertainty LCB exploration weight.')
    p.add_argument('--results_root', default=os.path.join('results', 'samos', 'compare'),
                    help='Output root for per-seed pkls. Point at a dedicated '
                         'smoke-test folder when testing -- never write test '
                         'data into results/ (see CLAUDE.md).')
    p.add_argument('--no_norm', action='store_true')
    p.add_argument('--no_indicators', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    sys.exit(main(p.parse_args()))
