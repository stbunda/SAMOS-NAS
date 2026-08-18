"""experiments2/sample_selection_bias/run_bias.py -- campaign runner.

Runs one (tightness, role, hw, arm) cell of the sample-selection-bias grid
(config.py) for a set of seeds. SAMOS2 with XGBoost surrogates on both
objectives plus an XGBoost constraint surrogate per declared constraint, CDP
throughout, hard gate always ON and always keyed to #Params.

The one seam under study is ``arm``:

  feasible : today's behaviour. Gated points leave the archive, so the
             objective surrogates only ever see architectures under the size
             budget. Their constraint surrogate still trains on archive UNION
             the rejection log (which architectures FAILED is observable even
             when their objectives are not) -- so the arm isolates the
             OBJECTIVE surrogates' training set and nothing else.
  oracle   : identical run, except the objective surrogates additionally train
             on the gated points' true objectives (``F_oracle``, the outer
             problem's unmasked F). Physically impossible -- the whole point is
             that those values cannot be measured -- so it is the upper bound
             the realistic arm is scored against, not a proposed method.

Everything else is held fixed across the two arms: same seed, same operators,
same archive gate, same evaluation budget. Termination is EVALUATION-based
(``('n_evals', N)``), for the reason spelled out in scenario_run's
run_scenario.py: the gated DOE can redraw batches and every draw is a real
benchmark query, so an n_gen budget would hand the tighter cells more
evaluations for the "same" run.

Output layout
-------------
  {results_root}/{tightness}/{role}/{hw}/{arm}/seed_{N}.pkl

Every pkl is self-describing via data['meta'].

Examples
--------
  python experiments2/sample_selection_bias/run_bias.py --tightness T2 \\
      --role constr --hw edgegpu_latency --arm feasible oracle --seeds 0 1 2

  # smoke test -- never into results2/ (see CLAUDE.md)
  python experiments2/sample_selection_bias/run_bias.py --tightness T4 \\
      --role obj --hw edgegpu_latency --seeds 0 --pop_size 8 --n_evals 80 \\
      --probe_n 200 --results_root smoke_tests/sample_selection_bias
"""

import argparse
import os
import pickle
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import config as CF
from _bias import BiasCallback, build_probe
from problem.evoxbench.constrained_problem import (
    ConstrainedEvoXBenchProblem,
    ConstrainedSurrogateProblemEvox,
)
from problem.evoxbench.utils import bounds_with_override, get_benchmark
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.sampler import EvoxBenchSampler
from strategy.surrogate.models import XGBoost
from strategy.surrogate.samos2 import SAMOS2

DEFAULT_POP_SIZE = 20
DEFAULT_N_EVALS = 1200
DEFAULT_PROBE_N = 2000
N_GEN_INNER = 20


def build(spec, arm, benchmark, seed, pop_size, inner_pop_size=None):
    """(outer problem, SAMOS2) for one cell. The gate is #Params in every
    role, so the outer problem always carries gate_index/gate_tau and the
    algorithm always reads its violation off ``G_gate`` -- never off G, which
    holds the DECLARED constraints and in the 'obj' role does not mention
    #Params at all."""
    xl, xu = bounds_with_override(benchmark, None)
    obj_idx = spec['obj_indices']

    problem = ConstrainedEvoXBenchProblem(
        benchmark, obj_idx, spec['constr_indices'], spec['thresholds'],
        sense=spec['senses'], gate=True,
        gate_index=spec['gate_index'], gate_threshold=spec['gate_tau'],
        flip_obj_pos=spec['flip_obj_pos'])
    problem.xl, problem.xu = xl.astype(float), xu.astype(float)

    rng = np.random.RandomState(seed)
    surrogates = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in obj_idx]
    constr_surrogate = [XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                        for _ in spec['constr_indices']]
    if len(constr_surrogate) == 1:
        constr_surrogate = constr_surrogate[0]

    def factory(surrs, fitted_constr_surrogate=None):
        return ConstrainedSurrogateProblemEvox(
            surrs, obj_idx, list(range(len(obj_idx))), [], benchmark,
            spec['constr_indices'], spec['thresholds'], sense=spec['senses'],
            constr_surrogate=fitted_constr_surrogate)

    algorithm = SAMOS2(
        sampling=EvoxBenchSampler(xl, xu), surrogates=surrogates,
        surrogate_problem_factory=factory,
        predict_obj_indices=list(range(len(obj_idx))),
        crossover=IntegerUniformCrossover(prob=0.9),
        mutation=IntegerPointMutation(xl, xu),
        eliminate_duplicates=IntegerVectorDuplicateElimination(),
        n_doe=pop_size, n_infill=pop_size, n_gen_inner=N_GEN_INNER,
        ga_pop_size=inner_pop_size or pop_size * 10,
        use_subset_selection=True, inner_algorithm=NSGA2,
        constr_surrogate=constr_surrogate,
        hard_gate=True,
        gate_g_fn=lambda pop: np.asarray(pop.get('G_gate'), dtype=float),
        constr_observable_when_gated=spec['observable_constr'],
        oracle_train_key=('F_oracle' if arm == 'oracle' else None),
    )
    return problem, algorithm


def run_single(space, tightness, role, hw, arm, seed, pop_size, n_evals,
               probe_n=DEFAULT_PROBE_N, inner_pop_size=None):
    """Run one (space, tightness, role, hw, arm, seed) cell -> result dict."""
    np.random.seed(seed)
    random.seed(seed)

    spec = CF.cell(tightness, role, hw, space)
    benchmark = get_benchmark(spec['suite'], spec['pid'])
    problem, algorithm = build(spec, arm, benchmark, seed, pop_size,
                               inner_pop_size)

    # Feasibility for scoring = declared constraints AND the size gate (see
    # config.scoring_constraints). The probe seed is fixed, NOT the run seed:
    # every cell and arm must be diagnosed against the same architectures.
    score_idx, score_thr, score_sns = CF.scoring_constraints(spec)
    probe = build_probe(benchmark, spec, problem.xl, problem.xu, n=probe_n)
    callback = BiasCallback(
        benchmark, spec['obj_indices'], score_idx, score_thr,
        sense=score_sns, ref_point=spec['ref_point'],
        flip_obj_pos=spec['flip_obj_pos'], probe=probe)

    results = minimize(problem=problem, algorithm=algorithm,
                       termination=('n_evals', n_evals), seed=seed,
                       callback=callback, save_history=False, verbose=True)
    algo = results.algorithm
    data = algo.callback.data

    # The rejection log makes the gated-out region reconstructable post hoc.
    data['rejected_X'] = (np.asarray(algo._rejected_X) if algo._rejected_X is not None
                          else np.empty((0, problem.n_var)))
    data['rejected_G'] = (np.asarray(algo._rejected_G) if algo._rejected_G is not None
                          else np.empty(0))

    data['meta'] = dict(
        space=space, tightness=tightness, role=role, hw=hw, arm=arm, seed=seed,
        suite=spec['suite'], pid=spec['pid'], exact=spec['exact'],
        obj_metrics=spec['obj_metrics'], constr_metrics=spec['constr_metrics'],
        obj_indices=spec['obj_indices'], constr_indices=spec['constr_indices'],
        thresholds=spec['thresholds'], senses=spec['senses'],
        flip_obj_pos=spec['flip_obj_pos'],
        observable_constr=spec['observable_constr'],
        gate_metric=spec['gate_metric'], gate_tau=spec['gate_tau'],
        size_feasible_fraction=spec['size_feasible_fraction'],
        score_constr_indices=score_idx, score_thresholds=score_thr,
        score_senses=score_sns, ref_point=spec['ref_point'],
        method='samos', handler='h4-cdp', mode='hard',
        pop_size=pop_size, n_evals=n_evals, probe_n=int(len(probe[0])),
        n_gen=int(algo.n_gen or 0), n_eval_realised=int(algo.evaluator.n_eval),
        n_hf_evaluated=int(algo.n_hf_evaluated),
        n_hf_evaluable=int(algo.n_hf_feasible),
        n_archive=int(len(algo._archive)),
        n_rejected=int(len(data['rejected_X'])),
        inner_pop_size=inner_pop_size, config='sample_selection_bias',
    )
    return data


def main(args):
    space = args.space
    tights = args.tightness or list(CF.tightness(space))
    roles = args.role or list(CF.ROLES)
    hws = args.hw or list(CF.hw(space))
    arms = args.arm or list(CF.ARMS)
    results_root = args.results_root or CF.results_root(space)

    # --tightness/--hw cannot be argparse choices (they are per-space), so
    # validate here. Without this a caller that names another space's metric
    # -- a runner script that forgets to pass --space is the realistic way in
    # -- gets a bare KeyError from deep inside config.cell, once per seed.
    for name, given, legal in (('tightness', tights, CF.tightness(space)),
                               ('hw', hws, CF.hw(space))):
        bad = [v for v in given if v not in legal]
        if bad:
            raise SystemExit(
                f'ERROR: {bad} is not a valid --{name} for --space {space} '
                f'(valid: {list(legal)}). If you meant a different benchmark, '
                f'pass --space explicitly.')

    pairs = [(r, h, a) for r in roles for h in hws for a in arms
             if not CF.excluded(r, h, space)]
    for r in roles:
        for h in hws:
            if CF.excluded(r, h, space):
                print(f'[SKIP] {r} x {h}: empty feasible set '
                      f"(see config.SPACES['{space}']['excluded']).")
    total = len(args.seeds) * len(tights) * len(pairs)
    summary, run_i = {}, 0

    for tightness in tights:
        for seed in args.seeds:
            for role, hw, arm in pairs:
                run_i += 1
                save_dir = os.path.join(results_root, tightness, role, hw, arm)
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f'seed_{seed}.pkl')

                if os.path.exists(out_path) and not args.overwrite:
                    print(f'[SKIP {run_i}/{total}] {tightness}/{role}/{hw}/{arm}/'
                          f'seed_{seed} already exists')
                    with open(out_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    frac = CF.tightness(space)[tightness]['feasible_fraction']
                    print(f'\n[RUN {run_i}/{total}] {space} tightness={tightness} '
                          f'(size feas={frac:.1%}) '
                          f'role={role} hw={hw} arm={arm} seed={seed} '
                          f'pop={args.pop_size} n_evals={args.n_evals}')
                    data = run_single(space, tightness, role, hw, arm, seed,
                                      args.pop_size, args.n_evals,
                                      probe_n=args.probe_n,
                                      inner_pop_size=args.inner_pop_size)
                    with open(out_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f'  Saved -> {out_path}  (n_gen={data["meta"]["n_gen"]}, '
                          f'n_eval={data["meta"]["n_eval_realised"]}, '
                          f'evaluable={data["meta"]["n_hf_evaluable"]}/'
                          f'{data["meta"]["n_hf_evaluated"]})')

                ind = data['indicators'][-1] if data['indicators'] else {'hv': float('nan')}
                probe_last = data['probe'][-1] if data.get('probe') else {}
                summary.setdefault((tightness, role, hw, arm), []).append(
                    (ind['hv'], probe_last.get('mae_gated_o0', float('nan')),
                     data['meta']['n_hf_evaluable'] / max(1, data['meta']['n_hf_evaluated'])))

    print('\n' + '=' * 96)
    print('Final feasible-HV / gated-side surrogate MAE (objective 0) / evaluable rate, '
          f'mean over up to {len(args.seeds)} seeds')
    for key, rows in summary.items():
        a = np.array(rows, dtype=float)
        print(f'  {"/".join(key):<52s} hv={np.nanmean(a[:, 0]):.4f} '
              f'mae_gated={np.nanmean(a[:, 1]):.4f} '
              f'evaluable={np.nanmean(a[:, 2]):.3f}  (n={len(rows)})')
    print('=' * 96)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--space', default=CF.DEFAULT_SPACE, choices=list(CF.SPACES),
                   help='Benchmark: NB201 (tabular, exact) or MobileNetV3 '
                        '(surrogate, unsaturated). Sets suite/pid, thresholds '
                        'and the default results root.')
    # Defaults are None and resolved against --space in main(), since the
    # legal tightness levels and hardware metrics are per-space.
    p.add_argument('--tightness', nargs='+', default=None)
    p.add_argument('--role', nargs='+', default=None, choices=list(CF.ROLES))
    p.add_argument('--hw', nargs='+', default=None)
    p.add_argument('--arm', nargs='+', default=None, choices=list(CF.ARMS))
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE)
    p.add_argument('--n_evals', type=int, default=DEFAULT_N_EVALS,
                   help="Evaluation budget -- termination is ('n_evals', N).")
    p.add_argument('--probe_n', type=int, default=DEFAULT_PROBE_N,
                   help='Architectures in the fixed surrogate-diagnostic probe set.')
    p.add_argument('--inner_pop_size', type=int, default=None,
                   help='Surrogate-side inner GA population. Default pop_size * 10.')
    p.add_argument('--results_root', default=None,
                   help="Default: the space's own root (config.results_root). "
                        'Point at a dedicated smoke-test folder when testing -- '
                        'never write test data into results2/ (see CLAUDE.md).')
    p.add_argument('--overwrite', action='store_true')
    sys.exit(main(p.parse_args()))
