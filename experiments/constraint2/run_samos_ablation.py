"""experiments/constraint2/run_samos_ablation.py --- run the two MISSING
corners of the samos vs samos-cheap 2x2 ablation (Experiment 2 of
analyse_samos.py).

samos and samos-cheap confound two independent knobs:

                    constraint exact          constraint predicted
    FLOPs exact     samos-cheap               samos-cheapflops   (this file)
    FLOPs pred      samos-cheapconstr         samos
                    (this file)

The diagonal (samos, samos-cheap) is already on disk from the main campaign;
this runner produces the two off-diagonal corners so the axes can be
decoupled:

  samos-cheapflops  : FLOPs objective EXACT, constraint PREDICTED. Same
                      objective split as samos-cheap, but the constraint gets
                      a surrogate like samos.
  samos-cheapconstr : FLOPs objective PREDICTED, constraint EXACT. Objectives
                      all predicted like samos, but the cheap constraint stays
                      exact like samos-cheap.

Both are just recombinations of the two knobs build_algorithm already
implements, so this file adds NOTHING to run_constraint.py: it monkeypatches
run_constraint.build_algorithm to recognize the two new method names (falling
through to the original for everything else) and reuses run_single unchanged.
The saved pkls -- path layout, self-describing ``meta``, everything the
callback records -- are byte-for-byte what run_constraint would write, so
analyse_samos.py / analyse_constraint.py read them with no special-casing.

Scope: the ablation lives on the cheap SINGLE-constraint scenarios (s1/s3,
where #Params is structural and so free to make exact) and the two handlers
whose wiring is self-contained -- h4-cdp (native constrained dominance, the
hard default) and h2-penalty (the soft default). The handler is held FIXED
across a run so the measured axis effect is not confounded by it; the default
mirrors each scenario's own default handler. Seeds are shared with the
existing samos/samos-cheap runs (run_single seeds np.random / random per seed
identically for every method), so the four corners pair blockwise seed-for-seed
in analyse_samos.py's Experiment 2.

Examples
--------
  # both corners, both cheap scenarios, all 4 cheap instances, 20 seeds:
  python experiments/constraint2/run_samos_ablation.py

  # one instance, one scenario, a quick check:
  python experiments/constraint2/run_samos_ablation.py \
      --scenarios s1 --instances c10mop/4 --seeds 0 1 2

  # smoke test -- NEVER write synthetic runs into the real results tree:
  python experiments/constraint2/run_samos_ablation.py \
      --scenarios s1 --instances c10mop/4 --seeds 0 \
      --results_root smoke_tests/samos_ablation
"""

import argparse
import os
import pickle
import sys

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

import numpy as np
from functools import partial

import run_constraint as rc
from run_constraint import (SCENARIOS, SCENARIO_INSTANCES, OBJ_METRICS,
                            THRESHOLDS_BY_VARIANT,
                            DESIGN_FEASIBLE_FRACTION_BY_VARIANT,
                            constr_metrics_for, _inner_ga, _obj_split,
                            _ConstraintsAsPenaltyMO)

from problem.evoxbench.benchmark_meta import BENCHMARK_META, metric_index
from problem.evoxbench.constrained_problem import ConstrainedSurrogateProblemEvox
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.sampler import EvoxBenchSampler
from strategy.surrogate.models import XGBoost
from strategy.surrogate.samos2 import SAMOS2

# The two new corners, by which knob each makes exact. Names double as the
# result-directory names analyse_samos.py's --ablation_* flags default to.
ABLATION_METHODS = {
    'samos-cheapflops':  dict(flops_exact=True,  constr_exact=False),
    'samos-cheapconstr': dict(flops_exact=False, constr_exact=True),
}

# Handlers whose constraint wiring is self-contained enough to reproduce here
# without importing run_constraint's full handler switch. h4-cdp/h2-penalty
# are the hard/soft scenario defaults -- the ones the ablation actually needs.
SUPPORTED_HANDLERS = ('h4-cdp', 'h2-penalty')

# The ablation is defined only where the constraint is a cheap structural
# metric that CAN be made exact -- the single-constraint cheap scenarios.
CHEAP_SCENARIOS = ('s1', 's3')

# The ablation writes to its OWN results tree, a dedicated subfolder of the
# campaign's results/constraint2 (kept apart from the s1-s12 scenario dirs the
# campaign analysis globs, so nothing collides), so all four 2x2 corners sit
# together for analysis. The two diagonal reference corners (samos,
# samos-cheap) are copied in from the campaign at the scenario-default
# handlers -- see analyse_samos.py --ablation_root and this file's header.
ABLATION_RESULTS_ROOT = os.path.join('results', 'constraint2', 'samos_ablation')

# Each cheap scenario's default handler (mirrors run_constraint._resolve_handlers).
_DEFAULT_HANDLER = {'hard': 'h4-cdp', 'soft': 'h2-penalty'}


# ─── the two new corners, patched into build_algorithm ─────────────────────

def _ablation_surrogates(spec, obj_indices, constr_indices, suite, pid, rng):
    """(predict_pos, real_pos, surrogates, constr_surrogate) for one 2x2
    corner, using the same primitives as run_constraint.build_algorithm:
    _obj_split for the objective knob, per-slot XGBoost / None for the
    constraint knob, and the single-constraint scalar-collapse convention."""
    cheap_cols = set(BENCHMARK_META[suite][pid].get('cheap_obj_indices', []))

    if spec['flops_exact']:
        # FLOPs (a cheap objective column) stays exact -- identical objective
        # split to samos-cheap.
        predict_pos, real_pos = _obj_split(obj_indices, cheap_cols)
    else:
        # Every objective predicted -- identical to samos.
        predict_pos, real_pos = list(range(len(obj_indices))), []
    surrogates = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_pos]

    if spec['constr_exact']:
        # Cheap constraint columns exact (None slot), expensive ones predicted
        # -- the samos-cheap constraint convention (all-exact -> None).
        constr_surrogate = [None if ci in cheap_cols
                            else XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                            for ci in constr_indices]
        if not any(s is not None for s in constr_surrogate):
            constr_surrogate = None
    else:
        # Constraint always predicted -- the samos constraint convention.
        constr_surrogate = [XGBoost(100, seed=rng.randint(0, 2**31 - 1))
                            for _ in constr_indices]

    # Single constraint keeps the scalar surrogate object (bit-identical wiring
    # to the original single-constraint campaign path).
    if isinstance(constr_surrogate, list) and len(constr_surrogate) == 1:
        constr_surrogate = constr_surrogate[0]
    return predict_pos, real_pos, surrogates, constr_surrogate


_ORIG_BUILD = rc.build_algorithm


def _patched_build(method, benchmark, suite, pid, obj_indices, constr_indices,
                   thresholds, handler, seed, pop_size, n_doe, n_infill,
                   n_gen_inner, inner_pop_size, penalty, n_gen, gated):
    """build_algorithm extended with the two ablation corners; every other
    (method, handler) is delegated untouched to the original."""
    if method not in ABLATION_METHODS:
        return _ORIG_BUILD(method, benchmark, suite, pid, obj_indices,
                           constr_indices, thresholds, handler, seed, pop_size,
                           n_doe, n_infill, n_gen_inner, inner_pop_size,
                           penalty, n_gen, gated)
    if handler not in SUPPORTED_HANDLERS:
        raise ValueError(
            f'run_samos_ablation supports handlers {SUPPORTED_HANDLERS} only '
            f'(self-contained wiring); got {handler!r}. Extend SUPPORTED_HANDLERS '
            f'and the tail below to add more.')

    xl = np.asarray(benchmark.search_space.lb, dtype=int)
    xu = np.asarray(benchmark.search_space.ub, dtype=int)
    sampler   = EvoxBenchSampler(xl, xu)
    crossover = IntegerUniformCrossover(prob=0.9)
    mutation  = IntegerPointMutation(xl, xu)
    n_doe_    = n_doe if n_doe is not None else pop_size
    n_infill_ = n_infill if n_infill is not None else pop_size
    inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

    rng = np.random.RandomState(seed)
    predict_pos, real_pos, surrogates, constr_surrogate = _ablation_surrogates(
        ABLATION_METHODS[method], obj_indices, constr_indices, suite, pid, rng)

    # h2-penalty penalizes the inner surrogate problem; h4-cdp needs no wrap
    # (native CDP via out['G'] + feasibility-first RankAndCrowding).
    wrap_inner = ((lambda inner: _ConstraintsAsPenaltyMO(inner, penalty=penalty))
                  if handler == 'h2-penalty' else None)

    def factory(surrs, fitted_constr_surrogate=None):
        inner = ConstrainedSurrogateProblemEvox(
            surrs, obj_indices, predict_pos, real_pos, benchmark,
            constr_indices, thresholds, constr_surrogate=fitted_constr_surrogate)
        return wrap_inner(inner) if wrap_inner is not None else inner

    algorithm = SAMOS2(
        sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
        predict_obj_indices=predict_pos,
        crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
        n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
        constr_surrogate=constr_surrogate, hard_gate=gated,
    )
    return algorithm, True, {}


rc.build_algorithm = _patched_build


# ─── driver ────────────────────────────────────────────────────────────────

def _parse_instances(scenario, values):
    """Resolve --instances ('suite/pid' strings, or None for all) against the
    scenario's own cheap instance set."""
    allowed = SCENARIO_INSTANCES[SCENARIOS[scenario]['constr']]
    if not values:
        return allowed
    picked = []
    for v in values:
        suite, pid = v.split('/')
        key = (suite, int(pid))
        if key not in allowed:
            raise SystemExit(f'{v} is not a cheap instance of {scenario}; valid: {allowed}')
        picked.append(key)
    return picked


def run(args):
    budget_folder = f'B{args.n_gen * args.pop_size}_P{args.pop_size}'
    thresh_dict = THRESHOLDS_BY_VARIANT[args.threshold_variant]
    feas_dict   = DESIGN_FEASIBLE_FRACTION_BY_VARIANT[args.threshold_variant]

    work = []   # (scenario, suite, pid, handler, method, seed)
    for scenario in args.scenarios:
        cfg = SCENARIOS[scenario]
        handler = args.handler or _DEFAULT_HANDLER[cfg['mode']]
        if handler not in SUPPORTED_HANDLERS:
            raise SystemExit(f'--handler must be one of {SUPPORTED_HANDLERS}; got {handler!r}')
        for suite, pid in _parse_instances(scenario, args.instances):
            for method in args.methods:
                for seed in args.seeds:
                    work.append((scenario, suite, pid, handler, method, seed))

    print(f'[ablation] {len(work)} runs: methods={args.methods} '
          f'scenarios={args.scenarios} seeds={list(args.seeds)}')
    for i, (scenario, suite, pid, handler, method, seed) in enumerate(work, 1):
        cfg = SCENARIOS[scenario]
        constr_metrics = constr_metrics_for(scenario, suite, pid)
        assert len(constr_metrics) == 1, (
            f'{scenario}/{suite}/pid{pid} is multi-constraint; the ablation is '
            f'single-constraint only.')
        constr_metric = constr_metrics[0]
        threshold     = thresh_dict[(suite, pid)][constr_metric]
        feas_frac     = feas_dict[(suite, pid)][constr_metric]

        save_dir = os.path.join(args.results_root, scenario, suite, f'pid{pid}',
                                budget_folder, method, handler)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f'seed_{seed}.pkl')
        if os.path.exists(out_path) and not args.overwrite:
            print(f'[SKIP {i}/{len(work)}] {method}/{handler} {scenario} '
                  f'{suite}/pid{pid} seed_{seed} exists')
            continue

        print(f'\n[RUN {i}/{len(work)}] {scenario} {suite}/pid{pid} '
              f'(mode={cfg["mode"]}, {constr_metric}<={threshold:.4f})  '
              f'method={method}  handler={handler}  seed={seed}')
        data = rc.run_single(
            method, scenario, suite, pid, handler, seed, args.pop_size,
            args.n_gen, args.n_doe, args.n_infill, args.n_gen_inner,
            args.inner_pop_size, args.penalty,
            threshold_variant=args.threshold_variant)
        # Self-describing meta, identical schema to run_constraint.main (these
        # are single-constraint, non-b0 runs, so no tuple/joint/search-obj
        # fields). inner_ga resolves to 'nsga2' for h4-cdp / h2-penalty.
        data['meta'] = dict(
            suite=suite, pid=pid, scenario=scenario,
            obj_metrics=OBJ_METRICS, constr_metric=constr_metric,
            constr_type=cfg['constr'], threshold=threshold,
            design_feasible_fraction=feas_frac,
            mode=cfg['mode'], gate=(cfg['mode'] == 'hard'),
            handler=handler, method=method, seed=seed,
            pop_size=args.pop_size, n_gen=args.n_gen,
            n_gen_inner=args.n_gen_inner, penalty=args.penalty,
            inner_ga=_inner_ga(method, handler), config='c2',
            threshold_variant=args.threshold_variant,
        )
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'  Saved -> {out_path}')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', nargs='+', default=list(ABLATION_METHODS),
                   choices=list(ABLATION_METHODS),
                   help='which 2x2 corner(s) to run (default: both)')
    p.add_argument('--scenarios', nargs='+', default=list(CHEAP_SCENARIOS),
                   choices=list(CHEAP_SCENARIOS),
                   help='cheap single-constraint scenarios only')
    p.add_argument('--instances', nargs='+', default=None,
                   help="'suite/pid' (e.g. c10mop/4); default: all cheap instances")
    p.add_argument('--handler', default=None, choices=list(SUPPORTED_HANDLERS),
                   help='held fixed across the run; default: the scenario default '
                        '(h4-cdp hard / h2-penalty soft)')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(20)),
                   help='shared with the existing samos/samos-cheap runs for pairing')
    p.add_argument('--pop_size', type=int, default=20)
    p.add_argument('--n_gen', type=int, default=60)
    p.add_argument('--n_doe', type=int, default=None)
    p.add_argument('--n_infill', type=int, default=None)
    p.add_argument('--n_gen_inner', type=int, default=20)
    p.add_argument('--inner_pop_size', type=int, default=None)
    p.add_argument('--penalty', type=float, default=1.0)
    p.add_argument('--threshold_variant', default='q25', choices=['q10', 'q25', 'q50'])
    p.add_argument('--results_root', default=ABLATION_RESULTS_ROOT,
                   help='the ablation results tree (default results/samos_ablation, '
                        'kept apart from the campaign); point at a dedicated smoke '
                        'folder when testing -- never write test runs into a real '
                        'results tree')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    sys.exit(run(args))
