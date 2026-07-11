"""experiments/constraint/run_constraint.py --- S1-S4 constraint-scenario campaign runner. 
Cloned from experiments/samos/compare_methods.py's run_single/main structure and output conventions 
(skip-if-exists per-seed pkl, EvoxBenchCallback-style ``data`` dict, budget-folder naming), but on
top of the constrained problem pair (problem/evoxbench/constrained_problem.py) and the 
feasibility-aware callback (problem/evoxbench/callbacks.py ::FeasibilityAwareEvoxBenchCallback).

See text/plans/EXPERIMENT_PLAN_constraint_scenarios.md for the campaign design and 
experiments/constraint/THRESHOLDS.md for threshold provenance.

Instances
---------
Selected via --suite/--pid (default c10mop/4, so every pre-existing
invocation behaves identically):

  c10mop/pid4  : NATS, 5 vars.
  in1kmop/pid9 : MobileNetV3, 21 vars.

Both share the metric column order (Err, #Params, FLOPs, Latency), the same obj_indices=[0, 2], 
the same constraint columns (1 = #Params, 3 = Latency) and the same BENCHMARK_META cheap_obj_indices=[1, 2], 
so the samos-cheap split logic is identical. They differ in normalized_objectives (pid4 True, pid9 False), 
which ConstrainedEvoXBenchProblem's normalize-once convention already absorbs: either way the 
constraint metric the problem sees lives in the benchmark-normalized space THRESHOLDS.md computed the per-instance
thresholds in (verified empirically for pid9: ~49% of a 2k random sample is feasible at each threshold, 
matching the ~50% design intent).

Scenarios
---------
Scenario definitions (constraint metric + handler mode) are instance-
independent; only the numeric threshold varies per instance (THRESHOLDS
dict below). Objectives are always {Err, FLOPs} (obj_indices=[0, 2]).

  s1 : #Params <= T_params, hard   (cheap constraint metric)
  s2 : Latency <= T_latency, hard  (expensive constraint metric)
  s3 : #Params <= T_params, soft   (cheap constraint metric)
  s4 : Latency <= T_latency, soft  (expensive constraint metric)

Handlers
--------
--handler adds the handler axis on top of scenario x method x instance.
Handler wiring is SCENARIO-INDEPENDENT: each handler fully defines its
mechanism; the scenario only picks metric/threshold and (via its hard/soft
mode) the DEFAULT handler. Omitting --handler runs the scenario default
(hard -> h4-cdp, soft -> h2-penalty); --all_handlers runs the
scenario's full row (SCENARIO_HANDLERS) instead. No validation rejects
an off-default scenario x handler combination: stress / wrong-model rows
(h2/h3 penalties on hard scenarios, h1/h4 on soft ones) are deliberate.

  h4-cdp             : native CDP, zero extra wiring. Both constrained
                       problems define out['G'], so pymoo's RankAndCrowding
                       survival (outer archive selection AND the SAMOS2
                       inner NSGA-II) is feasibility-first natively -- see
                       problem/evoxbench/constrained_problem.py.
  h2-penalty         : static penalty. Wraps ONLY the inner surrogate
                       problem in _ConstraintsAsPenaltyMO (pymoo 0.6.1.1's
                       own ConstraintsAsPenalty mis-handles n_obj>1, see its
                       docstring) at the --penalty weight; outer problem and
                       saved pkls keep unpenalized F / raw G.
  h3-adaptive-penalty: strategy.constraints.AdaptivePenaltyProblem. One
                       controller per run; each outer generation the factory
                       adapt()s the weight on the live archive's feasible
                       fraction (target 0.5, c=1.2), then wrap()s the fresh
                       inner problem. Needs copy_algorithm=False (the
                       factory closure reads the live algorithm's archive;
                       pymoo's default deepcopy severs that link).
  h5-eps             : strategy.constraints.EpsilonRelaxation. Outer-clocked
                       epsilon schedule (eps0 = mean DOE CV, linear to 0 at
                       50% of --n_gen); per outer generation the factory does
                       maybe_init_eps0 / wrap / advance. Needs
                       copy_algorithm=False (same live-archive reason as h3).
  h6-sr              : strategy.constraints.DominanceStochasticRanking
                       (Pf=0.45, seeded per run) as the inner NSGA-II
                       survival, via partial(NSGA2, survival=...).
  h1-rejection       : rejection. random -> RejectionSampling around the
                       base sampler with an exact make_benchmark_g_fn;
                       samos/samos-cheap -> RejectionInfillSelector on the
                       C3 infill seam (rejects on the inner problem's G:
                       predicted for samos, exact for cheap constraints),
                       AND the inner NSGA-II survival is a RankAndCrowding
                       whose filter_infeasible is set False so the row
                       isolates pure rejection rather than a rejection+CDP
                       hybrid. pymoo 0.6.1.1's RankAndCrowding
                       constructor hardcodes filter_infeasible=True (no
                       kwarg), but it is a plain instance attribute
                       (Survival.__init__ sets it), so it is flipped
                       post-construction. ponytail: NSGA2's binary_
                       tournament mating selection still compares CV-first
                       when a parent is infeasible -- only the survival
                       mechanism is replaced, so that residual feasibility
                       pressure is accepted and documented, not patched.

For method 'random', only h1-rejection and the scenario default are run:
every other handler acts purely on selection pressure that RandomGA does not
have, so those runs would be byte-identical duplicates of its default run --
the runner prints [SKIP] instead and analysis replicates the row.

Output layout (round 2 adds the handler level):
  {results_root}/{scenario}/{suite}/pid{pid}/{budget}/{method}/{handler}/seed_{N}.pkl
Round-1 flat pkls ({method}/seed_N.pkl) are migrated by COPY -- never moved
or deleted -- into the scenario-default handler subdir via --migrate
(s1/s2 -> h4-cdp, s3/s4 -> h2-penalty).

Methods
-------
  random       : RandomGA (no surrogate, no selection pressure) on
                 ConstrainedEvoXBenchProblem, exact benchmark G. Handler mode
                 (hard/soft) is irrelevant to it -- it has no ranking to
                 penalize or CDP-filter differently -- so it runs identically
                 in hard and soft scenarios; results should therefore be
                 near-identical between e.g. s1 and s3 for this method (any
                 difference is RNG-stream noise from the extra ConstraintsAs
                 Penalty wrapper never being constructed for this method, not
                 a real behavioural change).
  samos        : SAMOS2, ALL output objectives predicted (predict=[0, 1],
                 real=[]), XGBoost surrogates (one per predicted position,
                 per-seed subseeded like compare_methods.py). The constraint
                 is ALWAYS predicted for this method (constr_surrogate=
                 XGBoost), regardless of scenario.
  samos-cheap  : SAMOS2 cheap-real split, derived from BENCHMARK_META's
                 cheap_obj_indices for the instance ({1, 2} = #Params, FLOPs
                 for both supported instances): predict=[0] (Err), real=[1]
                 (FLOPs). Constraint: exact
                 (constr_surrogate=None) when constr_index is itself a cheap
                 column (s1/s3, #Params) -- this is H7 (exact cheap filter)
                 for free; predicted (constr_surrogate=XGBoost) otherwise
                 (s2/s4, Latency, expensive).

Known simplifications:
  (a) SAMOS2's archive-level RankAndCrowding seeding (top_pop in _infill)
      stays feasibility-first even in soft scenarios -- it operates on the
      *outer* archive, which always carries raw (unpenalized) G.
  (b) Infill/subset selection compares penalized candidate F (from the
      ConstraintsAsPenalty-wrapped inner problem, soft scenarios only)
      against unpenalized archive F (the outer archive) when ranking for
      diversity. Both are accepted for round 1; see the experiment plan.

Examples
--------
  python experiments/constraint/run_constraint.py --scenario s1
  python experiments/constraint/run_constraint.py --scenario s1 --suite in1kmop --pid 9
  python experiments/constraint/run_constraint.py --scenario s2 --method samos --handler h3-adaptive-penalty
  python experiments/constraint/run_constraint.py --scenario s1 --all_handlers
  python experiments/constraint/run_constraint.py --migrate                     # copy round-1 flat pkls
  python experiments/constraint/run_constraint.py --scenario s2 --method samos-cheap \\
      --pop_size 8 --n_gen 3 --n_gen_inner 4 --results_root /tmp/smoke
"""

import argparse
import glob
import os
import pickle
import random
import shutil
import sys
from functools import partial

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'samos'))  # for _common
sys.path.insert(0, _REPO_ROOT)

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.core.individual import calc_cv
from pymoo.optimize import minimize
from pymoo.util.misc import from_dict

from _common import _make_surrogate

from problem.evoxbench.benchmark_meta import BENCHMARK_META, metric_index
from problem.evoxbench.callbacks import FeasibilityAwareEvoxBenchCallback
from problem.evoxbench.constrained_problem import (
    ConstrainedEvoXBenchProblem,
    ConstrainedSurrogateProblemEvox,
)
from problem.evoxbench.utils import get_benchmark
from strategy.algorithm.algorithms import RandomGA
from strategy.constraints import (
    AdaptivePenaltyProblem,
    DominanceStochasticRanking,
    EpsilonRelaxation,
    RejectionInfillSelector,
    RejectionSampling,
    feasible_fraction,
    make_benchmark_g_fn,
)
from strategy.genetics.duplicate import IntegerVectorDuplicateElimination
from strategy.operations.crossover import IntegerUniformCrossover
from strategy.operations.mutation import IntegerPointMutation
from strategy.sampler import EvoxBenchSampler
from strategy.surrogate.models import XGBoost
from strategy.surrogate.samos2 import SAMOS2

# ─── default instance (module-level for backward compatibility) ──────────────
# analyse_constraint.py imports SUITE / PID directly; they remain the
# campaign's original default instance. The runner itself takes --suite/--pid
# and only falls back to these via the CLI defaults.
SUITE       = 'c10mop'
PID         = 4
OBJ_INDICES = [0, 2]   # Err, FLOPs (same benchmark column order on all instances)

# ─── per-instance thresholds (config, not code) ──────────────────────────────
# Provenance: experiments/constraint/THRESHOLDS.md -- per instance, the
# median of the constraint metric over a fixed-seed (0) 10k random sample, in
# the space the constrained problem operates in (benchmark.evaluate() output,
# benchmark.normalize() applied ONLY when not benchmark.normalized_objectives
# -- c10mop/pid4 is natively normalized, in1kmop/pid9 is normalized by that
# convention). ~50% of the sample is feasible at each
# threshold by construction. Keyed by (suite, pid), then by the same metric
# names SCENARIOS uses.
THRESHOLDS = {
    ('c10mop', 4): {
        '#Params': 0.4383147965648318,
        'Latency': 0.45870737801039463,
    },
    ('in1kmop', 9): {
        '#Params': 0.554389375817845,
        'Latency': 0.4606726558251625,
    },
}

# SCENARIOS is data, not branching code, and is
# instance-independent: the constraint metric and handler mode define the
# scenario; the numeric threshold is looked up per instance in THRESHOLDS.
SCENARIOS = {
    's1': dict(constr_metric='#Params', mode='hard'),
    's2': dict(constr_metric='Latency', mode='hard'),
    's3': dict(constr_metric='#Params', mode='soft'),
    's4': dict(constr_metric='Latency', mode='soft'),
}

METHODS = ['random', 'samos', 'samos-cheap']

# ─── handler axis ─────────────────────────────────────────────────────────────
HANDLERS = ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty',
            'h4-cdp', 'h5-eps', 'h6-sr']

# Scenario-default handler = the round-1 wiring, keyed by the scenario's mode.
DEFAULT_HANDLER = {'hard': 'h4-cdp', 'soft': 'h2-penalty'}

# Scenario -> handler-row matrix (config, not code). Includes each
# scenario's default; --all_handlers runs exactly this row. S2 carries the
# h2/h3 penalty stress rows; s3/s4 carry the h1/h4 wrong-model controls.
SCENARIO_HANDLERS = {
    's1': ['h1-rejection', 'h2-penalty', 'h4-cdp', 'h5-eps', 'h6-sr'],
    's2': ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp', 'h5-eps', 'h6-sr'],
    's3': ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp', 'h6-sr'],
    's4': ['h1-rejection', 'h2-penalty', 'h3-adaptive-penalty', 'h4-cdp', 'h6-sr'],
}

# Handlers that act on selection pressure only are meaningless for RandomGA:
# for method 'random' the runner executes only the scenario default
# and h1-rejection, and prints [SKIP] for the rest.
RANDOM_VALID_EXTRA = {'h1-rejection'}


class _ConstraintsAsPenaltyMO(ConstraintsAsPenalty):
    """Deviation from the plan's plain ``ConstraintsAsPenalty`` (found during
    verification): pymoo 0.6.1.1's ``ConstraintsAsPenalty.do()`` computes
    ``F + penalty * np.reshape(CV, F.shape)``. ``calc_cv`` returns one
    aggregated violation per individual, shape ``(n,)`` -- that reshape only
    works when ``F`` is single-column (``n.size == n_obj==1 * n``). Our inner
    problems have ``n_obj=len(OBJ_INDICES)=2`` (Err, FLOPs) with one
    constraint, so ``CV.size == n`` but ``F.size == 2n``, and pymoo's own
    reshape raises ``ValueError: cannot reshape array of size n into shape
    (n,2)`` (confirmed against the installed build). This override is
    otherwise identical to pymoo's; it only replaces the exact reshape with
    ``CV.reshape(-1, 1)`` broadcasting, applying the same per-individual
    penalty to every objective column -- the standard multi-objective
    generalization of static penalty (H2)."""

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)
        F, G, H = from_dict(out, 'F', 'G', 'H')
        out['__F__'], out['__G__'], out['__H__'] = F, G, H
        CV = calc_cv(G=G, H=H)
        out['F'] = F + self.penalty * CV.reshape(-1, 1)
        del out['G']
        del out['H']
        return out


def _obj_split(obj_indices, cheap_cols):
    """(predict_pos, real_pos): output-column POSITIONS into obj_indices,
    split by whether the underlying benchmark column is cheap."""
    real_pos    = [pos for pos, col in enumerate(obj_indices) if col in cheap_cols]
    predict_pos = [pos for pos in range(len(obj_indices)) if pos not in real_pos]
    return predict_pos, real_pos


def build_algorithm(method, benchmark, suite, pid, obj_indices, constr_index,
                     threshold, handler, seed, pop_size, n_doe, n_infill,
                     n_gen_inner, inner_pop_size, penalty, n_gen):
    """Returns ``(algorithm, copy_algorithm)``; the second element is the
    ``copy_algorithm`` value ``minimize`` must be called with. It is False
    only for h3/h5 (their per-generation state objects read the live
    algorithm's archive through the factory closure's ``algo_ref``; pymoo's
    default algorithm deepcopy would sever that handle) and pymoo's default
    True everywhere else."""
    xl = np.asarray(benchmark.search_space.lb, dtype=int)
    xu = np.asarray(benchmark.search_space.ub, dtype=int)

    sampler   = EvoxBenchSampler(xl, xu)
    crossover = IntegerUniformCrossover(prob=0.9)
    mutation  = IntegerPointMutation(xl, xu)
    elim      = IntegerVectorDuplicateElimination()

    n_doe_    = n_doe if n_doe is not None else pop_size
    n_infill_ = n_infill if n_infill is not None else pop_size
    inner_ps  = inner_pop_size if inner_pop_size is not None else pop_size * 10

    if method == 'random':
        # No surrogate, no selection pressure: every handler except
        # h1-rejection is a no-op for this method (main() [SKIP]s them).
        # h1 swaps the sampler for exact rejection sampling; DOE and every
        # infill batch RandomGA draws are then rejection-resampled.
        if handler == 'h1-rejection':
            g_fn    = make_benchmark_g_fn(benchmark, constr_index, threshold)
            sampler = RejectionSampling(sampler, g_fn)
        return RandomGA(pop_size=pop_size, sampling=sampler, eliminate_duplicates=elim), True

    rng = np.random.RandomState(seed)

    if method == 'samos':
        predict_pos = list(range(len(obj_indices)))
        real_pos    = []
        surrogates       = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_pos]
        constr_surrogate = XGBoost(100, seed=rng.randint(0, 2**31 - 1))

    elif method == 'samos-cheap':
        cheap_cols = set(BENCHMARK_META[suite][pid].get('cheap_obj_indices', []))
        predict_pos, real_pos = _obj_split(obj_indices, cheap_cols)
        surrogates = [XGBoost(100, seed=rng.randint(0, 2**31 - 1)) for _ in predict_pos]
        exact_constr = constr_index in cheap_cols   # H7: cheap constraint -> exact
        constr_surrogate = None if exact_constr else XGBoost(100, seed=rng.randint(0, 2**31 - 1))

    else:
        raise ValueError(f'Unknown method: {method!r}')

    # ── handler wiring (scenario-independent) ─────────────────────────────────
    # wrap_inner: callable(inner) applied inside the factory each outer
    # generation, so only the inner surrogate problem is ever wrapped -- the
    # outer archive / saved pkls always keep unpenalized F and raw, un-relaxed
    # G. algo_ref is a late-bound handle to the live SAMOS2 instance for the
    # stateful handlers (h3/h5).
    samos2_kwargs  = {}
    wrap_inner     = None
    algo_ref       = []
    copy_algorithm = True

    if handler == 'h4-cdp':
        pass   # native CDP: out['G'] + RankAndCrowding do everything.

    elif handler == 'h2-penalty':
        def wrap_inner(inner):
            # Static penalty on the inner problem only. _ConstraintsAsPenaltyMO,
            # not the plain pymoo class -- see its docstring (multi-objective
            # reshape bug in pymoo 0.6.1.1).
            return _ConstraintsAsPenaltyMO(inner, penalty=penalty)

    elif handler == 'h3-adaptive-penalty':
        pen = AdaptivePenaltyProblem(w0=1.0, target=0.5, c=1.2)

        def wrap_inner(inner):
            arc = algo_ref[0]._archive    # live archive (copy_algorithm=False)
            if len(arc) > 0:
                pen.adapt(feasible_fraction(arc))
            return pen.wrap(inner)

        copy_algorithm = False            # live-archive handle, see docstring

    elif handler == 'h5-eps':
        eps = EpsilonRelaxation(n_gen_total=n_gen)

        def wrap_inner(inner):
            eps.maybe_init_eps0(algo_ref[0]._archive)   # eps0 = mean DOE CV
            wrapped = eps.wrap(inner)                   # snapshots epsilon(t)
            eps.advance()                               # t+1 for next generation
            return wrapped

        copy_algorithm = False            # live-archive handle, see docstring

    elif handler == 'h6-sr':
        samos2_kwargs['inner_algorithm'] = partial(
            NSGA2, survival=DominanceStochasticRanking(Pf=0.45, seed=seed))

    elif handler == 'h1-rejection':
        # Rejection on the C3 infill seam (inner candidates already carry the
        # inner problem's G -- predicted for samos, exact for cheap
        # constraints), PLUS pure-rejection semantics: the inner
        # NSGA-II survival must NOT be feasibility-first. pymoo 0.6.1.1's
        # RankAndCrowding constructor hardcodes filter_infeasible=True (no
        # kwarg), but it is a plain instance attribute set by
        # Survival.__init__, so it is flipped post-construction; the single
        # instance is safely reused across outer generations (stateless).
        samos2_kwargs['infill_selector'] = RejectionInfillSelector()
        _surv = RankAndCrowding()
        _surv.filter_infeasible = False
        samos2_kwargs['inner_algorithm'] = partial(NSGA2, survival=_surv)

    else:
        raise ValueError(f'Unknown handler: {handler!r}')

    def factory(surrs, fitted_constr_surrogate=None):
        inner = ConstrainedSurrogateProblemEvox(
            surrs, obj_indices, predict_pos, real_pos, benchmark,
            constr_index, threshold, constr_surrogate=fitted_constr_surrogate,
        )
        return wrap_inner(inner) if wrap_inner is not None else inner

    algorithm = SAMOS2(
        sampling=sampler, surrogates=surrogates, surrogate_problem_factory=factory,
        predict_obj_indices=predict_pos,
        crossover=crossover, mutation=mutation, n_doe=n_doe_, n_infill=n_infill_,
        n_gen_inner=n_gen_inner, ga_pop_size=inner_ps, use_subset_selection=True,
        constr_surrogate=constr_surrogate,
        **samos2_kwargs,
    )
    algo_ref.append(algorithm)   # late-bind the live handle for h3/h5 closures
    return algorithm, copy_algorithm


def run_single(method, scenario, suite, pid, handler, seed, pop_size, n_gen,
               n_doe, n_infill, n_gen_inner, inner_pop_size, penalty,
               compute_indicators=True):
    np.random.seed(seed)
    random.seed(seed)

    cfg          = SCENARIOS[scenario]
    benchmark    = get_benchmark(suite, pid)
    constr_index = metric_index(suite, pid, cfg['constr_metric'])
    threshold    = THRESHOLDS[(suite, pid)][cfg['constr_metric']]

    problem  = ConstrainedEvoXBenchProblem(benchmark, OBJ_INDICES, constr_index, threshold)
    callback = FeasibilityAwareEvoxBenchCallback(
        benchmark, OBJ_INDICES, constr_index, threshold,
        compute_indicators=compute_indicators)

    algorithm, copy_algorithm = build_algorithm(
        method, benchmark, suite, pid, OBJ_INDICES, constr_index, threshold,
        handler, seed, pop_size, n_doe, n_infill, n_gen_inner, inner_pop_size,
        penalty, n_gen)

    results = minimize(
        problem=problem, algorithm=algorithm, termination=('n_gen', n_gen),
        seed=seed, callback=callback, save_history=False, verbose=True,
        copy_algorithm=copy_algorithm,
    )
    return results.algorithm.callback.data


def migrate(args):
    """Copy round-1 FLAT pkls ({...}/{method}/seed_N.pkl) into their
    scenario-default handler subdir ({...}/{method}/{h4-cdp|h2-penalty}/
    seed_N.pkl). Copies only -- originals are NEVER moved or deleted
    (CLAUDE.md / plan Round 2). Idempotent: existing destinations are left
    untouched. Works on whatever instances/budgets exist under
    --results_root."""
    n_copied = n_skipped = 0
    for scenario, cfg in SCENARIOS.items():
        default_handler = DEFAULT_HANDLER[cfg['mode']]
        # Fixed-depth flat layout: scenario/suite/pidX/budget/method/seed_N.pkl.
        # Handler-subdir pkls live one level deeper and cannot match this glob.
        pattern = os.path.join(args.results_root, scenario, '*', 'pid*', '*', '*', 'seed_*.pkl')
        for src in sorted(glob.glob(pattern)):
            method = os.path.basename(os.path.dirname(src))
            if method not in METHODS:
                continue   # not a flat method dir (defensive)
            dst_dir = os.path.join(os.path.dirname(src), default_handler)
            dst = os.path.join(dst_dir, os.path.basename(src))
            if os.path.exists(dst):
                n_skipped += 1
                print(f'[MIGRATE-SKIP] exists: {dst}')
                continue
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst)
            n_copied += 1
            print(f'[MIGRATE] {src}  ->  {dst}')
    print(f'[MIGRATE] done: {n_copied} copied, {n_skipped} already present '
          f'(originals left in place).')
    return 0


def _resolve_handlers(scenario, args):
    """Handler list for this invocation: explicit --handler > --all_handlers
    (the scenario's full SCENARIO_HANDLERS row) > the scenario default."""
    default = DEFAULT_HANDLER[SCENARIOS[scenario]['mode']]
    if args.handler is not None:
        return [args.handler], default
    if args.all_handlers:
        return list(SCENARIO_HANDLERS[scenario]), default
    return [default], default


def main(args):
    budget_folder = f'B{args.n_gen * args.pop_size}_P{args.pop_size}'
    scenario = args.scenario
    suite, pid = args.suite, args.pid
    cfg = SCENARIOS[scenario]
    if (suite, pid) not in THRESHOLDS:
        print(f'ERROR: no thresholds defined for {suite}/pid{pid} '
              f'(available: {sorted(THRESHOLDS)}) -- see THRESHOLDS.md.')
        return 1
    threshold = THRESHOLDS[(suite, pid)][cfg['constr_metric']]

    handlers, default_handler = _resolve_handlers(scenario, args)

    # (method, handler) work list; RandomGA runs only its default + h1.
    pairs = []
    for method in args.method:
        for handler in handlers:
            if method == 'random' and handler != default_handler \
                    and handler not in RANDOM_VALID_EXTRA:
                print(f'[SKIP] random x {handler}: handler acts on selection '
                      f'pressure RandomGA does not have -- would be byte-identical '
                      f'to random x {default_handler} (analysis replicates the row).')
                continue
            pairs.append((method, handler))

    total_runs = len(args.seeds) * len(pairs)
    run_i = 0
    summary: dict = {}   # (method, handler) -> list of (final_hv, n_feasible, n_total)

    for seed in args.seeds:
        for method, handler in pairs:
            run_i += 1
            save_dir = os.path.join(
                args.results_root, scenario, suite, f'pid{pid}', budget_folder,
                method, handler)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f'seed_{seed}.pkl')

            if os.path.exists(out_path) and not args.overwrite:
                print(f'[SKIP {run_i}/{total_runs}] {scenario}/{suite}/pid{pid}/'
                      f'{method}/{handler}/seed_{seed} already exists')
                with open(out_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f'\n[RUN {run_i}/{total_runs}] scenario={scenario} {suite}/pid{pid} '
                      f'(mode={cfg["mode"]}, constr={cfg["constr_metric"]}, T={threshold:.4f})  '
                      f'method={method}  handler={handler}  seed={seed}  '
                      f'pop={args.pop_size}  n_gen={args.n_gen}')
                data = run_single(
                    method, scenario, suite, pid, handler, seed, args.pop_size,
                    args.n_gen, args.n_doe, args.n_infill, args.n_gen_inner,
                    args.inner_pop_size, args.penalty)
                with open(out_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f'  Saved -> {out_path}')

            final_ind    = data['indicators'][-1] if data['indicators'] else {'hv': float('nan'), 'igd_plus': float('nan')}
            final_nfeas  = data['n_feasible'][-1] if data.get('n_feasible') else 0
            final_ntotal = data['n_total'][-1] if data.get('n_total') else 0
            summary.setdefault((method, handler), []).append(
                (final_ind['hv'], final_nfeas, final_ntotal))

    print('\n' + '=' * 88)
    print(f'Scenario {scenario} on {suite}/pid{pid} (mode={cfg["mode"]}, '
          f'constr={cfg["constr_metric"]}, T={threshold:.4f}) -- final feasible-HV summary '
          f'(mean +/- std over up to {len(args.seeds)} seeds)')
    for method, handler in pairs:
        rows = summary.get((method, handler))
        if not rows:
            continue
        hv_arr     = np.array([r[0] for r in rows], dtype=float)
        nfeas_arr  = np.array([r[1] for r in rows], dtype=float)
        ntotal_arr = np.array([r[2] for r in rows], dtype=float)
        print(f'  {method:>12s} x {handler:<19s} : hv={np.nanmean(hv_arr):.4f} +/- {np.nanstd(hv_arr):.4f}  '
              f'n_feasible={np.nanmean(nfeas_arr):.1f}/{np.nanmean(ntotal_arr):.1f}  (n={len(rows)})')
    print('=' * 88)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', default=None, choices=list(SCENARIOS),
                    help='Which of s1-s4 to run (see module docstring). '
                         'Required unless --migrate.')
    p.add_argument('--suite', default=SUITE, choices=['c10mop', 'in1kmop'],
                    help='Benchmark suite (instance must have an entry in THRESHOLDS).')
    p.add_argument('--pid', type=int, default=PID,
                    help='Benchmark problem id within --suite.')
    p.add_argument('--method', nargs='+', default=list(METHODS), choices=METHODS)
    p.add_argument('--handler', default=None, choices=HANDLERS,
                    help='Constraint handler (round 2, see module docstring). '
                         'Default: the scenario default (hard -> h4-cdp, '
                         'soft -> h2-penalty) = round-1 behavior.')
    p.add_argument('--all_handlers', action='store_true',
                    help='Run the scenario\'s full handler row '
                         '(SCENARIO_HANDLERS) instead of a single handler. '
                         'Ignored when --handler is given.')
    p.add_argument('--migrate', action='store_true',
                    help='No runs: copy round-1 flat {method}/seed_N.pkl files '
                         'under --results_root into their scenario-default '
                         'handler subdir (s1/s2 -> h4-cdp, s3/s4 -> h2-penalty). '
                         'Copies only; originals stay in place.')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--pop_size', type=int, default=20)
    p.add_argument('--n_gen', type=int, default=60)
    p.add_argument('--n_doe', type=int, default=None,
                    help='SAMOS2: initial DOE size (default: pop_size)')
    p.add_argument('--n_infill', type=int, default=None,
                    help='SAMOS2: real evaluations per outer generation (default: pop_size)')
    p.add_argument('--n_gen_inner', type=int, default=20,
                    help='SAMOS2: inner NSGA-II generations')
    p.add_argument('--inner_pop_size', type=int, default=None,
                    help='SAMOS2: inner NSGA-II population size (default: pop_size x 10)')
    p.add_argument('--results_root', default=os.path.join('results', 'constraint'),
                    help='Output root for per-seed pkls. Point at a dedicated '
                         'smoke-test folder when testing -- never write test '
                         'data into results/ (see CLAUDE.md).')
    p.add_argument('--penalty', type=float, default=1.0,
                    help='ConstraintsAsPenalty weight for soft scenarios (s3/s4). '
                         'Unused for hard scenarios (s1/s2).')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.migrate:
        sys.exit(migrate(args))
    if args.scenario is None:
        p.error('--scenario is required (unless --migrate is given).')
    sys.exit(main(args))
