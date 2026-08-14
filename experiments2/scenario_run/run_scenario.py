"""experiments2/scenario_run/run_scenario.py -- scenario_run campaign runner.

Runs one (scenario, mode, method, handler) cell of the S1-S6 campaign
(scenarios.py) for a set of seeds. Mirrors experiments/constraint2/
run_constraint.py's run_single/main shape and CLI conventions, with THREE
deliberate differences:

1. TERMINATION IS EVALUATION-BASED: ``('n_evals', N)``, never ``('n_gen', N)``.
   This is the single most important difference from run_constraint.py. Two
   of the seven handlers spend REAL evaluations beyond the nominal
   pop_size-per-generation budget: h1-rejection resamples infeasible draws
   (every probe, feasible or not, is evaluated and charged), and SAMOS2's
   hard-gate DOE retries batches until enough feasible points exist. Under
   n_gen termination those extra evaluations would be bought OFF-BUDGET,
   handing h1/hard-gate SAMOS2 more real benchmark queries than every other
   handler for the "same" number of generations -- silently unfair. Under
   n_evals termination every handler gets the identical evaluation budget,
   and the handlers that waste evaluations on infeasible probes simply
   complete FEWER generations. Both the budget (``n_evals``) and the
   REALISED generation count (``algorithm.n_gen`` after ``minimize``
   returns) are recorded in ``meta`` so this cost is visible in analysis.

2. Per-scenario ``ref_point`` and ``penalty`` (scenarios.py) are threaded
   into the callback and the algorithm respectively -- see ``run_single``.
   Leaving either at its hardcoded default (1.05 / 1.0) would silently wrong
   two things: the ref point is derived per scenario as max(1.05, p95) per
   objective (TAU.md, 'Reference point'), so a space whose normalized
   objectives run past 1.05 would score zero hypervolume over most of the
   archive; and a flat penalty=1.0
   varies h2-static_penalty's effective pressure ~30x across the suite
   (TAU.md, 'Penalty'), collapsing it into unconstrained search on one end
   and into h4-cdp on the other.

3. ``sense`` (scenarios.constr_sense) is threaded through explicitly since
   S6 is a floor constraint (feasible <=> metric >= tau), not S1-S5's
   ceiling.

Output layout
-------------
  {results_root}/{sid}/{mode}/{method}/{handler}/seed_{N}.pkl

Every pkl is self-describing via data['meta'] (see run_single); hard mode
additionally persists the rejection log (rejected_X / rejected_G) exactly as
run_constraint.py's run_single does.

Examples
--------
  python experiments2/scenario_run/run_scenario.py --scenario S1 --mode hard
  python experiments2/scenario_run/run_scenario.py --scenario S6 --mode soft --all_handlers
  python experiments2/scenario_run/run_scenario.py --scenario S4 --mode hard \\
      --method samos --handler h1-rejection --seeds 0 1 2
  python experiments2/scenario_run/run_scenario.py --scenario S1 --mode hard \\
      --all_handlers --method nsga2 --seeds 0 --pop_size 8 --n_evals 120 \\
      --results_root smoke_tests/scenario_run
"""

import argparse
import os
import pickle
import random
import sys

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
from pymoo.optimize import minimize

import algorithms as ALG
import scenarios as SC
from problem.evoxbench.callbacks import FeasibilityAwareEvoxBenchCallback
from problem.evoxbench.utils import get_benchmark

_DEFAULT_RESULTS_ROOT = os.path.join('results2', 'scenario_run')

DEFAULT_POP_SIZE = 20
DEFAULT_N_EVALS = 1200


def run_single(sid, mode, method, handler, seed, pop_size, n_evals,
               resample_cap=10, compute_indicators=True):
    """Run one (scenario, mode, method, handler, seed) cell and return the
    self-describing result dict (callback data + meta, and the hard-mode
    rejection log when applicable)."""
    np.random.seed(seed)
    random.seed(seed)

    scenario  = SC.SCENARIOS[sid]
    benchmark = get_benchmark(scenario['suite'], scenario['pid'])
    obj_idx   = list(SC.obj_indices(sid))
    constr_idx = SC.constr_index(sid)
    tau       = scenario['tau']
    sense     = SC.constr_sense(sid)
    ref_point = scenario['ref_point']
    penalty   = scenario['penalty']
    gated     = (mode == 'hard')

    problem = ALG.build_problem(sid, benchmark, mode, handler=handler)

    # ref_point reaches the HV indicator here; sense makes S6's floor
    # constraint feasible in the right direction.
    callback = FeasibilityAwareEvoxBenchCallback(
        benchmark, obj_idx, constr_idx, tau, sense=sense, ref_point=ref_point,
        compute_indicators=compute_indicators)

    # Planned outer-generation count -- feeds ONLY h5-epsilon's decay clock
    # (eps -> 0 at 50% of this plan). Actual termination below is
    # evaluation-based, so the REALISED generation count (recorded in meta
    # after minimize returns) can land well below this plan for handlers
    # that waste evaluations (h1-rejection, hard-gate SAMOS2 DOE retries).
    planned_n_gen = max(1, n_evals // pop_size)

    # penalty reaches h2-static_penalty's fixed weight and h3-adaptive's
    # starting w0 inside build() -- see algorithms.py.
    algorithm, copy_algorithm, handler_state = ALG.build(
        method, handler, sid, benchmark, seed, pop_size, mode=mode,
        n_gen=planned_n_gen, penalty=penalty, resample_cap=resample_cap)

    results = minimize(
        problem=problem, algorithm=algorithm, termination=('n_evals', n_evals),
        seed=seed, callback=callback, save_history=False, verbose=True,
        copy_algorithm=copy_algorithm,
    )
    data = results.algorithm.callback.data
    if handler_state:
        data['handler_state'] = handler_state

    algo = results.algorithm

    # Hard mode: persist the rejection log so discarded evaluations are
    # reconstructable post hoc (mirrors run_constraint.py's run_single; a
    # single constrained metric per scenario here, so the legacy 1-D G shape
    # always applies -- no multi-constraint (n, n_constr) branching needed).
    if gated:
        if getattr(algo, '_rejected_X', None) is not None:
            data['rejected_X'] = np.asarray(algo._rejected_X)
            data['rejected_G'] = np.asarray(algo._rejected_G)
        elif getattr(algo, '_rejected_pop', None) is not None and len(algo._rejected_pop) > 0:
            X_rej = np.asarray(algo._rejected_pop.get('X'))
            data['rejected_X'] = X_rej
            data['rejected_G'] = np.full(len(X_rej), np.nan)
        else:
            data['rejected_X'] = np.empty((0, problem.n_var))
            data['rejected_G'] = np.empty(0)

    # Self-describing pkl: everything needed to re-derive this run's config
    # without consulting the output path or scenarios.py at read time.
    data['meta'] = dict(
        sid=sid, space=scenario['space'], suite=scenario['suite'], pid=scenario['pid'],
        obj_metrics=scenario['obj_metrics'], constr_metric=scenario['constr_metric'],
        tau=tau, direction=scenario['direction'], sense=sense,
        feasible_fraction=scenario['feasible_fraction'], ref_point=ref_point,
        penalty=penalty, mode=mode, gate=gated, method=method, handler=handler,
        seed=seed, pop_size=pop_size, n_evals=n_evals,
        n_gen=int(algo.n_gen or 0), n_eval_realised=int(algo.evaluator.n_eval),
        resample_cap=resample_cap, config='scenario_run',
    )
    if handler == 'h1-rejection':
        # nsga2's H1RejectionNSGA2 tracks this live; samos/random's delegated
        # h1-rejection path (run_constraint.build_algorithm) does not, so it
        # is None there.
        data['meta']['h1_rejected_count'] = handler_state.get('h1_rejected_count')

    return data


def _resolve_handlers(mode, args):
    """Handler list for this invocation: explicit --handler > --all_handlers
    (every scenarios.HANDLERS entry) > the scenario/mode default. Applies to
    the full-row methods only -- random/ctaea take their slot from
    scenarios.fixed_handler (see main)."""
    if args.handler is not None:
        return [args.handler]
    if args.all_handlers:
        return list(SC.HANDLERS)
    return [SC.DEFAULT_HANDLER[mode]]


def main(args):
    sid, mode = args.scenario, args.mode
    scenario = SC.SCENARIOS[sid]
    handlers = _resolve_handlers(mode, args)

    # (method, handler) work list. random and ctaea run ONE handler slot each
    # (scenarios.fixed_handler): random has no selection pressure or sampling
    # strategy for a handler to act on, and ctaea's handler is intrinsic to the
    # algorithm. Their slot is used DIRECTLY rather than filtered out of
    # ``handlers``, so e.g. `--method ctaea --mode soft` still runs its h4-cdp
    # slot instead of silently producing nothing when the mode default differs.
    pairs = []
    for method in args.method:
        fixed = SC.fixed_handler(method, mode)
        if fixed is not None:
            if args.handler is not None and args.handler != fixed:
                print(f'[SKIP] {method} x {args.handler}: {method} runs only its own '
                      f'handler slot ({fixed!r} for mode={mode!r}).')
            else:
                pairs.append((method, fixed))
            continue
        for handler in handlers:
            pairs.append((method, handler))

    total_runs = len(args.seeds) * len(pairs)
    run_i = 0
    summary: dict = {}   # (method, handler) -> list of (final_hv, n_feasible, n_total)

    for seed in args.seeds:
        for method, handler in pairs:
            run_i += 1
            save_dir = os.path.join(args.results_root, sid, mode, method, handler)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f'seed_{seed}.pkl')

            if os.path.exists(out_path) and not args.overwrite:
                print(f'[SKIP {run_i}/{total_runs}] {sid}/{mode}/{method}/{handler}/'
                      f'seed_{seed} already exists')
                with open(out_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f'\n[RUN {run_i}/{total_runs}] scenario={sid} '
                      f'({scenario["space"]} {scenario["suite"]}/pid{scenario["pid"]}, '
                      f'mode={mode}) method={method} handler={handler} seed={seed} '
                      f'pop={args.pop_size} n_evals={args.n_evals}')
                data = run_single(sid, mode, method, handler, seed,
                                   args.pop_size, args.n_evals)
                with open(out_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f'  Saved -> {out_path}  '
                      f'(n_gen={data["meta"]["n_gen"]}, '
                      f'n_eval={data["meta"]["n_eval_realised"]})')

            final_ind    = data['indicators'][-1] if data['indicators'] else {'hv': float('nan'), 'igd_plus': float('nan')}
            final_nfeas  = data['n_feasible'][-1] if data.get('n_feasible') else 0
            final_ntotal = data['n_total'][-1] if data.get('n_total') else 0
            summary.setdefault((method, handler), []).append(
                (final_ind['hv'], final_nfeas, final_ntotal))

    print('\n' + '=' * 88)
    print(f'Scenario {sid} ({scenario["space"]} {scenario["suite"]}/pid{scenario["pid"]}, '
          f'mode={mode}) -- final feasible-HV summary '
          f'(mean +/- std over up to {len(args.seeds)} seeds)')
    for method, handler in pairs:
        rows = summary.get((method, handler))
        if not rows:
            continue
        hv_arr     = np.array([r[0] for r in rows], dtype=float)
        nfeas_arr  = np.array([r[1] for r in rows], dtype=float)
        ntotal_arr = np.array([r[2] for r in rows], dtype=float)
        print(f'  {method:>8s} x {handler:<19s} : hv={np.nanmean(hv_arr):.4f} +/- {np.nanstd(hv_arr):.4f}  '
              f'n_feasible={np.nanmean(nfeas_arr):.1f}/{np.nanmean(ntotal_arr):.1f}  (n={len(rows)})')
    print('=' * 88)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', required=True, choices=list(SC.SCENARIOS),
                    help='Which of S1-S6 to run (see scenarios.py).')
    p.add_argument('--mode', required=True, choices=list(SC.MODES),
                    help='hard (evaluability gate) or soft (archive infeasible '
                         'with real F/G) -- tau is shared across both.')
    p.add_argument('--method', nargs='+', default=list(SC.METHODS), choices=SC.METHODS)
    p.add_argument('--handler', default=None, choices=SC.HANDLERS,
                    help='Constraint handler. Default: the scenario/mode default '
                         '(scenarios.DEFAULT_HANDLER).')
    p.add_argument('--all_handlers', action='store_true',
                    help='Run every handler in scenarios.HANDLERS. Ignored when '
                         '--handler is given.')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE)
    p.add_argument('--n_evals', type=int, default=DEFAULT_N_EVALS,
                    help='Evaluation budget -- termination is (\'n_evals\', N), '
                         'NOT (\'n_gen\', N) (see module docstring).')
    p.add_argument('--results_root', default=_DEFAULT_RESULTS_ROOT,
                    help='Output root for per-seed pkls. Point at a dedicated '
                         'smoke-test folder when testing -- never write test '
                         'data into results2/ (see CLAUDE.md).')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    sys.exit(main(args))
