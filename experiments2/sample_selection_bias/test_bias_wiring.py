"""Self-check for the three seams this experiment added to shared code.

  python experiments2/sample_selection_bias/test_bias_wiring.py

Two of the three seams live in strategy/surrogate/samos2.py and
problem/evoxbench/constrained_problem.py, which the scenario_run campaign also
runs, so the first check is a REGRESSION check: with no gate metric given, the
rejection log must still be exactly what it was before the split into gate
violations and declared-constraint violations.
"""

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

from pymoo.optimize import minimize

import config as CF
import run_bias as RB
from _bias import build_probe, evaluate_metrics
from problem.evoxbench.utils import get_benchmark

POP, N_EVALS = 8, 48


def _hw(space, sense=1):
    """First hardware metric on ``space`` with the given direction, or None.
    Keeps the checks space-agnostic: NB201 has four (one a floor), the
    surrogate space has one."""
    return next((k for k, v in CF.hw(space).items() if v['sense'] == sense), None)


def _run(spec, arm, benchmark, seed=0, problem=None):
    """Run one short cell and return the algorithm THE RUN USED. pymoo's
    minimize deep-copies the algorithm by default, so the instance handed in
    stays pristine -- reading the rejection log off it finds nothing."""
    built_problem, algorithm = RB.build(spec, arm, benchmark, seed, POP)
    if problem is None:
        problem = built_problem
    else:
        algorithm.gate_g_fn = None    # legacy wiring: gate on the declared G
    res = minimize(problem=problem, algorithm=algorithm,
                   termination=('n_evals', N_EVALS), seed=seed, verbose=False)
    return built_problem, res.algorithm


def check_legacy_rejection_log(benchmark, space):
    """Regression: when the gate IS the declared constraint set (no
    gate_g_fn), the new declared-constraint log must equal the old gate log,
    so every scenario_run cell trains its constraint surrogate on exactly the
    same rows as before."""
    from problem.evoxbench.constrained_problem import ConstrainedEvoXBenchProblem
    spec = CF.cell('T2', 'constr', _hw(space), space)
    # The pre-change configuration: gate keyed to the declared constraints
    # (gate_index unset), no gate_g_fn.
    legacy = ConstrainedEvoXBenchProblem(
        benchmark, spec['obj_indices'], spec['constr_indices'],
        spec['thresholds'], sense=spec['senses'], gate=True)
    _, algorithm = _run(spec, 'feasible', benchmark, problem=legacy)

    assert algorithm._rejected_X is not None and len(algorithm._rejected_X) > 0, \
        'no rejections to compare -- widen the check'
    gate_log = np.asarray(algorithm._rejected_G, float).reshape(len(algorithm._rejected_X), -1)
    decl_log = np.asarray(algorithm._rejected_Gdecl, float)
    assert decl_log.shape == gate_log.shape, (decl_log.shape, gate_log.shape)
    assert np.allclose(decl_log, gate_log), 'legacy rejection log changed'
    assert algorithm.constr_observable_when_gated == {0}, \
        'the size constraint must stay observable in the constr role'
    print(f'  ok  legacy rejection log unchanged ({len(gate_log)} rejected rows)')


def check_gate_is_size_only(benchmark, space):
    """The gate must fire on #Params alone, never on the hardware constraint:
    in the 'obj' role #Params is not even a declared constraint, so a run that
    gated on G would gate on latency instead."""
    spec = CF.cell('T3', 'obj', _hw(space), space)
    problem, algorithm = _run(spec, 'feasible', benchmark)
    X = np.round(algorithm._rejected_X).astype(int)
    F = evaluate_metrics(benchmark, X)
    size = F[:, spec['gate_index']]
    assert (size > spec['gate_tau']).all(), \
        'a rejected architecture was inside the size budget -- gate read the wrong metric'
    arc = np.round(algorithm._archive.get('X')).astype(int)
    arc_size = evaluate_metrics(benchmark, arc)[:, spec['gate_index']]
    assert (arc_size <= spec['gate_tau']).all(), 'an over-budget architecture entered the archive'
    # The hardware constraint is unobservable for a gated architecture, so its
    # surrogate must NOT have been fed the rejection log.
    assert algorithm.constr_observable_when_gated == set(), \
        'the hardware constraint must be unobservable when gated in the obj role'
    print(f'  ok  gate keys on #Params only ({len(X)} rejected, {len(arc)} archived)')


def check_oracle_arm(benchmark, space):
    """The oracle arm must capture finite objective values for gated points;
    the feasible arm must capture none. Nothing else may differ."""
    spec = CF.cell('T2', 'constr', _hw(space), space)
    _, feas = _run(spec, 'feasible', benchmark)
    _, orac = _run(spec, 'oracle', benchmark)
    assert feas._rejected_F is None, 'feasible arm captured oracle objectives'
    assert orac._rejected_F is not None and len(orac._rejected_F) > 0, \
        'oracle arm captured no gated objectives'
    assert np.isfinite(orac._rejected_F).all(), \
        'oracle training set holds the inf mask, not the true objectives'
    assert len(orac._rejected_F) == len(orac._rejected_X)
    assert orac._rejected_F.shape[1] == len(spec['obj_indices'])
    print(f'  ok  oracle arm sees {len(orac._rejected_F)} gated architectures, '
          f'feasible arm sees 0')


def check_flip(benchmark, space, floor_hw):
    """A maximize-better objective must be minimized as 1 - v, consistently in
    the problem and in the probe set."""
    spec = CF.cell('T4', 'constr', floor_hw, space)
    assert spec['flip_obj_pos'] == [1], spec['flip_obj_pos']
    problem, _ = RB.build(spec, 'feasible', benchmark, 0, POP)
    X, F_obj, gated = build_probe(benchmark, spec, problem.xl, problem.xu,
                                     n=200, seed=7)
    raw = evaluate_metrics(benchmark, X)
    assert np.allclose(F_obj[:, 1], 1.0 - raw[:, spec['obj_indices'][1]]), \
        'probe objective was not flipped like the problem flips it'
    assert np.array_equal(gated, raw[:, spec['gate_index']] > spec['gate_tau'])
    print('  ok  maximize-better objective flipped consistently')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--space', nargs='+', default=list(CF.SPACES),
                    choices=list(CF.SPACES))
    args = ap.parse_args()
    for space in args.space:
        print(f'--- {space} ---')
        benchmark = get_benchmark(*CF.suite_pid(space))
        check_legacy_rejection_log(benchmark, space)
        check_gate_is_size_only(benchmark, space)
        check_oracle_arm(benchmark, space)
        floor_hw = _hw(space, sense=-1)
        if floor_hw is None:
            print('  --  no floor metric on this space, flip check not applicable')
        else:
            check_flip(benchmark, space, floor_hw)
    print('all checks passed')


if __name__ == '__main__':
    main()
