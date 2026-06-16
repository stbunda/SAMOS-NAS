"""Shared utilities for building pymoo benchmark problems and reference data."""

import numpy as np
from pymoo.problems import get_problem


def build_problem(problem_name: str, n_obj: int, n_var: int = None):
    name = problem_name.lower()
    kwargs = {}
    if n_var is not None:
        kwargs['n_var'] = n_var
    elif name.startswith('wfg'):
        kwargs['n_var'] = 2 * (n_obj - 1) + 10
    if not name.startswith('zdt'):
        kwargs['n_obj'] = n_obj
    return get_problem(problem_name, **kwargs)


def get_pareto_front(problem, n_obj: int, min_pts: int = 300) -> np.ndarray:
    """Return a reference Pareto front from a pymoo problem."""
    try:
        pf = problem.pareto_front(n_points=1000)
        if pf is not None and len(pf) >= min_pts:
            return pf
    except Exception:
        pass

    from pymoo.util.ref_dirs import get_reference_directions
    _p_for_500 = {2: 499, 3: 30, 4: 14}
    n_partitions = _p_for_500.get(n_obj, 10)
    ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=n_partitions)
    try:
        pf = problem.pareto_front(ref_dirs)
        if pf is not None and len(pf) > 0:
            return pf
    except Exception:
        pass

    pf = problem.pareto_front()
    if pf is None:
        raise RuntimeError(f'Cannot obtain Pareto front for {problem}.')
    return pf


def default_ref_point(problem_name: str, n_obj: int) -> np.ndarray:
    """Heuristic reference point for HV (slightly dominates the full Pareto front)."""
    name = problem_name.lower()
    if name.startswith('dtlz1'):
        return np.full(n_obj, 0.6)
    if name.startswith('wfg'):
        return np.array([2.0 * (i + 1) * 1.1 for i in range(n_obj)])
    return np.full(n_obj, 1.1)
