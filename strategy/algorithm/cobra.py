"""IOC-SAMO-COBRA runner for the pymoo benchmark.

Exposes a single ``run_cobra`` function with the same return contract as the
other standalone runners (mosmac, parego): a dict with keys
``indicators``, ``obj_pop``, ``var_pop``.
"""

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from problem.pymoo.benchmark_utils import build_problem, get_pareto_front, default_ref_point
from strategy.algorithm.ioc_samo_cobra import cheap_SAMO_COBRA_Init, cheap_SAMO_COBRA_PhaseII


def run_cobra(
    problem_name: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    n_obj: int = 2,
    n_var: int = None,
    n_doe: int = None,
    n_infill: int = None,
) -> dict:
    """Run IOC-SAMO-COBRA on a continuous pymoo benchmark.

    Total budget: ``pop_size × n_gen`` evaluations.
    Results are split into ``pop_size``-sized pseudo-generations for fair
    comparison with population-based methods.

    Returns
    -------
    dict with keys ``indicators``, ``obj_pop``, ``var_pop``.
    """
    prob      = build_problem(problem_name, n_obj, n_var)
    pf        = get_pareto_front(prob, prob.n_obj)
    ref_point = default_ref_point(problem_name, prob.n_obj)
    hv_ind    = HV(ref_point=ref_point)
    igd_ind   = IGDPlus(pf)

    n_trials = pop_size * n_gen
    batch    = n_infill if n_infill is not None else pop_size
    init_pts = n_doe    if n_doe    is not None else None   # None → COBRA default

    class _CobraAdapter:
        def __init__(self):
            self.lower        = prob.xl.astype(float)
            self.upper        = prob.xu.astype(float)
            self.nObj         = prob.n_obj
            # One always-satisfied dummy constraint prevents empty Gres arrays
            # (COBRA's init fails with zero-size Gres).  cheapConstr = [] means
            # cheap_evaluate provides the constraint (the dummy -1).
            self.nConstraints = 1
            self.cheapConstr  = []
            self.cheapObj     = [False] * prob.n_obj
            self.ref          = list(ref_point)

        def evaluate(self, x):
            out = {}
            prob._evaluate(np.array(x, dtype=float).reshape(1, -1), out)
            return [out['F'][0].astype(float), np.array([-1.0])]

        def cheap_evaluate(self, x):
            return [np.full(prob.n_obj, np.nan), np.array([-1.0])]

    cobra = cheap_SAMO_COBRA_Init(
        _CobraAdapter(),
        batch=batch,
        nCores=1,
        feval=n_trials,
        initDesPoints=init_pts,
        cobraSeed=seed,
        initDesign='LHS',
    )
    cobra = cheap_SAMO_COBRA_PhaseII(cobra)

    all_X = np.asarray(cobra['A'],    dtype=float)   # (n_evals, n_var)
    all_F = np.asarray(cobra['Fres'], dtype=float)   # (n_evals, n_obj)

    indicators, obj_pop, var_pop = [], [], []
    for step in range(n_gen):
        lo = step * pop_size
        hi = min((step + 1) * pop_size, len(all_F))
        if lo >= len(all_F):
            break
        F_so_far = all_F[:hi]
        nd_idx   = NonDominatedSorting().do(F_so_far, only_non_dominated_front=True)
        nd_F     = F_so_far[nd_idx]
        indicators.append({
            'hv':       float(hv_ind(nd_F)),
            'igd_plus': float(igd_ind(nd_F)),
        })
        obj_pop.append(all_F[lo:hi].copy())
        var_pop.append(all_X[lo:hi].copy())

    return {'indicators': indicators, 'obj_pop': obj_pop, 'var_pop': var_pop}
