"""ParEGO runner for the pymoo benchmark (SMAC3 HPOFacade + ParEGO scalarisation).

Exposes a single ``run_parego`` function with the same return contract as the
other standalone runners (mosmac, cobra): a dict with keys
``indicators``, ``obj_pop``, ``var_pop``.
"""

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from problem.pymoo.benchmark_utils import build_problem, get_pareto_front, default_ref_point


def run_parego(
    problem_name: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    n_obj: int = 2,
    n_var: int = None,
) -> dict:
    """Run SMAC3 HPOFacade + ParEGO on a continuous pymoo benchmark.

    Drop-in replacement for ``run_mosmac``: identical Scenario/ConfigSpace and
    run-history extraction; differs only in the scalarisation strategy.

    Returns
    -------
    dict with keys ``indicators``, ``obj_pop``, ``var_pop``.
    """
    from ConfigSpace import ConfigurationSpace, Float as CSFloat
    from smac import HyperparameterOptimizationFacade as HPOFacade, Scenario
    from smac.multi_objective.parego import ParEGO

    prob      = build_problem(problem_name, n_obj, n_var)
    pf        = get_pareto_front(prob, prob.n_obj)
    ref_point = default_ref_point(problem_name, prob.n_obj)
    hv_ind    = HV(ref_point=ref_point)
    igd_ind   = IGDPlus(pf)

    n_trials  = pop_size * n_gen
    obj_names = [f'obj{i}' for i in range(prob.n_obj)]

    cs = ConfigurationSpace(seed=seed)
    cs.add([
        CSFloat(f'x{i}', (float(prob.xl[i]), float(prob.xu[i])))
        for i in range(prob.n_var)
    ])

    def target_fn(config, seed=0):
        x = np.array([config[f'x{i}'] for i in range(prob.n_var)])
        out = {}
        prob._evaluate(x.reshape(1, -1), out)
        F = out['F'][0]
        return {name: float(F[j]) for j, name in enumerate(obj_names)}

    scenario = Scenario(
        configspace=cs,
        objectives=obj_names,
        n_trials=n_trials,
        seed=seed,
        deterministic=True,
        n_workers=1,
    )
    smac = HPOFacade(
        scenario=scenario,
        target_function=target_fn,
        multi_objective_algorithm=ParEGO(scenario),
        overwrite=True,
    )
    smac.optimize()

    rh = smac.runhistory
    _internal = getattr(rh, 'data', None) or getattr(rh, '_data', {})
    sorted_trials = sorted(_internal.items(), key=lambda kv: kv[1].starttime)
    _id2cfg = getattr(rh, 'ids_config', None) or getattr(rh, '_ids_config', {})
    all_X_arr = np.array([
        [_id2cfg[k.config_id][f'x{i}'] for i in range(prob.n_var)]
        for k, _ in sorted_trials
    ])
    all_F_arr = np.array([
        list(v.cost) if hasattr(v.cost, '__iter__') else [v.cost]
        for _, v in sorted_trials
    ])

    indicators, obj_pop, var_pop = [], [], []
    for step in range(n_gen):
        lo        = step * pop_size
        hi        = (step + 1) * pop_size
        F_so_far  = all_F_arr[:hi]
        nd_idx    = NonDominatedSorting().do(F_so_far, only_non_dominated_front=True)
        nd_F      = F_so_far[nd_idx]
        indicators.append({
            'hv':       float(hv_ind(nd_F)),
            'igd_plus': float(igd_ind(nd_F)),
        })
        obj_pop.append(all_F_arr[lo:hi].copy())
        var_pop.append(all_X_arr[lo:hi].copy())

    return {'indicators': indicators, 'obj_pop': obj_pop, 'var_pop': var_pop}
