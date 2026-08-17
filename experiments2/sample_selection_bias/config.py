"""Grid for the sample-selection-bias experiment (NB201, c10mop/pid 7).

The premise: under a HARD model-size budget an architecture that exceeds it
cannot be run at all, so its error and every hardware-dependent metric are
unobservable. SAMOS discards those points, and its objective surrogates
therefore train on the feasible population only -- a textbook sample-selection
bias. This grid measures the impact along three axes.

  tightness : four #Params budgets (~5 / 14 / 25 / 51% of NB201 evaluable).
              This is ALWAYS the evaluability gate, in every role.
  role      : where #Params sits in the formulation the optimizer sees --
              an objective, the constraint, or one of two constraints.
  hw        : which hardware metric plays the other budget, each held at its
              own ~10% operating point.
  arm       : 'feasible' (realistic: surrogates see gated points not at all)
              vs 'oracle' (counterfactual: surrogates additionally train on
              the gated points' true objectives). Selection is gated in both;
              the ONLY difference is the surrogate training set, so the
              feasible-oracle gap IS the bias impact.

Every cell is 2-objective and uses CDP (native out['G'] + RankAndCrowding).

Thresholds sit strictly BETWEEN attained values rather than on a percentile:
NB201's #Params has 59 distinct values with huge ties (2 421 architectures
share one value), so a percentile threshold lands on a plateau -- the 5th and
10th percentiles are the SAME number, and a tie at the boundary makes
feasibility a float-equality question. Midpoint thresholds make the realised
feasible fraction exact and tie-free. Regenerate with derive_taus.py --check.
"""

from problem.evoxbench.benchmark_meta import metric_index as _metric_index

SPACE, SUITE, PID = 'NB201', 'c10mop', 7

SIZE_METRIC = '#Params'
ERR_METRIC = 'Err.'

# HV reference point, per objective. NB201 returns normalized objectives in
# [0, 1], so 1.05 dominates the nadir for every pair used here (including the
# flipped 1 - AI column). Same value the scenario_run campaign uses on NB201.
REF_POINT = (1.05, 1.05)

# #Params evaluability budgets: tightest to mildest.
TIGHTNESS = {
    'T1': dict(tau=0.013997760601341724, feasible_fraction=0.046656),
    'T2': dict(tau=0.0513251218944788,   feasible_fraction=0.141504),
    'T3': dict(tau=0.10731616243720055,  feasible_fraction=0.252608),
    'T4': dict(tau=0.2943262457847595,   feasible_fraction=0.511296),
}

# Hardware budgets, each at its own ~10% operating point.
#   sense : +1 ceiling (metric <= tau), -1 floor (metric >= tau).
#   partner : second objective for the 'multi' role, which puts BOTH #Params
#       and this metric in G and so needs an objective that is neither. The
#       partner is always the other device's metric -- a same-device energy
#       partner would be a relabelling (rho(latency, energy) = 0.99 within a
#       device), and a cross-device pair reads as a two-target deployment.
HW = {
    'edgegpu_latency': dict(
        metric='EdgeGPU Lat.', tau=0.3297163099050522, sense=1,
        feasible_fraction=0.099968, partner='eyeriss_latency'),
    'eyeriss_latency': dict(
        metric='Eyeriss Lat.', tau=0.180555559694767, sense=1,
        feasible_fraction=0.071232, partner='edgegpu_latency'),
    'edgegpu_energy': dict(
        metric='EdgeGPU En.', tau=0.3443908393383026, sense=1,
        feasible_fraction=0.099968, partner='eyeriss_latency'),
    # The suite's only FLOOR budget (arithmetic intensity is maximize-better,
    # a utilisation floor). As the 'constr' role's objective it is therefore
    # flipped to 1 - AI so every objective column is minimized.
    'eyeriss_arithmetic_intensity': dict(
        metric='Eyeriss AI', tau=0.9015783071517944, sense=-1,
        feasible_fraction=0.093824, partner='edgegpu_latency'),
}

# Where #Params sits in the formulation. The gate is #Params in all three --
# only the optimizer's view of it changes.
ROLES = ('obj', 'constr', 'multi')

# Surrogate training set. 'feasible' is what SAMOS does today.
ARMS = ('feasible', 'oracle')

# NB201 spearman(#Params, hw), for reading the results: how much of the size
# budget a hardware constraint can stand in for. From the stage-1 enumeration.
RHO_WITH_SIZE = {'edgegpu_latency': 0.490, 'eyeriss_latency': 0.542,
                 'edgegpu_energy': 0.549, 'eyeriss_arithmetic_intensity': 0.779}


def cell(tightness, role, hw):
    """Full specification of one (tightness, role, hw) cell.

    Returns objectives / declared constraints / the always-#Params gate, all
    as benchmark metric indices in the evaluate space, plus the objective
    positions needing a maximize->minimize flip.
    """
    t, h = TIGHTNESS[tightness], HW[hw]
    size_tau = t['tau']

    if role == 'obj':
        obj = (ERR_METRIC, SIZE_METRIC)
        constr = ((h['metric'], h['tau'], h['sense']),)
    elif role == 'constr':
        obj = (ERR_METRIC, h['metric'])
        constr = ((SIZE_METRIC, size_tau, 1),)
    elif role == 'multi':
        obj = (ERR_METRIC, HW[h['partner']]['metric'])
        constr = ((SIZE_METRIC, size_tau, 1), (h['metric'], h['tau'], h['sense']))
    else:
        raise ValueError(f'Unknown role: {role!r}')

    obj_idx = [_metric_index(SUITE, PID, m) for m in obj]
    # A maximize-better metric used as an OBJECTIVE is minimized as 1 - v.
    # Only ever the AI metric, and only in the 'constr' role.
    flip = [i for i, m in enumerate(obj)
            if m == HW[hw]['metric'] and h['sense'] < 0]

    # Which declared constraints stay MEASURABLE for a gated-out architecture:
    # #Params is a static structural count, so it is known even for a model
    # that cannot be run, while a hardware metric of an unrunnable model is
    # not. Only the former may train its constraint surrogate on the rejection
    # log -- otherwise the 'feasible' arm quietly gets oracle information.
    observable = [i for i, c in enumerate(constr) if c[0] == SIZE_METRIC]

    return dict(
        tightness=tightness, role=role, hw=hw,
        obj_metrics=obj, obj_indices=obj_idx, flip_obj_pos=flip,
        observable_constr=observable,
        constr_metrics=tuple(c[0] for c in constr),
        constr_indices=[_metric_index(SUITE, PID, c[0]) for c in constr],
        thresholds=[c[1] for c in constr],
        senses=[c[2] for c in constr],
        gate_metric=SIZE_METRIC,
        gate_index=_metric_index(SUITE, PID, SIZE_METRIC),
        gate_tau=size_tau,
        size_feasible_fraction=t['feasible_fraction'],
        ref_point=REF_POINT,
    )


def scoring_constraints(spec):
    """(indices, thresholds, senses) the callback scores feasibility on: the
    declared constraints PLUS the #Params gate. In the 'obj' role #Params is
    not a declared constraint, but an architecture that cannot be run is not a
    solution -- HV must be scored over the same feasible set in all roles."""
    idx = list(spec['constr_indices'])
    thr = list(spec['thresholds'])
    sns = list(spec['senses'])
    if spec['gate_index'] not in idx:
        idx.append(spec['gate_index'])
        thr.append(spec['gate_tau'])
        sns.append(1)
    return idx, thr, sns


# (role, hw) combinations with an EMPTY feasible set, excluded from the grid.
#
# Arithmetic intensity is antagonistic to a small-model budget:
# spearman(#Params, AI) = 0.78, and among the architectures under the T1-T3
# size budgets the LARGEST attainable AI is 0.633 -- below any floor worth
# calling a budget (the suite's 10% floor is 0.902, and even a floor of 0.60,
# which 37% of the whole space clears, leaves 64 architectures). So whenever
# AI is a CONSTRAINT alongside the size gate -- the 'obj' and 'multi' roles --
# the joint feasible set is exactly empty and HV is identically 0 at three of
# the four tightness levels. Nothing is measurable there, so the cells are not
# run. AI is kept in the 'constr' role, where it is the OBJECTIVE and only
# #Params constrains: that cell is not just viable but the most informative
# one in the grid, since it is the only place the gate removes precisely the
# good end of an objective.
EXCLUDED = {(role, 'eyeriss_arithmetic_intensity') for role in ('obj', 'multi')}


def excluded(role, hw):
    return (role, hw) in EXCLUDED


def cells():
    """Every runnable (tightness, role, hw, arm) cell, in a stable order."""
    for t in TIGHTNESS:
        for role in ROLES:
            for hw in HW:
                if excluded(role, hw):
                    continue
                for arm in ARMS:
                    yield t, role, hw, arm
