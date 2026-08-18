"""Grid for the sample-selection-bias experiment.

The premise: under a HARD model-size budget an architecture that exceeds it
cannot be run at all, so its error and every hardware-dependent metric are
unobservable. SAMOS discards those points, and its objective surrogates
therefore train on the feasible population only -- a textbook sample-selection
bias. This grid measures the impact along four axes.

  space     : NB201 (tabular, exhaustively enumerable, exact reference front)
              or MobileNetV3 (surrogate benchmark, 21 variables, sampled
              reference). See SPACES.
  tightness : four #Params budgets. ALWAYS the evaluability gate, in every role.
  role      : where #Params sits in the formulation the optimizer sees --
              an objective, the constraint, or one of two constraints.
  hw        : which hardware metric plays the other budget, at its ~10% point.
  arm       : 'feasible' (realistic: surrogates never see gated points) vs
              'oracle' (counterfactual: surrogates additionally train on the
              gated points' true objectives). Selection is gated in both; the
              ONLY difference is the surrogate training set, so the
              feasible-oracle gap IS the bias impact.

Every cell is 2-objective and uses CDP (native out['G'] + RankAndCrowding).

Thresholds sit strictly BETWEEN attained values rather than on a percentile.
NB201's #Params has 59 distinct values with huge ties (2 421 architectures
share one value), so a percentile threshold lands on a plateau -- its 5th and
10th percentiles are the SAME number -- and a tie at the boundary makes
feasibility a float-equality question. Midpoint thresholds make the realised
feasible fraction exact and tie-free. Regenerate with derive_taus.py --check.

Only #Params and the hardware metric are budgets. FLOPs is never one: like
#Params it is a static structural count, observable even for a model that
cannot be run, so it cannot carry the evaluability premise. It appears only as
an ordinary objective.
"""

import os

from problem.evoxbench.benchmark_meta import metric_index as _metric_index

SIZE_METRIC = '#Params'
ERR_METRIC = 'Err.'

# Where #Params sits in the formulation. The gate is #Params in all three --
# only the optimizer's view of it changes.
ROLES = ('obj', 'constr', 'multi')

# Surrogate training set. 'feasible' is what SAMOS does today.
ARMS = ('feasible', 'oracle')

SPACES = {
    # ── tabular, 15 625 architectures, exhaustively enumerable ───────────────
    # Every reference number is exact. Caveat found in the results: 1 200
    # evaluations enumerate 7.7% of this space and cells reach 95% of the exact
    # optimum in 40-120 evaluations, so final HV saturates and cannot separate
    # the arms. MobileNetV3 below exists to remove that ceiling.
    'NB201': dict(
        suite='c10mop', pid=7, exact=True,
        results_root=os.path.join('results2', 'sample_selection_bias'),
        ref_point=(1.05, 1.05),
        tightness={
            'T1': dict(tau=0.013997760601341724, feasible_fraction=0.046656),
            'T2': dict(tau=0.0513251218944788,   feasible_fraction=0.141504),
            'T3': dict(tau=0.10731616243720055,  feasible_fraction=0.252608),
            'T4': dict(tau=0.2943262457847595,   feasible_fraction=0.511296),
        },
        hw={
            'edgegpu_latency': dict(
                metric='EdgeGPU Lat.', tau=0.3297163099050522, sense=1,
                feasible_fraction=0.099968, partner_metric='Eyeriss Lat.'),
            'eyeriss_latency': dict(
                metric='Eyeriss Lat.', tau=0.180555559694767, sense=1,
                feasible_fraction=0.071232, partner_metric='EdgeGPU Lat.'),
            'edgegpu_energy': dict(
                metric='EdgeGPU En.', tau=0.3443908393383026, sense=1,
                feasible_fraction=0.099968, partner_metric='Eyeriss Lat.'),
            # The suite's only FLOOR budget (arithmetic intensity is
            # maximize-better, a utilisation floor). As the 'constr' role's
            # objective it is flipped to 1 - AI so every column is minimized.
            'eyeriss_arithmetic_intensity': dict(
                metric='Eyeriss AI', tau=0.9015783071517944, sense=-1,
                feasible_fraction=0.093824, partner_metric='EdgeGPU Lat.'),
        },
        # Arithmetic intensity is antagonistic to a small-model budget:
        # spearman(#Params, AI) = 0.78, and among architectures under the
        # T1-T3 size budgets the LARGEST attainable AI is 0.633 -- below any
        # floor worth calling a budget (even a floor of 0.60, which 37% of the
        # space clears, leaves 64 architectures). So wherever AI is a
        # CONSTRAINT next to the size gate -- 'obj' and 'multi' -- the joint
        # feasible set is exactly empty and HV is identically 0. Nothing is
        # measurable there. AI is kept in 'constr', where it is the OBJECTIVE
        # and only #Params constrains: the one cell where the gate removes
        # precisely the good end of an objective.
        excluded={(role, 'eyeriss_arithmetic_intensity')
                  for role in ('obj', 'multi')},
        # NB201 spearman(#Params, hw): how much of the size budget a hardware
        # constraint can stand in for.
        rho_with_size={'edgegpu_latency': 0.490, 'eyeriss_latency': 0.542,
                       'edgegpu_energy': 0.549,
                       'eyeriss_arithmetic_intensity': 0.779},
    ),

    # ── surrogate benchmark: MLP accuracy predictor, 21 variables ────────────
    # Chosen to remove NB201's saturation ceiling. The space is ~1e20
    # architectures, so 1 200 evaluations cannot solve it; the reference front
    # is estimated from the WHOLE 1M-architecture stage-1 cache rather than
    # enumerated, and hv_frac is therefore APPROXIMATE (it can sit slightly
    # above 1).
    #
    # The only surrogate space in the suite carrying a real hardware metric.
    # Its geometry is close to the mirror image of NB201's: spearman(#Params,
    # Latency) = 0.16, so the size budget and the hardware budget are nearly
    # INDEPENDENT here, where on NB201 they ran together at 0.49-0.78.
    # Objectives are already normalized into evaluate space by the problem
    # (in1kmop is not normalized_objectives), where Err. runs negative -- the
    # ref point still dominates every column (max 1.11 across metrics).
    'MobileNetV3': dict(
        suite='in1kmop', pid=9, exact=False,
        results_root=os.path.join('results2', 'sample_selection_bias_mobilenetv3'),
        ref_point=(1.05, 1.05),
        tightness={
            'T1': dict(tau=0.35053444469578315, feasible_fraction=0.049999),
            'T2': dict(tau=0.42143462537330956, feasible_fraction=0.149998),
            'T3': dict(tau=0.4663391468928333,  feasible_fraction=0.249999),
            'T4': dict(tau=0.5528896950816198,  feasible_fraction=0.500000),
        },
        hw={
            # The multi role's partner objective must be neither #Params nor
            # the constrained hardware metric, and MobileNetV3 exposes only
            # four metrics, so it can only be FLOPs. Flag when reporting:
            # spearman(FLOPs, Latency) = 0.99 here, so in the 'multi' role the
            # second objective is close to a relabelling of the constrained
            # metric. Unavoidable on this space.
            'latency': dict(
                metric='Latency', tau=0.27869882805799917, sense=1,
                feasible_fraction=0.100000, partner_metric='FLOPs'),
        },
        excluded=set(),
        rho_with_size={'latency': 0.162},
    ),
}

DEFAULT_SPACE = 'NB201'


# ── accessors ────────────────────────────────────────────────────────────────

def space(name=DEFAULT_SPACE):
    return SPACES[name]


def suite_pid(name=DEFAULT_SPACE):
    s = SPACES[name]
    return s['suite'], s['pid']


def tightness(name=DEFAULT_SPACE):
    return SPACES[name]['tightness']


def hw(name=DEFAULT_SPACE):
    return SPACES[name]['hw']


def results_root(name=DEFAULT_SPACE):
    return SPACES[name]['results_root']


def is_exact(name=DEFAULT_SPACE):
    return SPACES[name]['exact']


def excluded(role, hw_name, name=DEFAULT_SPACE):
    return (role, hw_name) in SPACES[name]['excluded']


def cell(tight, role, hw_name, name=DEFAULT_SPACE):
    """Full specification of one (tightness, role, hw) cell on ``name``.

    Returns objectives / declared constraints / the always-#Params gate, all
    as benchmark metric indices in the evaluate space, plus the objective
    positions needing a maximize->minimize flip.
    """
    s = SPACES[name]
    suite, pid = s['suite'], s['pid']
    t, h = s['tightness'][tight], s['hw'][hw_name]
    size_tau = t['tau']

    if role == 'obj':
        obj = (ERR_METRIC, SIZE_METRIC)
        constr = ((h['metric'], h['tau'], h['sense']),)
    elif role == 'constr':
        obj = (ERR_METRIC, h['metric'])
        constr = ((SIZE_METRIC, size_tau, 1),)
    elif role == 'multi':
        obj = (ERR_METRIC, h['partner_metric'])
        constr = ((SIZE_METRIC, size_tau, 1), (h['metric'], h['tau'], h['sense']))
    else:
        raise ValueError(f'Unknown role: {role!r}')

    obj_idx = [_metric_index(suite, pid, m) for m in obj]
    # A maximize-better metric used as an OBJECTIVE is minimized as 1 - v.
    flip = [i for i, m in enumerate(obj) if m == h['metric'] and h['sense'] < 0]

    # Which declared constraints stay MEASURABLE for a gated-out architecture:
    # #Params is a static structural count, so it is known even for a model
    # that cannot be run, while a hardware metric of an unrunnable model is
    # not. Only the former may train its constraint surrogate on the rejection
    # log -- otherwise the 'feasible' arm quietly gets oracle information.
    observable = [i for i, c in enumerate(constr) if c[0] == SIZE_METRIC]

    return dict(
        space=name, tightness=tight, role=role, hw=hw_name,
        suite=suite, pid=pid, exact=s['exact'],
        obj_metrics=obj, obj_indices=obj_idx, flip_obj_pos=flip,
        observable_constr=observable,
        constr_metrics=tuple(c[0] for c in constr),
        constr_indices=[_metric_index(suite, pid, c[0]) for c in constr],
        thresholds=[c[1] for c in constr],
        senses=[c[2] for c in constr],
        gate_metric=SIZE_METRIC,
        gate_index=_metric_index(suite, pid, SIZE_METRIC),
        gate_tau=size_tau,
        size_feasible_fraction=t['feasible_fraction'],
        ref_point=s['ref_point'],
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


def cells(name=DEFAULT_SPACE):
    """Every runnable (tightness, role, hw, arm) cell, in a stable order."""
    for t in tightness(name):
        for role in ROLES:
            for h in hw(name):
                if excluded(role, h, name):
                    continue
                for arm in ARMS:
                    yield t, role, h, arm
