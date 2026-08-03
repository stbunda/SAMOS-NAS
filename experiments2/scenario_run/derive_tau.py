"""Derive tau@10% feasibility, per-scenario HV reference points, and the h2
static-penalty weight for S1-S8.

Auditable regeneration: reads from results2/constraint_analysis/full/<space>,
applies normalization as ConstrainedEvoXBenchProblem sees it, filters MoSegNAS
x0!=0, and computes tau, ref_point, and penalty per scenario. Use --check to
verify against scenarios.py.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = r'c:\Users\BundaST\PycharmProjects\SAMOS-Project\SAMOS-NAS-TEVC'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'experiments2', 'constraint_analysis'))

import _adapter as A
from problem.evoxbench.utils import get_benchmark
from problem.evoxbench.benchmark_meta import metric_index


FULL = os.path.join(REPO, 'results2', 'constraint_analysis', 'full')

SCENARIOS = {
    'S1': ('NATS',        'c10mop',     4, ('err', 'params'),          'flops'),
    'S2': ('ResNet-50D',  'in1kmop',    3, ('err', 'flops'),           'params'),
    'S3': ('MobileNetV3', 'in1kmop',    9, ('err', 'flops'),           'params'),
    'S4': ('NB201',       'c10mop',     7, ('err', 'edgegpu_latency'), 'params'),
    'S5': ('NB201',       'c10mop',     7, ('err', 'params'),          'edgegpu_latency'),
    'S6': ('MoSegNAS',    'citysegmop', 15, ('err', 'params'),         'h1_latency'),
    'S7': ('MoSegNAS',    'citysegmop', 15, ('err', 'params'),         'h2_latency'),
    'S8': ('NB201',       'c10mop',     7, ('err', 'eyeriss_latency'), 'eyeriss_arithmetic_intensity'),
}

NAME_MAP = {
    'NATS':        {'err': 'Err.', 'params': '#Params', 'flops': 'FLOPs'},
    'NB201':       {'err': 'Err.', 'params': '#Params', 'flops': 'FLOPs',
                    'edgegpu_latency': 'EdgeGPU Lat.', 'eyeriss_latency': 'Eyeriss Lat.',
                    'eyeriss_arithmetic_intensity': 'Eyeriss AI'},
    'ResNet-50D':  {'err': 'Err.', 'params': '#Params', 'flops': 'FLOPs'},
    'MobileNetV3': {'err': 'Err.', 'params': '#Params', 'flops': 'FLOPs'},
    'MoSegNAS':    {'err': 'Err.', 'params': '#Params', 'h1_latency': 'H1 Lat.',
                    'h2_latency': 'H2 Lat.'},
}

PCT = 10.0
REF_PCT = 95.0


def eval_space_frame(space):
    """Sampled metric matrix in benchmark-evaluate space (normalized as
    ConstrainedEvoXBenchProblem sees it), plus the manifest metric order."""
    mf = json.load(open(os.path.join(FULL, space, 'manifest.json')))
    df = pd.read_parquet(os.path.join(FULL, space, 'samples.parquet'))
    if space == 'MoSegNAS':
        df = df[df.x0 != 0].reset_index(drop=True)
    cols = list(mf['metrics'])
    F = df[cols].to_numpy(float)
    suite, pid = mf['suite_pid']
    bench = get_benchmark(suite, pid)
    if not bench.normalized_objectives:
        u = np.asarray(bench.utopian_point, float)
        n = np.asarray(bench.nadir_point, float)
        F = (F - u) / (n - u)
    return F, cols, bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='Compare derived values against scenarios.py')
    args = parser.parse_args()

    cache = {}
    rows = []

    for sid, (space, suite, pid, objs, constr) in SCENARIOS.items():
        if space not in cache:
            cache[space] = eval_space_frame(space)
        F, cols, bench = cache[space]

        cv = F[:, cols.index(constr)]
        direction = A.metric_direction(constr)
        tau = A.feasibility_tau(cv, PCT, constr)
        feas = A.feasible_mask(cv, tau, constr)
        sense = 1 if direction == 'min' else -1

        obj_idx = [cols.index(o) for o in objs]
        ref = [float(max(1.05, np.percentile(F[:, j], REF_PCT))) for j in obj_idx]

        # penalty: a median-violating architecture (CV_median, over the WHOLE
        # sample, not just the infeasible rows) is penalised by one objective
        # range (span). At this campaign's 10% feasibility operating point,
        # >=50% of the sample IS infeasible, so CV_median > 0 (see TAU.md).
        CV = np.maximum(0.0, sense * (cv - tau) / tau)
        cv_median = float(np.percentile(CV, 50))
        span = max(r - float(np.min(F[:, j])) for r, j in zip(ref, obj_idx))
        penalty = span / cv_median

        rows.append(dict(
            sid=sid, space=space, suite=suite, pid=pid,
            obj_metrics=tuple(NAME_MAP[space][o] for o in objs),
            constr_metric=NAME_MAP[space][constr],
            direction=direction, tau=float(tau),
            feasible_fraction=float(feas.mean()), ref_point=tuple(ref),
            penalty=float(penalty),
        ))

    print(f'{"S":<3} {"space":<12} {"inst":<15} {"constraint":<14} {"dir":<4} '
          f'{"tau":>10} {"feas%":>7} {"ref point":>22} {"penalty":>10}')
    print('-' * 108)
    for r in rows:
        inst = f'{r["suite"]}/{r["pid"]}'
        ref = '(' + ', '.join(f'{v:.3f}' for v in r['ref_point']) + ')'
        print(f'{r["sid"]:<3} {r["space"]:<12} {inst:<15} {r["constr_metric"]:<14} '
              f'{r["direction"]:<4} {r["tau"]:>10.5f} {100*r["feasible_fraction"]:>6.2f}% {ref:>22} '
              f'{r["penalty"]:>10.5f}')

    print('\n-- guards --')
    bad = [r for r in rows if r['tau'] <= 0]
    print(f'tau > 0 (G=(m-tau)/tau sign safety): '
          f'{"OK" if not bad else "FAIL " + str([r["sid"] for r in bad])}')

    if args.check:
        print('\n-- check against scenarios.py --')
        from scenarios import SCENARIOS as SCEN_LIVE
        mismatches = []
        for r in rows:
            sid = r['sid']
            live = SCEN_LIVE[sid]
            tol = 1e-9
            if abs(r['tau'] - live['tau']) > tol:
                mismatches.append(f'{sid} tau: derived={r["tau"]}, live={live["tau"]}')
            if abs(r['feasible_fraction'] - live['feasible_fraction']) > tol:
                mismatches.append(f'{sid} feasible_fraction: derived={r["feasible_fraction"]}, '
                                  f'live={live["feasible_fraction"]}')
            for i, (d, l) in enumerate(zip(r['ref_point'], live['ref_point'])):
                if abs(d - l) > tol:
                    mismatches.append(f'{sid} ref_point[{i}]: derived={d}, live={l}')
            if abs(r['penalty'] - live['penalty']) > tol:
                mismatches.append(f'{sid} penalty: derived={r["penalty"]}, live={live["penalty"]}')
        if mismatches:
            print('MISMATCH:')
            for m in mismatches:
                print(f'  {m}')
            sys.exit(1)
        else:
            print('All values match scenarios.py OK')


if __name__ == '__main__':
    main()
