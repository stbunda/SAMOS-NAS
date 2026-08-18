"""Regenerate (and check) the thresholds hardcoded in config.py.

Reads the stage-1 cache (results2/constraint_analysis/full/{space}) and derives
every threshold the grid uses, in the same evaluate space the problem sees --
normalization is applied only where the benchmark does not already normalize
(NB201 does, MobileNetV3 does not).

Thresholds are MIDPOINTS between attained values, not percentiles: NB201's
#Params has 59 distinct values with plateaus of up to 2 421 architectures, so
percentile thresholds collapse (the 5th and 10th are the same number) and sit
exactly on a tie. Each tau here is the midpoint between the last attained value
inside the target fraction and the next one up, so no architecture sits on the
boundary and the realised feasible fraction is exact.

Realised fractions are EXACT for an enumerable space and SAMPLE ESTIMATES for a
sampled one (MobileNetV3's come from the 1M stage-1 draw).

  python experiments2/sample_selection_bias/derive_taus.py --space MobileNetV3
  python experiments2/sample_selection_bias/derive_taus.py --check     # all spaces
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CF
from problem.evoxbench.utils import get_benchmark

PARQUET = os.path.join(_REPO_ROOT, 'results2', 'constraint_analysis', 'full',
                       '{space}', 'samples.parquet')

# Target feasible fractions. The size levels are the four tightness steps; the
# hardware budgets all sit at the campaign's 10% operating point. NB201's size
# targets are offset above the round numbers because its #Params plateaus put
# the attainable fractions at 4.7 / 14.2 / 25.3 / 51.1%.
SIZE_TARGETS = {
    'NB201': dict(T1=0.05, T2=0.15, T3=0.26, T4=0.52),
    'MobileNetV3': dict(T1=0.05, T2=0.15, T3=0.25, T4=0.50),
}
HW_TARGET = 0.10

# Stage-1 parquet column name per benchmark metric display name.
COLUMN = {'#Params': 'params', 'FLOPs': 'flops', 'Err.': 'err',
          'EdgeGPU Lat.': 'edgegpu_latency', 'EdgeGPU En.': 'edgegpu_energy',
          'Eyeriss Lat.': 'eyeriss_latency', 'Eyeriss En.': 'eyeriss_energy',
          'Eyeriss AI': 'eyeriss_arithmetic_intensity', 'Latency': 'latency'}


def tau_midpoint(v, target, sense=1):
    """(tau, realised_fraction) for the largest feasible fraction <= target,
    with tau strictly between two attained values. sense=+1 is a ceiling
    (feasible <=> v <= tau), -1 a floor (feasible <=> v >= tau).

    Cumulative counts over the unique values, not a scan per unique value: the
    sampled spaces have ~350k distinct values in 1M rows, where the naive
    O(n_unique * n) form does not finish.
    """
    u, c = np.unique(v, return_counts=True)
    if sense > 0:
        cum = np.cumsum(c) / len(v)
        i = np.flatnonzero(cum <= target + 1e-12)[-1]
        tau = float((u[i] + u[i + 1]) / 2)
        return tau, float((v <= tau).mean())
    rev = (len(v) - np.concatenate(([0], np.cumsum(c)[:-1]))) / len(v)
    i = np.flatnonzero(rev <= target + 1e-12)[0]
    tau = float((u[i] + u[i - 1]) / 2)
    return tau, float((v >= tau).mean())


def eval_space_frame(space):
    """Stage-1 metric matrix in the evaluate space the problem sees: the
    benchmark's own normalization is applied only when it does not already
    return normalized objectives (in1kmop does not, c10mop does)."""
    df = pd.read_parquet(PARQUET.format(space=space))
    bench = get_benchmark(*CF.suite_pid(space))
    metrics = [c for c in df.columns if not c.startswith('x')]
    F = df[metrics].to_numpy(float)
    if not bench.normalized_objectives:
        u = np.asarray(bench.utopian_point, float)
        n = np.asarray(bench.nadir_point, float)
        F = (F - u) / (n - u)
    return pd.DataFrame(F, columns=metrics)


def derive(space):
    df = eval_space_frame(space)
    size_col = COLUMN[CF.SIZE_METRIC]
    size = {}
    for level, target in SIZE_TARGETS[space].items():
        tau, frac = tau_midpoint(df[size_col].to_numpy(float), target)
        size[level] = dict(tau=tau, feasible_fraction=frac)
    hw = {}
    for name, cfg in CF.hw(space).items():
        tau, frac = tau_midpoint(df[COLUMN[cfg['metric']]].to_numpy(float),
                                 HW_TARGET, sense=cfg['sense'])
        hw[name] = dict(tau=tau, feasible_fraction=frac)
    return size, hw


def report(space, size, hw):
    print(f'\n=== {space} ({"exact" if CF.is_exact(space) else "sampled"}) ===')
    print(f'{"level":<32} {"target":>8} {"tau":>22} {"realised":>10}')
    print('-' * 76)
    for level, d in size.items():
        print(f'{CF.SIZE_METRIC + " " + level:<32} {SIZE_TARGETS[space][level]:>8.2f} '
              f'{d["tau"]:>22.17g} {d["feasible_fraction"]:>9.4%}')
    for name, d in hw.items():
        sense = '>=' if CF.hw(space)[name]['sense'] < 0 else '<='
        print(f'{name + " (" + sense + ")":<32} {HW_TARGET:>8.2f} '
              f'{d["tau"]:>22.17g} {d["feasible_fraction"]:>9.4%}')


def check(space, size, hw):
    bad = []
    for level, d in size.items():
        live = CF.tightness(space)[level]
        for key in ('tau', 'feasible_fraction'):
            if abs(d[key] - live[key]) > 1e-12:
                bad.append(f'{space} tightness[{level}][{key}]: '
                           f'derived={d[key]!r} live={live[key]!r}')
    for name, d in hw.items():
        live = CF.hw(space)[name]
        for key in ('tau', 'feasible_fraction'):
            if abs(d[key] - live[key]) > 1e-12:
                bad.append(f'{space} hw[{name}][{key}]: '
                           f'derived={d[key]!r} live={live[key]!r}')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--space', nargs='+', default=list(CF.SPACES),
                    choices=list(CF.SPACES))
    ap.add_argument('--check', action='store_true',
                    help='Compare derived values against config.py and exit 1 on '
                         'any mismatch.')
    args = ap.parse_args()

    bad = []
    for space in args.space:
        size, hw = derive(space)
        report(space, size, hw)
        if args.check:
            bad += check(space, size, hw)

    if args.check:
        print('\n-- check against config.py --')
        if bad:
            for b in bad:
                print(f'  MISMATCH {b}')
            return 1
        print('  all values match config.py OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
