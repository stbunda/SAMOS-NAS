"""Regenerate (and check) the thresholds hardcoded in config.py.

Reads the NB201 stage-1 enumeration (results2/constraint_analysis/full/NB201,
all 15 625 architectures) and derives every threshold the grid uses. NB201
returns already-normalized objectives, so the parquet columns are exactly the
space ConstrainedEvoXBenchProblem's G lives in -- no benchmark call needed.

Thresholds are MIDPOINTS between attained values, not percentiles: NB201's
#Params has 59 distinct values with plateaus of up to 2 421 architectures, so
percentile thresholds collapse (the 5th and 10th are the same number) and sit
exactly on a tie. Each tau here is the midpoint between the last attained
value inside the target fraction and the next one up, so no architecture sits
on the boundary and the realised feasible fraction is exact.

  python experiments2/sample_selection_bias/derive_taus.py          # print
  python experiments2/sample_selection_bias/derive_taus.py --check  # verify
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

PARQUET = os.path.join(_REPO_ROOT, 'results2', 'constraint_analysis', 'full',
                       'NB201', 'samples.parquet')

# Target feasible fractions. The size levels are the four tightness steps; the
# hardware budgets all sit at the campaign's 10% operating point.
SIZE_TARGETS = dict(zip(CF.TIGHTNESS, (0.05, 0.15, 0.26, 0.52)))
HW_TARGET = 0.10


def tau_midpoint(v, target, sense=1):
    """(tau, realised_fraction) for the largest feasible fraction <= target,
    with tau strictly between two attained values. sense=+1 is a ceiling
    (feasible <=> v <= tau), -1 a floor (feasible <=> v >= tau)."""
    u = np.unique(v)
    if sense > 0:
        frac = np.array([(v <= t).mean() for t in u])
        i = np.where(frac <= target + 1e-12)[0][-1]
        tau = float((u[i] + u[i + 1]) / 2)
        return tau, float((v <= tau).mean())
    frac = np.array([(v >= t).mean() for t in u])
    i = np.where(frac <= target + 1e-12)[0][0]
    tau = float((u[i] + u[i - 1]) / 2)
    return tau, float((v >= tau).mean())


def derive():
    df = pd.read_parquet(PARQUET)
    size = {}
    for level, target in SIZE_TARGETS.items():
        tau, frac = tau_midpoint(df['params'].to_numpy(float), target)
        size[level] = dict(tau=tau, feasible_fraction=frac)
    hw = {}
    for name, cfg in CF.HW.items():
        tau, frac = tau_midpoint(df[name].to_numpy(float), HW_TARGET,
                                 sense=cfg['sense'])
        hw[name] = dict(tau=tau, feasible_fraction=frac)
    return size, hw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='Compare derived values against config.py and exit 1 on '
                         'any mismatch.')
    args = ap.parse_args()
    size, hw = derive()

    print(f'{"level":<32} {"target":>8} {"tau":>22} {"realised":>10}')
    print('-' * 76)
    for level, d in size.items():
        print(f'{"#Params " + level:<32} {SIZE_TARGETS[level]:>8.2f} '
              f'{d["tau"]:>22.17g} {d["feasible_fraction"]:>9.4%}')
    for name, d in hw.items():
        sense = '>=' if CF.HW[name]['sense'] < 0 else '<='
        print(f'{name + " (" + sense + ")":<32} {HW_TARGET:>8.2f} '
              f'{d["tau"]:>22.17g} {d["feasible_fraction"]:>9.4%}')

    if not args.check:
        return 0

    bad = []
    for level, d in size.items():
        live = CF.TIGHTNESS[level]
        for key in ('tau', 'feasible_fraction'):
            if abs(d[key] - live[key]) > 1e-12:
                bad.append(f'TIGHTNESS[{level}][{key}]: derived={d[key]!r} live={live[key]!r}')
    for name, d in hw.items():
        live = CF.HW[name]
        for key in ('tau', 'feasible_fraction'):
            if abs(d[key] - live[key]) > 1e-12:
                bad.append(f'HW[{name}][{key}]: derived={d[key]!r} live={live[key]!r}')
    print('\n-- check against config.py --')
    if bad:
        for b in bad:
            print(f'  MISMATCH {b}')
        return 1
    print('  all values match config.py OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
