"""experiments/samos/experiment_uncertainty.py --- C3 ablation: uncertainty-
aware infill selection vs. SAMOS2's diversity-only baseline.

Compares three variants, identical in every other respect (surrogate family
held fixed at RFR -- an ensemble-of-trees surrogate exposing predict_std --
so the only thing that varies is the *selection* policy, not which surrogate
is used):
  baseline      : DiversitySelector (SAMOSMinimal's crowding-distance
                  diversity policy over the mean prediction).
  acq_lcb       : AcquisitionSelector(kind='lcb') -- select on the
                  lower-confidence-bound-shifted objectives (mean - kappa*std).
  acq_hvi       : AcquisitionSelector(kind='hvi') -- rank by the hypervolume
                  contribution of the LCB-shifted point against the true
                  archive front (see strategy/surrogate/infill.py docstring
                  for why this is an approximation to EHVI, not exact).

Examples
--------
  python experiments/samos/experiment_uncertainty.py                     # default: c10mop/pid1
  python experiments/samos/experiment_uncertainty.py --pid 5 --kappa 1.0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from _common import add_common_args, infer_obj_split, run_comparison

from problem.evoxbench.utils import get_benchmark
from strategy.surrogate.infill import AcquisitionSelector, DiversitySelector


def _problem_scale_ref_point(benchmark):
    """HV reference point in the same normalisation convention SAMOS2 sees
    internally (mirrors EvoXBenchProblem._evaluate's conditional normalize --
    NOT the same convention as _common.TrueMetricsCallback, which always
    normalizes true-eval objectives for offline reporting)."""
    ref = np.asarray(benchmark.hv_ref_point, dtype=float)[None, :]
    if not benchmark.normalized_objectives:
        ref = benchmark.normalize(ref)
    return ref[0]


def main(args):
    benchmark = get_benchmark(args.suite, args.pid)
    n_obj = benchmark.evaluator.n_objs
    predict_idx, _ = infer_obj_split(args.suite, args.pid, n_obj)
    ref_point = _problem_scale_ref_point(benchmark)

    variants = {
        'baseline': {
            'samos2_kwargs': {'infill_selector': DiversitySelector()},
        },
        'acq_lcb': {
            'samos2_kwargs': {
                'infill_selector': AcquisitionSelector(
                    predict_idx, kind='lcb', kappa=args.kappa),
            },
        },
        'acq_hvi': {
            'samos2_kwargs': {
                'infill_selector': AcquisitionSelector(
                    predict_idx, kind='hvi', kappa=args.kappa, ref_point=ref_point),
            },
        },
    }
    run_comparison(args, contribution_name='uncertainty', variants=variants)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, default_output='results/samos/uncertainty.pkl')
    p.add_argument('--kappa', type=float, default=2.0, help='LCB exploration weight.')
    p.set_defaults(pid=1, surrogate='RFR')
    sys.exit(main(p.parse_args()))
