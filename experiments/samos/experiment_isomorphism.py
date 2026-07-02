"""experiments/samos/experiment_isomorphism.py --- C2 ablation: canonical-
phenotype dedup vs. SAMOS2's raw-vector-dedup baseline.

Compares two variants, identical in every other respect:
  baseline  : raw-vector archive dedup + raw-vector inner-GA duplicate
              elimination (SAMOSMinimal's default -- genotypes that decode
              to the same phenotype are NOT recognised as duplicates).
  canonical : Canonicalizer-based archive dedup + phenotype-aware inner-GA
              duplicate elimination + surrogate training-set collapse to one
              row per unique phenotype (see strategy/surrogate/canonical.py).

Targets search spaces with redundant/many-to-one genotype encodings, where
raw-vector dedup under-counts how many *distinct* architectures have already
been evaluated: NASBench-101 (many op/edge genotypes decode to the same
graph) and DARTS (genotype encoding redundancy).

Examples
--------
  python experiments/samos/experiment_isomorphism.py                    # default: c10mop/pid1 (NB101)
  python experiments/samos/experiment_isomorphism.py --pid 8 --seeds 5  # DARTS
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from _common import add_common_args, run_comparison

from problem.evoxbench.utils import get_benchmark
from strategy.surrogate.canonical import Canonicalizer


def main(args):
    benchmark = get_benchmark(args.suite, args.pid)
    canonicalizer = Canonicalizer(benchmark)

    variants = {
        'baseline': {
            'samos2_kwargs': {},
        },
        'canonical': {
            'samos2_kwargs': {
                'canonicalizer': canonicalizer,
                'collapse_training_set': True,
            },
        },
    }
    run_comparison(args, contribution_name='isomorphism', variants=variants)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, default_output='results/samos/isomorphism.pkl')
    p.set_defaults(pid=1)
    sys.exit(main(p.parse_args()))
