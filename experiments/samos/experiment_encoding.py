"""experiments/samos/experiment_encoding.py --- C1 ablation: encoding-aware
surrogate features + operator vs. SAMOS2's raw-integer baseline.

Compares two variants, identical in every other respect:
  baseline  : surrogates trained on raw integer genes; IntegerPointMutation
              (uniform-reset every gene, categorical or not).
  encoding  : surrogates trained on EncodingSpec-transformed features
              (categorical op genes one-hot, ordinal genes passthrough);
              EncodingAwareMutation (uniform-reset for categorical genes,
              local +/-1 step for ordinal genes).

Only runs on search spaces with an OP_VAR_GROUPS entry (NB101/NB201/NATS) --
raises for search spaces with no categorical-op structure, where
encoding_spec would be a no-op (see benchmark_meta.get_op_var_group).

Examples
--------
  python experiments/samos/experiment_encoding.py                      # default: c10mop/pid5 (NB201)
  python experiments/samos/experiment_encoding.py --pid 1 --seeds 5    # NB101
  python experiments/samos/experiment_encoding.py --suite c10mop --pid 3 --n_gen_max 10  # NATS
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from _common import add_common_args, run_comparison

from problem.evoxbench.benchmark_meta import BENCHMARK_META, get_op_var_group
from problem.evoxbench.utils import get_benchmark
from strategy.operations.mutation import EncodingAwareMutation, IntegerPointMutation
from strategy.surrogate.encoding import EncodingSpec


def main(args):
    meta = BENCHMARK_META.get(args.suite, {}).get(args.pid)
    if meta is None:
        raise ValueError(f'Unknown {args.suite}/pid{args.pid}')
    search_space = meta['search_space']
    if get_op_var_group(search_space) is None:
        raise ValueError(
            f'{args.suite}/pid{args.pid} ({search_space}) has no categorical-op '
            f'structure in OP_VAR_GROUPS -- encoding_spec would be a no-op here. '
            f'Pick a NB101/NB201/NATS pid instead.')

    benchmark = get_benchmark(args.suite, args.pid)
    n_var = benchmark.search_space.n_var
    xu = benchmark.search_space.ub
    encoding_spec = EncodingSpec.from_search_space(search_space, n_var, xu)
    categorical_cols = encoding_spec.categorical_cols

    variants = {
        'baseline': {
            'samos2_kwargs': {},
            'mutation_factory': lambda xl, xu: IntegerPointMutation(xl, xu),
        },
        'encoding': {
            'samos2_kwargs': {'encoding_spec': encoding_spec},
            'mutation_factory': lambda xl, xu: EncodingAwareMutation(
                xl, xu, categorical_cols=categorical_cols),
        },
    }
    run_comparison(args, contribution_name='encoding', variants=variants)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, default_output='results/samos/encoding.pkl')
    p.set_defaults(pid=5)
    sys.exit(main(p.parse_args()))
