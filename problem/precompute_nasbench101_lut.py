"""
One-time preprocessing script: build and save the NASBench-101 lookup table.

The LUT maps canonical_bytes -> arch_str for every valid architecture in the
search space (<=9 raw edges).  Once built (~3-5 min), it is loaded automatically
by NASBench101DuplicateElimination, replacing the expensive ModelSpec + 50-MD5
hash_module pipeline with a fast numpy pruner + O(1) dict lookup on cache misses.

Usage:
    python -m problem.precompute_nasbench101_lut
"""

import pickle
import sys
import time

from problem.nasbench101_utils import LUT_PATH, build_nasbench101_lut

DATA_FILE = 'problem/data/data_nasbench101.pkl'


def main():
    print(f'Loading bench_db from {DATA_FILE} ...')
    with open(DATA_FILE, 'rb') as f:
        bench_db = pickle.load(f)
    print(f'  {len(bench_db):,} entries')

    print(f'\nBuilding LUT (saves to {LUT_PATH}) ...')
    t0 = time.time()
    lut = build_nasbench101_lut(LUT_PATH)
    elapsed = time.time() - t0

    print(f'\nDone in {elapsed:.1f}s  —  {len(lut):,} canonical architectures')

    # Quick sanity check: every LUT value should be a key in bench_db
    missing = sum(1 for v in lut.values() if v not in bench_db)
    if missing:
        print(f'WARNING: {missing} LUT entries not found in bench_db', file=sys.stderr)
    else:
        print('Sanity check passed: all LUT arch_str values present in bench_db.')


if __name__ == '__main__':
    main()
