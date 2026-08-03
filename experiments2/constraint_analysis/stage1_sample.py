"""Stage 1 - sample/enumerate every eligible space and cache one Parquet each.

Enumerable spaces (small cardinality) are exhaustively enumerated; large spaces
are randomly sampled to a target unique-valid count under a wall-time budget.
Each Parquet is written with a sidecar manifest recording EvoXBench version,
database fingerprint, seed, families, and the real sampling effort so every
downstream analyzer reads cache only and never re-enters the evaluator.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _adapter as A


def _load_existing(space_dir: str):
    """Return (df, manifest) cached at space_dir, or (None, None) if absent."""
    pq_path = os.path.join(space_dir, 'samples.parquet')
    mf_path = os.path.join(space_dir, 'manifest.json')
    if not (os.path.exists(pq_path) and os.path.exists(mf_path)):
        return None, None
    with open(mf_path) as fh:
        manifest = json.load(fh)
    return pd.read_parquet(pq_path), manifest


def run_space(space: str, out_root: str, n_target: int, seed: int,
              time_budget_s: float, append: bool = False) -> dict:
    space_dir = os.path.join(out_root, space)
    prev_df, prev_manifest = _load_existing(space_dir) if append else (None, None)

    print(f'[{space}] loading benchmark ...')
    bench = A.get_space(space)
    names = A.metric_names(bench)
    print(f'[{space}] metrics={names} normalized={bench.normalized_objectives} '
          f'card={A.cardinality(bench)} enumerable={A.is_enumerable(bench)}')

    if prev_manifest is not None and (prev_manifest.get('complete') or prev_manifest.get('enumerable')):
        print(f'[{space}] already complete ({prev_manifest["n_rows"]:,} rows) '
              f'-- nothing to append, skipping.')
        return prev_manifest

    if prev_df is not None:
        # Same seed as the prior run: the RNG replays that run's exact draw
        # sequence first (cheaply skipped via seen_keys) before drawing new
        # architectures, so the same seed must be reused here.
        seed = int(prev_manifest['seed'])
        seen_keys = set(prev_df.index)
        print(f'[{space}] appending to {len(prev_df):,} cached rows (seed={seed}) '
              f'with a further {time_budget_s:.0f}s budget ...')
    else:
        seen_keys = None

    X, F, keys, stats = A.collect_space(bench, n_target, seed, time_budget_s,
                                         seen_keys=seen_keys)
    print(f'[{space}] collected stats={stats}')

    n_var = X.shape[1]
    df = pd.DataFrame(
        {f'x{i}': X[:, i].astype('int32') for i in range(n_var)}
    )
    for j, m in enumerate(names):
        df[m] = F[:, j].astype('float32')
    df.insert(0, 'arch_key', keys.astype(str))
    df = df.set_index('arch_key')
    if prev_df is not None:
        df = pd.concat([prev_df, df])

    os.makedirs(space_dir, exist_ok=True)
    pq_path = os.path.join(space_dir, 'samples.parquet')
    df.to_parquet(pq_path)

    run_record = {
        'seed': int(seed), 'n_target': int(n_target), 'time_budget_s': float(time_budget_s),
        'stats': stats, 'at_utc': _dt.datetime.utcnow().isoformat() + 'Z',
    }
    runs = (prev_manifest.get('runs', []) if prev_manifest else []) + [run_record]
    manifest = {
        'space': space,
        'suite_pid': A.RICHEST_PID[space],
        'n_var': int(n_var),
        'metrics': names,
        'families': {m: A.metric_family(m) for m in names},
        'normalized': bool(bench.normalized_objectives),
        'cardinality': A.cardinality(bench),
        'enumerable': A.is_enumerable(bench),
        'complete': bool(stats.get('complete', stats.get('hit_target', False))),
        'n_rows': int(len(df)),
        'seed': int(seed),
        'n_target': int(n_target),
        'time_budget_s': float(time_budget_s),
        'sampling_stats': {**stats, 'n_kept': int(len(df))},
        'runs': runs,
        'evoxbench_version': A.evoxbench_version(),
        'database_fingerprint': A.database_fingerprint(),
        'created_utc': _dt.datetime.utcnow().isoformat() + 'Z',
        'parquet': os.path.basename(pq_path),
    }
    with open(os.path.join(space_dir, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'[{space}] wrote {pq_path}  ({len(df):,} rows, {len(names)} metrics)')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spaces', nargs='*', default=list(A.RICHEST_PID),
                    help='space abbreviations to process')
    ap.add_argument('--n_target', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--time_budget', type=float, default=900.0,
                    help='per-space wall-time cap for sampling (enumerable spaces ignore this)')
    ap.add_argument('--output_root', default='results2/constraint_analysis/full',
                    help='directory to write {space}/samples.parquet + manifest.json')
    ap.add_argument('--overwrite', action='store_true',
                    help='overwrite any existing cache instead of resuming+extending it '
                         '(default is to append: reuses the cached seed; spaces with no '
                         'existing cache fall back to a fresh run either way)')
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    summary = {}
    for space in args.spaces:
        m = run_space(space, args.output_root, args.n_target, args.seed, args.time_budget,
                       append=not args.overwrite)
        summary[space] = {k: m[k] for k in ('n_rows', 'enumerable', 'complete', 'sampling_stats')}
    with open(os.path.join(args.output_root, 'stage1_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('Stage 1 done:', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
