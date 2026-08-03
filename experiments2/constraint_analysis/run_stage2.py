"""Stage 2 driver: run analyzers (a)-(g) over every cached space.

Reads only the Stage-1 Parquet cache; writes per-space CSV/TeX/figures under the
same full/{space}/ tree and a stage2_summary.json for Stage 3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _adapter as A
import _cache as IO
import analyzers as AN


def run_space(full_root, space, force=False):
    df, manifest = IO.load_space(full_root, space)
    od = IO.out_dir(full_root, space)
    print(f'[{space}] rows={len(df):,} metrics={manifest["metrics"]}')
    summ = {'space': space, 'n_rows': int(len(df)), 'enumerable': manifest['enumerable'],
            'tabular': A.is_tabular(space), 'metrics': manifest['metrics']}
    b_ret = None
    # (e)/(f) are the expensive analyzers (repeated front/dominance
    # computation over every row); they accept `force` and, when it's False
    # and their output already exists, just re-render figures from the
    # already-computed numbers/cache instead of recomputing. (a)/(b)/(c)/(d)/(g)
    # are cheap even at 1M rows, so they always just recompute.
    for name, fn in [('a', AN.analyze_a), ('b', AN.analyze_b), ('c', AN.analyze_c),
                     ('d', AN.analyze_d)]:
        try:
            r = fn(space, df, manifest, od)
            if name == 'b':
                b_ret = r
            print(f'  ({name}) ok')
        except Exception as e:
            print(f'  ({name}) FAILED: {e}')
            traceback.print_exc()
            summ.setdefault('errors', {})[name] = str(e)
    for name, fn in [('e', AN.analyze_e), ('f', AN.analyze_f)]:
        try:
            r = fn(space, df, manifest, od, force=force)
            print(f'  ({name}) ok{" (cached)" if r.get("cached") else ""}')
        except Exception as e:
            print(f'  ({name}) FAILED: {e}')
            traceback.print_exc()
            summ.setdefault('errors', {})[name] = str(e)
    try:
        AN.analyze_g(space, df, manifest, od,
                     spearman=b_ret['spearman'] if b_ret else None)
        print('  (g) ok')
    except Exception as e:
        print(f'  (g) FAILED: {e}')
        traceback.print_exc()
        summ.setdefault('errors', {})['g'] = str(e)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_root', default='results2/constraint_analysis/full')
    ap.add_argument('--spaces', nargs='*', default=None)
    ap.add_argument('--force', action='store_true',
                    help="recompute (e)/(f) from scratch instead of re-rendering their "
                         "figures from already-written numerics/cache")
    args = ap.parse_args()
    spaces = args.spaces or [d for d in sorted(os.listdir(args.full_root))
                             if os.path.isdir(os.path.join(args.full_root, d))
                             and os.path.exists(os.path.join(args.full_root, d, 'manifest.json'))]
    # merge into any existing summary: a --spaces run must not drop the entries
    # of spaces it wasn't asked to touch
    summary_path = os.path.join(args.full_root, 'stage2_summary.json')
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for sp in spaces:
        summary[sp] = run_space(args.full_root, sp, force=args.force)
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print('Stage 2 done for:', list(summary))


if __name__ == '__main__':
    main()
