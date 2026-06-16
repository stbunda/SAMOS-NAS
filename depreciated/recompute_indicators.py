"""recompute_indicators.py — Fix HV / IGD+ in existing pkl result files.

The original runs used `ref_point = pareto_front.max(axis=0) * 1.05` (raw
objective space) while `test_obj_archive` is stored in normalized [0,1] space.
This script recomputes the indicators correctly from the already-stored
`test_obj_archive` arrays and overwrites the `indicators` key in-place.

Usage
-----
  # Fix everything under results/evoxbench/
  python recompute_indicators.py

  # Fix a specific suite / PID / method
  python recompute_indicators.py --suite c10mop --pids 1 2 3 --methods nsga2 samos-xgb

  # Dry-run: report what would change without writing anything
  python recompute_indicators.py --dry_run
"""

import argparse
import os
import pickle
import warnings

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from problem.evoxbench.utils import get_benchmark


# ─── indicator helpers ────────────────────────────────────────────────────────

def _make_indicators(suite: str, pid: int):
    """Return (hv_ind, igd_ind) built from the *normalized* PF."""
    bm = get_benchmark(suite, pid)
    n_obj = bm.evaluator.n_objs
    ref_point = np.ones(n_obj) * 1.05

    pf_raw = getattr(bm, 'pareto_front', None)
    if pf_raw is not None and len(pf_raw) > 0:
        pf_norm = bm.normalize(pf_raw)
        pf_norm = np.where(np.isfinite(pf_norm), pf_norm, 1.0)
    else:
        pf_norm = None

    hv_ind  = HV(ref_point=ref_point)
    igd_ind = IGDPlus(pf_norm) if pf_norm is not None else None
    return hv_ind, igd_ind


# ─── per-file fix ─────────────────────────────────────────────────────────────

def fix_pkl(path: str, hv_ind: HV, igd_ind, dry_run: bool) -> bool:
    """Recompute indicators in a single pkl file.  Returns True if changed."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    test_obj_archive = data.get('test_obj_archive', [])
    indicators_old   = data.get('indicators', [])

    if not test_obj_archive:
        print(f'  [SKIP] no test_obj_archive: {path}')
        return False

    indicators_new = []
    changed = False

    for gen_idx, test_obj_nd in enumerate(test_obj_archive):
        if test_obj_nd is None or len(test_obj_nd) == 0:
            new_ind = {'hv': 0.0, 'igd_plus': np.inf}
        else:
            new_ind = {
                'hv':       float(hv_ind(test_obj_nd)),
                'igd_plus': float(igd_ind(test_obj_nd))
                            if igd_ind is not None else float('nan'),
            }

        old_ind = indicators_old[gen_idx] if gen_idx < len(indicators_old) else {}
        if old_ind.get('hv') != new_ind['hv'] or old_ind.get('igd_plus') != new_ind['igd_plus']:
            changed = True

        indicators_new.append(new_ind)

    if not changed:
        return False

    if not dry_run:
        data['indicators'] = indicators_new
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    return True


# ─── discovery helpers ────────────────────────────────────────────────────────

def _discover(root: str, suite_filter, pid_filter, method_filter):
    """Yield (suite, pid_str, method, pkl_path) tuples under results/evoxbench/."""
    suites = sorted(os.listdir(root))
    for suite in suites:
        if suite_filter and suite not in suite_filter:
            continue
        suite_dir = os.path.join(root, suite)
        if not os.path.isdir(suite_dir):
            continue
        for pid_str in sorted(os.listdir(suite_dir)):
            if not pid_str.startswith('pid'):
                continue
            try:
                pid = int(pid_str[3:])
            except ValueError:
                continue
            if pid_filter and pid not in pid_filter:
                continue
            pid_dir = os.path.join(suite_dir, pid_str)
            for budget_folder in sorted(os.listdir(pid_dir)):
                bdir = os.path.join(pid_dir, budget_folder)
                if not os.path.isdir(bdir):
                    continue
                for method in sorted(os.listdir(bdir)):
                    if method_filter and method not in method_filter:
                        continue
                    mdir = os.path.join(bdir, method)
                    if not os.path.isdir(mdir):
                        continue
                    for fname in sorted(os.listdir(mdir)):
                        if fname.endswith('.pkl'):
                            yield suite, pid, method, os.path.join(mdir, fname)


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    root = os.path.join('results', 'evoxbench')
    if not os.path.isdir(root):
        print(f'[ERROR] Results directory not found: {root}')
        return

    suite_filter  = set(args.suites)  if args.suites  else None
    pid_filter    = set(args.pids)    if args.pids    else None
    method_filter = set(args.methods) if args.methods else None

    # Cache indicators per (suite, pid) to avoid re-instantiating the benchmark
    _ind_cache: dict[tuple, tuple] = {}

    total = fixed = skipped = 0

    for suite, pid, method, pkl_path in _discover(root, suite_filter, pid_filter, method_filter):
        key = (suite, pid)
        if key not in _ind_cache:
            try:
                _ind_cache[key] = _make_indicators(suite, pid)
            except Exception as e:
                warnings.warn(f'Could not build indicators for {suite}/pid{pid}: {e}')
                _ind_cache[key] = (None, None)

        hv_ind, igd_ind = _ind_cache[key]
        if hv_ind is None:
            skipped += 1
            continue

        total += 1
        changed = fix_pkl(pkl_path, hv_ind, igd_ind, dry_run=args.dry_run)
        if changed:
            fixed += 1
            tag = '[DRY]' if args.dry_run else '[FIX]'
            print(f'{tag} {pkl_path}')

    label = 'Would fix' if args.dry_run else 'Fixed'
    print(f'\nDone. {label} {fixed}/{total} files ({skipped} skipped — benchmark unavailable).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recompute HV/IGD+ in evoxbench pkl files')
    parser.add_argument('--suites',  type=str, nargs='+', default=None,
                        help='Limit to these suites (default: all)')
    parser.add_argument('--pids',    type=int, nargs='+', default=None,
                        help='Limit to these PIDs (default: all)')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Limit to these methods (default: all)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Report what would change without writing files')
    args = parser.parse_args()
    main(args)
