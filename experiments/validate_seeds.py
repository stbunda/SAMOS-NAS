"""validate_seeds.py — Check seed completeness across all experiments.

Recursively searches all experiment result directories and checks whether each
leaf folder (where actual result files are stored) has exactly 20 seeds.

Supports filtering by experiment type for focused validation.

Usage:
    # Check all experiments
    python experiments/validate_seeds.py

    # Check only obj_ga (Exp1)
    python experiments/validate_seeds.py --experiments obj_ga

    # Check only obj_ga_pred (Exp2)
    python experiments/validate_seeds.py --experiments obj_ga_pred

    # Multiple experiments
    python experiments/validate_seeds.py --experiments obj_ga obj_ga_pred

    # Verbose output (show all leaf dirs)
    python experiments/validate_seeds.py --verbose

    # Show only incomplete (default if issues found)
    python experiments/validate_seeds.py --incomplete_only
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import json
from typing import Optional, List, Dict


def find_leaf_dirs(root_path: str, target_dirs: Optional[List[str]] = None) -> List[Dict]:
    """Find all leaf directories (directories containing .pkl files).

    A leaf directory is one that contains .pkl files and is not a parent of other
    such directories.

    Parameters
    ----------
    root_path : str
        Root directory to search (e.g., 'results/')
    target_dirs : list[str] | None
        If provided, only search within these subdirectories (e.g., ['obj_ga', 'obj_ga_pred'])

    Returns
    -------
    list[dict]
        Each dict: {
            'path': str,              # absolute path to leaf dir
            'relative_path': str,     # relative from root_path
            'n_seeds': int,           # number of .pkl files
            'status': str,            # 'complete', 'incomplete', 'missing'
            'experiment': str,        # e.g., 'obj_ga', 'obj_ga_pred'
        }
    """
    leaf_dirs = []
    root = Path(root_path)

    if not root.exists():
        print(f'[WARN] Root path not found: {root_path}')
        return []

    # Walk the directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # Count .pkl files in this directory
        pkl_count = sum(1 for f in filenames if f.endswith('.pkl'))

        # If this directory has .pkl files, check if any subdirs have them
        # (to avoid duplicating parent dirs)
        has_pkl_children = False
        if pkl_count > 0:
            for dirname in dirnames:
                subdir_path = os.path.join(dirpath, dirname)
                for _, _, subfiles in os.walk(subdir_path):
                    if any(f.endswith('.pkl') for f in subfiles):
                        has_pkl_children = True
                        break
                if has_pkl_children:
                    break

        # If this dir has pkl files but no subdirs have them, it's a leaf
        if pkl_count > 0 and not has_pkl_children:
            rel_path = os.path.relpath(dirpath, root)

            # Determine experiment type from path
            exp_type = rel_path.split(os.sep)[0]  # First component

            # Filter by target directories if specified
            if target_dirs and exp_type not in target_dirs:
                continue

            status = 'complete' if pkl_count == 20 else ('incomplete' if pkl_count > 0 else 'missing')

            leaf_dirs.append({
                'path': dirpath,
                'relative_path': rel_path,
                'n_seeds': pkl_count,
                'status': status,
                'experiment': exp_type,
            })

    return leaf_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Check seed completeness across all experiment result directories.',
    )
    parser.add_argument(
        '--experiments', nargs='+',
        default=['obj_ga', 'obj_ga_pred'],
        choices=['obj_ga', 'obj_ga_pred'],  # Extend this list for future experiments
        help='Which experiments to check. Expand list as new experiments are added.'
    )
    parser.add_argument(
        '--root', default='results',
        help='Root directory containing experiment results (default: results/)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print all leaf directories (including complete ones).'
    )
    parser.add_argument(
        '--incomplete_only', action='store_true',
        help='Print only incomplete directories.'
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Save detailed report to this directory (optional).'
    )
    args = parser.parse_args()

    print(f'\n{"=" * 70}')
    print('  Seed Completeness Validation')
    print(f'{"=" * 70}')
    print(f'  Root:        {args.root}')
    print(f'  Experiments: {", ".join(args.experiments)}')
    print()

    # Find all leaf directories
    leaf_dirs = find_leaf_dirs(args.root, target_dirs=args.experiments)

    if not leaf_dirs:
        print(f'[WARN] No leaf directories found.')
        return

    # Organize by experiment type
    by_experiment = defaultdict(list)
    for leaf in leaf_dirs:
        by_experiment[leaf['experiment']].append(leaf)

    # Summary statistics
    summary = {}
    issues = []

    for exp_type in sorted(by_experiment.keys()):
        leaves = by_experiment[exp_type]
        complete = sum(1 for l in leaves if l['status'] == 'complete')
        incomplete = sum(1 for l in leaves if l['status'] == 'incomplete')
        missing = sum(1 for l in leaves if l['status'] == 'missing')
        total = len(leaves)

        pct_complete = 100 * complete / total if total > 0 else 0
        status_icon = '[OK]' if pct_complete == 100 else '[!!]'

        summary[exp_type] = {
            'total': total,
            'complete': complete,
            'incomplete': incomplete,
            'missing': missing,
            'pct_complete': pct_complete,
        }

        print(f'  {status_icon} {exp_type:12s}: {complete:4d}/{total:4d} complete '
              f'({pct_complete:5.1f}%) | {incomplete} incomplete | {missing} missing')

        # Collect issues
        for leaf in leaves:
            if leaf['status'] != 'complete':
                issues.append(leaf)

    # Print detailed issues if found
    if issues:
        print(f'\n{"=" * 70}')
        print('  Incomplete Directories (need to rerun)')
        print(f'{"=" * 70}')

        for issue in sorted(issues, key=lambda x: (x['experiment'], x['relative_path'])):
            status_str = f"{issue['n_seeds']:2d} seeds" if issue['n_seeds'] > 0 else "MISSING"
            print(f'  {issue["experiment"]:12s} | {status_str:12s} | {issue["relative_path"]}')

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            csv_path = os.path.join(args.output_dir, 'seed_validation_issues.csv')
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['experiment', 'relative_path', 'n_seeds', 'status'])
                writer.writeheader()
                for issue in issues:
                    writer.writerow({
                        'experiment': issue['experiment'],
                        'relative_path': issue['relative_path'],
                        'n_seeds': issue['n_seeds'],
                        'status': issue['status'],
                    })
            print(f'\n  Issues written -> {csv_path}')

    # Print verbose output if requested
    if args.verbose:
        print(f'\n{"=" * 70}')
        print('  All Leaf Directories')
        print(f'{"=" * 70}')

        for leaf in sorted(leaf_dirs, key=lambda x: (x['experiment'], x['relative_path'])):
            status_icon = '✓' if leaf['status'] == 'complete' else '✗'
            print(f'  {status_icon} {leaf["experiment"]:12s} | {leaf["n_seeds"]:2d}/20 | {leaf["relative_path"]}')

    # Summary JSON
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, 'validation_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'  Summary written -> {json_path}')

    # Exit status
    print(f'\n{"=" * 70}')
    if issues:
        n_incomplete = len([i for i in issues if i['status'] == 'incomplete'])
        n_missing = len([i for i in issues if i['status'] == 'missing'])
        print(f'[RESULT] Found {len(issues)} incomplete directories')
        print(f'         ({n_incomplete} with partial seeds, {n_missing} completely missing)')
        print(f'{"=" * 70}\n')
        sys.exit(1)
    else:
        print(f'[RESULT] ✓ All directories validated successfully!')
        print(f'{"=" * 70}\n')
        sys.exit(0)


if __name__ == '__main__':
    main()
