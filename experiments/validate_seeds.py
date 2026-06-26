"""validate_seeds.py — Check seed completeness and integrity across all experiments.

Recursively searches all experiment result directories and checks whether each
leaf folder (where actual result files are stored) has exactly 20 seeds.

Can optionally detect corrupt pickle files (indicates crashed runs).

Supports filtering by experiment type for focused validation.

Usage:
    # Check all experiments (fast)
    python experiments/validate_seeds.py

    # Check only obj_ga (Exp1)
    python experiments/validate_seeds.py --experiments obj_ga

    # Check for corrupt pickle files (SLOW, ~1-2 min)
    python experiments/validate_seeds.py --check_corruption

    # Show sbatch commands to rerun incomplete seeds
    python experiments/validate_seeds.py --print_commands

    # Verbose output (show all leaf dirs)
    python experiments/validate_seeds.py --verbose

    # Comprehensive check with commands (saves to file)
    python experiments/validate_seeds.py --check_corruption --print_commands --output_dir results/validation
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from collections import defaultdict
import json
from typing import Optional, List, Dict


def parse_incomplete_path(relative_path: str, experiment: str) -> Optional[Dict]:
    """Parse an incomplete directory path and extract parameters for rerunning.

    Returns dict with keys needed for sbatch export, or None if can't parse.

    Path formats:
      obj_ga/wfg/{n_obj}_obj/{problem}/{ga}/ga_obj
      obj_ga/evoxbench/{suite}/pid{pid}/{ga}/ga_obj
      obj_ga_pred/wfg/{n_obj}_obj/{problem}/{pred}/{ga}/ga_obj_pred[_1k]
      obj_ga_pred/evoxbench/{suite}/pid{pid}/{pred}/{ga}/ga_obj_pred[_1k]
    """
    parts = relative_path.replace('\\', '/').split('/')

    try:
        if experiment == 'obj_ga':
            if 'wfg' in parts:
                idx = parts.index('wfg')
                # obj_ga/wfg/{n_obj}_obj/{problem}/{ga}/ga_obj
                n_obj_str = parts[idx + 1]  # e.g., '2_obj'
                n_obj = int(n_obj_str.split('_')[0])
                problem = parts[idx + 2]
                ga = parts[idx + 3]
                return {
                    'benchmark': 'wfg',
                    'problem': problem,
                    'n_obj': n_obj,
                    'ga': ga,
                }
            elif 'evoxbench' in parts:
                idx = parts.index('evoxbench')
                # obj_ga/evoxbench/{suite}/pid{pid}/{ga}/ga_obj
                suite = parts[idx + 1]
                pid_str = parts[idx + 2]  # e.g., 'pid1'
                pid = int(pid_str.replace('pid', ''))
                ga = parts[idx + 3]
                # Skip if ga is 'ga_obj' (aggregate file)
                if ga == 'ga_obj':
                    return None
                return {
                    'benchmark': suite,
                    'pid': pid,
                    'ga': ga,
                }

        elif experiment == 'obj_ga_pred':
            if 'wfg' in parts:
                idx = parts.index('wfg')
                # obj_ga_pred/wfg/{n_obj}_obj/{problem}/{pred}/{ga}/ga_obj_pred[_1k]
                n_obj_str = parts[idx + 1]
                n_obj = int(n_obj_str.split('_')[0])
                problem = parts[idx + 2]
                pred = parts[idx + 3]
                ga = parts[idx + 4]
                return {
                    'benchmark': 'wfg',
                    'problem': problem,
                    'n_obj': n_obj,
                    'ga': ga,
                    'predictor': pred,
                }
            elif 'evoxbench' in parts:
                idx = parts.index('evoxbench')
                # obj_ga_pred/evoxbench/{suite}/pid{pid}/{pred}/{ga}/ga_obj_pred[_1k]
                suite = parts[idx + 1]
                pid_str = parts[idx + 2]
                pid = int(pid_str.replace('pid', ''))
                pred = parts[idx + 3]
                ga = parts[idx + 4]
                # Skip if ga is 'ga_obj_pred' (aggregate file)
                if ga == 'ga_obj_pred' or ga == 'ga_obj_pred_1k':
                    return None
                return {
                    'benchmark': suite,
                    'pid': pid,
                    'ga': ga,
                    'predictor': pred,
                }
    except (IndexError, ValueError):
        return None

    return None


def generate_sbatch_command(experiment: str, params: Dict, array_spec: Optional[str] = None) -> Optional[str]:
    """Generate sbatch command to rerun an incomplete experiment.

    Parameters
    ----------
    experiment : str
        'obj_ga' or 'obj_ga_pred'
    params : dict
        Extracted parameters from parse_incomplete_path
    array_spec : str or None
        Array specification (e.g., "0-19" or "5,10,15")

    Returns
    -------
    str or None
        sbatch command, or None if can't generate
    """
    array_part = f"--array={array_spec} " if array_spec else ""

    if experiment == 'obj_ga':
        if params.get('benchmark') == 'wfg':
            # WFG: needs problem and n_obj
            return (f"sbatch {array_part}--export=ALL,BENCHMARK=wfg,PROBLEM={params['problem']},"
                   f"N_OBJ={params['n_obj']} experiments/obj_ga/OBJ_GA_WFG.sbatch")
        else:
            # EvoXBench: needs benchmark and pid
            return (f"sbatch {array_part}--export=ALL,BENCHMARK={params['benchmark']},PID={params['pid']} "
                   f"experiments/obj_ga/OBJ_GA_EVOXBENCH.sbatch")

    elif experiment == 'obj_ga_pred':
        pred = params.get('predictor', 'xgboost')
        if params.get('benchmark') == 'wfg':
            # WFG: needs problem, n_obj, predictor
            return (f"sbatch {array_part}--export=ALL,BENCHMARK=wfg,PROBLEM={params['problem']},"
                   f"N_OBJ={params['n_obj']},PREDICTORS={pred} "
                   f"experiments/obj_ga_pred/OBJ_GA_PRED_WFG.sbatch")
        else:
            # EvoXBench: needs benchmark, pid, predictor
            return (f"sbatch {array_part}--export=ALL,BENCHMARK={params['benchmark']},PID={params['pid']},"
                   f"PREDICTORS={pred} experiments/obj_ga_pred/OBJ_GA_PRED_EVOXBENCH.sbatch")

    return None


def check_pkl_integrity(filepath: str) -> tuple[bool, Optional[str]]:
    """Check if a pickle file is readable and valid.

    Returns: (is_valid, error_message)
        is_valid: True if file is readable, False if corrupt or error
        error_message: None if valid, otherwise error description
    """
    if not os.path.isfile(filepath):
        return False, "File not found"

    try:
        with open(filepath, 'rb') as f:
            pickle.load(f)
        return True, None
    except EOFError:
        return False, "EOF error (incomplete write)"
    except pickle.UnpicklingError as e:
        return False, f"Unpickling error: {str(e)[:50]}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {str(e)[:50]}"


def get_seed_status(seed_dir: str) -> Dict:
    """Get detailed status of all seeds in a directory.

    Returns dict with:
        valid: list of valid seed indices
        corrupt: list of (seed_index, error_msg) tuples
        missing: list of missing seed indices
    """
    if not os.path.isdir(seed_dir):
        return {
            'valid': [],
            'corrupt': [],
            'missing': list(range(20)),
        }

    valid = []
    corrupt = []
    existing = set()

    for fname in sorted(os.listdir(seed_dir)):
        if fname.startswith('seed_') and fname.endswith('.pkl'):
            try:
                seed_num = int(fname.replace('seed_', '').replace('.pkl', ''))
                existing.add(seed_num)

                filepath = os.path.join(seed_dir, fname)
                is_valid, error = check_pkl_integrity(filepath)

                if is_valid:
                    valid.append(seed_num)
                else:
                    corrupt.append((seed_num, error))
            except ValueError:
                pass

    missing = sorted([i for i in range(20) if i not in existing])

    return {
        'valid': valid,
        'corrupt': corrupt,
        'missing': missing,
    }


def get_missing_seeds(seed_dir: str) -> List[int]:
    """Get list of missing seed indices (0-19) in a directory.

    Assumes 20 total seeds indexed 0-19 in format seed_{i}.pkl.
    Returns missing + corrupt seeds.
    """
    status = get_seed_status(seed_dir)
    return sorted(status['missing'] + [s for s, _ in status['corrupt']])


def format_array_spec(missing_seeds: List[int]) -> str:
    """Format missing seed indices as sbatch --array specification.

    Examples:
        [0,1,2] -> "0-2"
        [5,10,15] -> "5,10,15"
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] -> "0-19"
    """
    if not missing_seeds:
        return None

    if len(missing_seeds) == 20:
        return "0-19"

    # Check if it's a continuous range
    if missing_seeds[-1] - missing_seeds[0] + 1 == len(missing_seeds):
        return f"{missing_seeds[0]}-{missing_seeds[-1]}"

    # Otherwise list them explicitly
    return ",".join(map(str, missing_seeds))


def find_leaf_dirs(root_path: str, target_dirs: Optional[List[str]] = None, check_corruption: bool = False) -> List[Dict]:
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

            missing = get_missing_seeds(dirpath)

            # Optionally check for corruption
            seed_status = {}
            if check_corruption:
                seed_status = get_seed_status(dirpath)

            leaf_dirs.append({
                'path': dirpath,
                'relative_path': rel_path,
                'n_seeds': pkl_count,
                'status': status,
                'experiment': exp_type,
                'missing_seeds': missing,
                'seed_status': seed_status,  # Only populated if check_corruption=True
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
    parser.add_argument(
        '--print_commands', action='store_true',
        help='Print sbatch commands to rerun incomplete experiments.'
    )
    parser.add_argument(
        '--check_corruption', action='store_true',
        help='Check for corrupt pickle files (SLOW: ~1-2 min for full dataset; '
             'useful for detecting crashed runs).'
    )
    args = parser.parse_args()

    print(f'\n{"=" * 70}')
    print('  Seed Completeness Validation')
    print(f'{"=" * 70}')
    print(f'  Root:        {args.root}')
    print(f'  Experiments: {", ".join(args.experiments)}')
    if args.check_corruption:
        print(f'  Corruption:  Enabled (may take 1-2 min on full dataset)')
    print()

    # Find all leaf directories
    leaf_dirs = find_leaf_dirs(args.root, target_dirs=args.experiments, check_corruption=args.check_corruption)

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
        print('  Incomplete Directories')
        print(f'{"=" * 70}')

        # Separate aggregate files from actual experiments
        actual_issues = []
        aggregate_files = []

        for issue in sorted(issues, key=lambda x: (x['experiment'], x['relative_path'])):
            # Check if this is an aggregate file (no GA, just 1 seed at pid level)
            parts = issue['relative_path'].replace('\\', '/').split('/')
            is_aggregate = (
                (issue['n_seeds'] == 1) and
                ('ga_obj' in parts[-1] or 'ga_obj_pred' in parts[-1])
            )

            if is_aggregate:
                aggregate_files.append(issue)
            else:
                actual_issues.append(issue)

        # Print actual experiments that need rerun
        if actual_issues:
            print('  [NEED RERUN] Actual experiments:')
            for issue in actual_issues:
                status_str = f"{issue['n_seeds']:2d} seeds" if issue['n_seeds'] > 0 else "MISSING"
                print(f'    {issue["experiment"]:12s} | {status_str:12s} | {issue["relative_path"]}')

                # Show corruption details if available
                if issue.get('seed_status'):
                    status = issue['seed_status']
                    if status.get('corrupt'):
                        print(f'      WARNING: Corrupt seeds: {[s for s, _ in status["corrupt"]]}')
                        for seed_num, error in status['corrupt']:
                            print(f'        Seed {seed_num}: {error}')

        # Print aggregate files (can be ignored)
        if aggregate_files:
            print(f'\n  [CAN IGNORE] Aggregate/pool files (computed, not run):')
            for issue in aggregate_files:
                print(f'    {issue["experiment"]:12s} | {issue["n_seeds"]:2d} seed      | {issue["relative_path"]}')

        # Generate and print sbatch commands if requested
        if args.print_commands:
            print(f'\n{"=" * 70}')
            print('  Commands to rerun incomplete experiments')
            print(f'  (with specific seeds to rerun)')
            print(f'{"=" * 70}')

            # Group actual issues (not aggregates) by parameters to collect all missing seeds
            grouped = defaultdict(list)

            for issue in actual_issues:
                params = parse_incomplete_path(issue['relative_path'], issue['experiment'])
                if params:  # Skip aggregate files and unparseable paths
                    # Create a hashable key from params
                    exp = issue['experiment']
                    key_parts = [exp, params.get('benchmark')]
                    if 'problem' in params:
                        key_parts.append(params['problem'])
                    if 'pid' in params:
                        key_parts.append(str(params['pid']))
                    if 'predictor' in params:
                        key_parts.append(params['predictor'])
                    key = tuple(key_parts)

                    grouped[key].append((params, issue['missing_seeds']))

            # Generate commands with array specs
            commands_with_seeds = []
            for key in sorted(grouped.keys()):
                items = grouped[key]
                exp = items[0][0]  # Use first item's experiment type
                params = items[0][0]  # Use first item's params
                experiment_type = key[0]

                # Collect all missing seeds for this parameter combo
                all_missing = set()
                for _, missing_seeds in items:
                    all_missing.update(missing_seeds)

                if all_missing:
                    all_missing = sorted(all_missing)
                    array_spec = format_array_spec(all_missing)
                    cmd = generate_sbatch_command(experiment_type, params, array_spec)
                    if cmd:
                        commands_with_seeds.append({
                            'cmd': cmd,
                            'missing_seeds': all_missing,
                        })

            for item in commands_with_seeds:
                cmd = item['cmd']
                seeds = item['missing_seeds']
                print(f'  {cmd}')
                print(f'    # Seeds: {seeds}')

            print(f'\n  Total commands: {len(commands_with_seeds)}')

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
