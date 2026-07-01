"""validate_seeds.py — Check seed completeness and integrity across all experiments.

Recursively searches all experiment result directories and checks whether each
leaf folder (where actual result files are stored) has exactly 20 valid seeds.

Empty (0-byte) seed pickles — left behind by crashed/timed-out runs — are ALWAYS
detected (a fast size check) and counted as missing, so they no longer masquerade
as complete. ``--check_corruption`` additionally unpickles each non-empty file to
catch truncated writes (slow).

Supports filtering by experiment type for focused validation.

Usage:
    # Check all experiments (fast)
    python experiments/validate_seeds.py

    # Check only obj_ga (Exp1)
    python experiments/validate_seeds.py --experiments obj_ga

    # Deep integrity check — also unpickle non-empty files (SLOW, ~1-2 min)
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


def infer_n_obj_from_pid(benchmark: str, pid: int) -> Optional[int]:
    """Infer N_OBJ from BENCHMARK and PID using sbatch script mappings.

    The sbatch scripts hardcode PID sets per (BENCHMARK, N_OBJ) combo.
    This function reverses that mapping to find which N_OBJ contains a given PID.

    Returns the inferred N_OBJ, or None if not found.
    """
    # Mappings from OBJ_GA_EVOXBENCH.sbatch lines 100-105
    pid_sets = {
        'c10mop': {
            2: (1, 8),
            3: (2, 3, 9),
            4: (4, 5, 6, 7),
        },
        'in1kmop': {
            2: (1, 2, 4, 5, 7),
            3: (3, 6, 8),
            4: (9,),
        },
    }

    if benchmark not in pid_sets:
        return None

    for n_obj, pids in pid_sets[benchmark].items():
        if pid in pids:
            return n_obj

    return None


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
                # Infer N_OBJ from benchmark and PID
                n_obj = infer_n_obj_from_pid(suite, pid)
                if n_obj is None:
                    return None
                return {
                    'benchmark': suite,
                    'pid': pid,
                    'n_obj': n_obj,
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
                # Infer N_OBJ from benchmark and PID
                n_obj = infer_n_obj_from_pid(suite, pid)
                if n_obj is None:
                    return None
                return {
                    'benchmark': suite,
                    'pid': pid,
                    'n_obj': n_obj,
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

    NOTE: EvoXBench sbatch scripts use (BENCHMARK, N_OBJ) not PID in exports,
    because PIDs are hardcoded in the script based on that combo.
    """
    array_part = f"--array={array_spec} " if array_spec else ""

    if experiment == 'obj_ga':
        if params.get('benchmark') == 'wfg':
            # WFG: needs problem and n_obj
            return (f"sbatch {array_part}--export=ALL,BENCHMARK=wfg,PROBLEM={params['problem']},"
                   f"N_OBJ={params['n_obj']} experiments/obj_ga/OBJ_GA_WFG.sbatch")
        else:
            # EvoXBench: can use N_OBJ (hardcoded mapping) or explicit PID
            if 'pid' in params:
                # Explicit PID: more targeted, for single-PID reruns
                return (f"sbatch {array_part}--export=ALL,BENCHMARK={params['benchmark']},PID={params['pid']} "
                       f"experiments/obj_ga/OBJ_GA_EVOXBENCH.sbatch")
            else:
                # N_OBJ: uses hardcoded mapping, for multi-PID reruns
                return (f"sbatch {array_part}--export=ALL,BENCHMARK={params['benchmark']},N_OBJ={params.get('n_obj', 2)} "
                       f"experiments/obj_ga/OBJ_GA_EVOXBENCH.sbatch")

    elif experiment == 'obj_ga_pred':
        pred = params.get('predictor', 'xgboost')
        if params.get('benchmark') == 'wfg':
            # WFG: needs problem, n_obj, predictor
            return (f"sbatch {array_part}--export=ALL,BENCHMARK=wfg,PROBLEM={params['problem']},"
                   f"N_OBJ={params['n_obj']},PREDICTORS={pred} "
                   f"experiments/obj_ga_pred/OBJ_GA_PRED_WFG.sbatch")
        else:
            # EvoXBench: can use N_OBJ (hardcoded mapping) or explicit PID
            if 'pid' in params:
                # Explicit PID: more targeted, for single-PID reruns
                return (f"sbatch {array_part}--export=ALL,BENCHMARK={params['benchmark']},PID={params['pid']},"
                       f"PREDICTORS={pred} experiments/obj_ga_pred/OBJ_GA_PRED_EVOXBENCH.sbatch")
            else:
                # N_OBJ: uses hardcoded mapping, for multi-PID reruns
                return (f"sbatch {array_part}--export=ALL,BENCHMARK={params['benchmark']},N_OBJ={params.get('n_obj', 2)},"
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

    if os.path.getsize(filepath) == 0:
        return False, "empty (0 bytes)"

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


def get_seed_status(seed_dir: str, deep: bool = False) -> Dict:
    """Get detailed status of all seeds in a directory.

    Parameters
    ----------
    deep : bool
        ``False`` (default, fast): a seed is "valid" if its file is non-empty —
        just an ``os.path.getsize`` per file, no unpickling.  ``True`` (slow):
        also fully unpickle each non-empty file to catch truncated/corrupt writes.
        Empty (0-byte) files are flagged as corrupt in BOTH modes.

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
            except ValueError:
                continue
            existing.add(seed_num)

            filepath = os.path.join(seed_dir, fname)
            if os.path.getsize(filepath) == 0:
                corrupt.append((seed_num, "empty (0 bytes)"))
            elif deep:
                is_valid, error = check_pkl_integrity(filepath)
                if is_valid:
                    valid.append(seed_num)
                else:
                    corrupt.append((seed_num, error))
            else:
                valid.append(seed_num)   # non-empty; assume OK without unpickling

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

            # Seed validity (fast: size-only by default; deep unpickle if asked).
            # A 0-byte / corrupt seed_*.pkl does NOT count toward completeness.
            seed_status = get_seed_status(dirpath, deep=check_corruption)
            n_valid = len(seed_status['valid'])
            n_seed_files = n_valid + len(seed_status['corrupt'])

            if n_seed_files == 0:
                # No seed_*.pkl here (e.g. an aggregate pareto_approx.pkl) — fall
                # back to the raw .pkl count so aggregate detection still works.
                n_seeds = pkl_count
                status = 'complete' if pkl_count == 20 else ('incomplete' if pkl_count > 0 else 'missing')
                missing = []
            else:
                n_seeds = n_valid
                status = 'complete' if n_valid == 20 else ('incomplete' if n_valid > 0 else 'missing')
                missing = sorted(seed_status['missing'] + [s for s, _ in seed_status['corrupt']])

            leaf_dirs.append({
                'path': dirpath,
                'relative_path': rel_path,
                'n_seeds': n_seeds,
                'status': status,
                'experiment': exp_type,
                'missing_seeds': missing,
                'seed_status': seed_status,
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
        help='Deep integrity check: fully unpickle every non-empty seed to catch '
             'truncated/corrupt writes (SLOW). Empty 0-byte files are ALWAYS '
             'detected and counted as missing, even without this flag (fast).'
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

    # Separate aggregate files from actual experiments
    actual_leaves = []
    aggregate_leaves = []

    for leaf in leaf_dirs:
        parts = leaf['relative_path'].replace('\\', '/').split('/')
        is_aggregate = (
            (leaf['n_seeds'] == 1) and
            ('ga_obj' in parts[-1] or 'ga_obj_pred' in parts[-1])
        )
        if is_aggregate:
            aggregate_leaves.append(leaf)
        else:
            actual_leaves.append(leaf)

    # Organize actual experiments by type
    by_experiment = defaultdict(list)
    for leaf in actual_leaves:
        by_experiment[leaf['experiment']].append(leaf)

    # Summary statistics (excluding aggregate files)
    summary = {}
    issues = []

    for exp_type in sorted(by_experiment.keys()):
        leaves = by_experiment[exp_type]
        complete = sum(1 for l in leaves if l['status'] == 'complete')
        incomplete = sum(1 for l in leaves if l['status'] == 'incomplete')
        missing = sum(1 for l in leaves if l['status'] == 'missing')
        corrupt_seeds = sum(len(l['seed_status'].get('corrupt', [])) for l in leaves)
        total = len(leaves)

        pct_complete = 100 * complete / total if total > 0 else 0
        status_icon = '[OK]' if pct_complete == 100 else '[!!]'

        summary[exp_type] = {
            'total': total,
            'complete': complete,
            'incomplete': incomplete,
            'missing': missing,
            'corrupt_seeds': corrupt_seeds,
            'pct_complete': pct_complete,
            'aggregate_files': sum(1 for l in aggregate_leaves if l['experiment'] == exp_type),
        }

        print(f'  {status_icon} {exp_type:12s}: {complete:4d}/{total:4d} complete '
              f'({pct_complete:5.1f}%) | {incomplete} incomplete | {missing} missing '
              f'| {corrupt_seeds} empty/corrupt seeds')

        # Collect issues
        for leaf in leaves:
            if leaf['status'] != 'complete':
                issues.append(leaf)

    # Print detailed issues if found
    if issues:
        print(f'\n{"=" * 70}')
        print('  Incomplete Directories')
        print(f'{"=" * 70}')

        # Sort issues for display
        actual_issues = sorted(issues, key=lambda x: (x['experiment'], x['relative_path']))

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
        if aggregate_leaves:
            print(f'\n  [CAN IGNORE] Aggregate/pool files (computed, not run):')
            for agg in sorted(aggregate_leaves, key=lambda x: (x['experiment'], x['relative_path'])):
                print(f'    {agg["experiment"]:12s} | {agg["n_seeds"]:2d} seed      | {agg["relative_path"]}')

        # Generate and print sbatch commands if requested
        if args.print_commands:
            print(f'\n{"=" * 70}')
            print('  Commands to rerun incomplete experiments')
            print(f'  (optimized: N_OBJ for multi-PID, PID for single-PID)')
            print(f'{"=" * 70}')

            # Group by (exp, benchmark, n_obj, problem, predictor) to find which PIDs need seeds
            grouped_by_config = defaultdict(lambda: defaultdict(set))

            for issue in actual_issues:
                params = parse_incomplete_path(issue['relative_path'], issue['experiment'])
                if params:
                    exp = issue['experiment']
                    bench = params.get('benchmark')
                    n_obj = params.get('n_obj')
                    problem = params.get('problem')
                    pred = params.get('predictor')
                    pid = params.get('pid')

                    # Create config key (without pid)
                    config_key = (exp, bench, n_obj, problem, pred)

                    # Collect PIDs and their missing seeds
                    if pid is not None:
                        grouped_by_config[config_key][pid].update(issue['missing_seeds'])

            # Generate optimized commands
            commands_with_seeds = []
            for config_key in sorted(grouped_by_config.keys()):
                exp, bench, n_obj, problem, pred = config_key
                pids_missing = grouped_by_config[config_key]

                # Decision: use N_OBJ if multiple PIDs, or PID if just one
                if len(pids_missing) > 1:
                    # Multiple PIDs missing: use N_OBJ approach (more efficient)
                    all_missing = set()
                    for pid, seeds in pids_missing.items():
                        all_missing.update(seeds)

                    all_missing = sorted(all_missing)
                    array_spec = format_array_spec(all_missing)

                    params = {
                        'benchmark': bench,
                        'n_obj': n_obj,
                    }
                    if problem:
                        params['problem'] = problem
                    if pred:
                        params['predictor'] = pred

                    cmd = generate_sbatch_command(exp, params, array_spec)
                    if cmd:
                        pids_list = sorted(pids_missing.keys())
                        commands_with_seeds.append({
                            'cmd': cmd,
                            'missing_seeds': all_missing,
                            'pids': pids_list,
                            'method': 'N_OBJ (multiple PIDs)',
                        })
                else:
                    # Single PID missing: use PID export (more targeted)
                    pid = list(pids_missing.keys())[0]
                    missing_seeds = sorted(pids_missing[pid])
                    array_spec = format_array_spec(missing_seeds)

                    params = {
                        'benchmark': bench,
                        'pid': pid,
                    }
                    if n_obj:
                        params['n_obj'] = n_obj
                    if problem:
                        params['problem'] = problem
                    if pred:
                        params['predictor'] = pred

                    cmd = generate_sbatch_command(exp, params, array_spec)
                    if cmd:
                        commands_with_seeds.append({
                            'cmd': cmd,
                            'missing_seeds': missing_seeds,
                            'pids': [pid],
                            'method': 'PID (single)',
                        })

            for item in commands_with_seeds:
                cmd = item['cmd']
                seeds = item['missing_seeds']
                method = item['method']
                pids = item['pids']
                print(f'  {cmd}')
                print(f'    # Method: {method} | PIDs: {pids} | Seeds: {seeds}')

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
        print(f'[RESULT] [OK] All directories validated successfully!')
        print(f'{"=" * 70}\n')
        sys.exit(0)


if __name__ == '__main__':
    main()
