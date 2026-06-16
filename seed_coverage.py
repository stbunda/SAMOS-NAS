#!/usr/bin/env python
"""
Map seed coverage across experiments and algorithms.

Scans results/obj_ga/ for completed pickles and summarizes how many seeds
are present per (benchmark, n_obj, problem/pid, method) combination.

Usage:
    python seed_coverage.py [--results_root RESULTS_ROOT] [--format {table,csv,json}] [--output OUTPUT_PATH]

Output formats:
    table: human-readable ASCII table grouped by benchmark/n_obj
    csv: CSV output (benchmark, n_obj, problem/pid, method, seed_count, seeds)
    json: JSON with nested structure
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def scan_results(results_root):
    """
    Scan results_root and return a nested dict of seed counts.

    Structure:
      {
        'benchmark': {
          'wfg': {
            'n_obj': {
              2: {
                'problem': {
                  'wfg1': {
                    'method': {
                      'nsga2': {'seeds': [0, 1, 2, ...], 'count': 3}
                    }
                  }
                }
              }
            }
          },
          'evoxbench': {
            'suite': {
              'c10mop': {
                'n_obj': {
                  2: {
                    'pid': {
                      1: {
                        'method': {
                          'nsga2': {'seeds': [0, 1, ...], 'count': 3}
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    """
    coverage = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))))))

    results_path = Path(results_root)

    # ─── WFG scanning ─────────────────────────────────────────────────────────
    wfg_root = results_path / 'wfg'
    if wfg_root.exists():
        for n_obj_dir in wfg_root.iterdir():
            if not n_obj_dir.is_dir():
                continue
            try:
                n_obj = int(n_obj_dir.name.split('_')[0])
            except (ValueError, IndexError):
                continue

            for problem_dir in n_obj_dir.iterdir():
                if not problem_dir.is_dir():
                    continue
                problem = problem_dir.name

                for method_dir in problem_dir.iterdir():
                    if not method_dir.is_dir():
                        continue
                    method = method_dir.name

                    # Look for ga_obj subdirectory
                    ga_obj_dir = method_dir / 'ga_obj'
                    if not ga_obj_dir.exists():
                        ga_obj_dir = method_dir

                    seeds = []
                    for pkl_file in ga_obj_dir.glob('seed_*.pkl'):
                        try:
                            seed = int(pkl_file.stem.split('_')[1])
                            seeds.append(seed)
                        except (ValueError, IndexError):
                            continue

                    if seeds:
                        seeds.sort()
                        coverage['wfg']['wfg'][n_obj][problem][method] = {
                            'seeds': seeds,
                            'count': len(seeds)
                        }

    # ─── EvoXBench scanning ───────────────────────────────────────────────────
    evoxbench_root = results_path / 'evoxbench'
    if evoxbench_root.exists():
        for suite_dir in evoxbench_root.iterdir():
            if not suite_dir.is_dir():
                continue
            suite = suite_dir.name

            for pid_dir in suite_dir.iterdir():
                if not pid_dir.is_dir() or not pid_dir.name.startswith('pid'):
                    continue
                try:
                    pid = int(pid_dir.name[3:])
                except ValueError:
                    continue

                for method_dir in pid_dir.iterdir():
                    if not method_dir.is_dir():
                        continue
                    method = method_dir.name

                    # Look for ga_obj subdirectory
                    ga_obj_dir = method_dir / 'ga_obj'
                    if not ga_obj_dir.exists():
                        ga_obj_dir = method_dir

                    seeds = []
                    for pkl_file in ga_obj_dir.glob('seed_*.pkl'):
                        try:
                            seed = int(pkl_file.stem.split('_')[1])
                            seeds.append(seed)
                        except (ValueError, IndexError):
                            continue

                    if seeds:
                        seeds.sort()
                        # Infer n_obj from the first available pickle's config
                        n_obj = infer_n_obj(ga_obj_dir, suite, pid) or 'unknown'
                        coverage['evoxbench'][suite][n_obj][pid][method] = {
                            'seeds': seeds,
                            'count': len(seeds)
                        }

    return coverage


def infer_n_obj(ga_obj_dir, suite, pid):
    """Try to infer n_obj from a pickle's config dict."""
    try:
        import pickle
        for pkl_file in ga_obj_dir.glob('seed_*.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    if 'config' in data and 'n_obj' in data['config']:
                        return data['config']['n_obj']
            except Exception:
                continue
    except Exception:
        pass
    return None


def format_table(coverage):
    """Format coverage as human-readable ASCII table grouped by benchmark."""
    lines = []

    # ─── WFG tables ───────────────────────────────────────────────────────────
    if 'wfg' in coverage and coverage['wfg'].get('wfg'):
        lines.append('\n' + '='*100)
        lines.append('WFG Results')
        lines.append('='*100)

        for n_obj in sorted(coverage['wfg']['wfg'].keys()):
            subexp_map = {2: '1.1', 3: '1.2', 4: '1.3'}
            subexp_id = subexp_map.get(int(n_obj), f'?.{int(n_obj)}')
            lines.append(f'\nSub-exp {subexp_id} (n_obj={n_obj})')
            lines.append('-' * 100)

            problems_methods = defaultdict(dict)
            for problem, methods in coverage['wfg']['wfg'][n_obj].items():
                for method, data in methods.items():
                    if problem not in problems_methods:
                        problems_methods[problem] = {}
                    problems_methods[problem][method] = data['count']

            # Header
            all_methods = set()
            for methods in problems_methods.values():
                all_methods.update(methods.keys())
            all_methods = sorted(all_methods)

            header = f"{'Problem':<15}" + ''.join(f"{m:<12}" for m in all_methods) + "  Status"
            lines.append(header)
            lines.append('-' * len(header))

            # Rows
            for problem in sorted(problems_methods.keys()):
                row = f"{problem:<15}"
                counts = []
                for method in all_methods:
                    count = problems_methods[problem].get(method, 0)
                    counts.append(count)
                    row += f"{count:<12}"

                # Status: all complete (20/20), partial, or missing
                if all(c == 20 for c in counts if c > 0):
                    status = "[OK] complete"
                elif any(c > 0 for c in counts):
                    status = f"[PART] partial ({sum(1 for c in counts if c > 0)}/{len(all_methods)} methods)"
                else:
                    status = "[MISS] missing"

                row += f"  {status}"
                lines.append(row)

    # ─── EvoXBench tables ─────────────────────────────────────────────────────
    if 'evoxbench' in coverage:
        for suite in sorted(coverage['evoxbench'].keys()):
            lines.append('\n' + '='*100)
            lines.append(f'EvoXBench - {suite.upper()}')
            lines.append('='*100)

            for n_obj in sorted(coverage['evoxbench'][suite].keys()):
                subexp_map = {2: '1.1', 3: '1.2', 4: '1.3'}
                subexp_id = subexp_map.get(int(n_obj), f'?.{int(n_obj)}')
                lines.append(f'\nSub-exp {subexp_id} (n_obj={n_obj})')
                lines.append('-' * 100)

                pids_methods = defaultdict(dict)
                for pid, methods in coverage['evoxbench'][suite][n_obj].items():
                    for method, data in methods.items():
                        if pid not in pids_methods:
                            pids_methods[pid] = {}
                        pids_methods[pid][method] = data['count']

                # Header
                all_methods = set()
                for methods in pids_methods.values():
                    all_methods.update(methods.keys())
                all_methods = sorted(all_methods)

                header = f"{'PID':<6}" + ''.join(f"{m:<12}" for m in all_methods) + "  Status"
                lines.append(header)
                lines.append('-' * len(header))

                # Rows
                for pid in sorted(pids_methods.keys()):
                    row = f"{pid:<6}"
                    counts = []
                    for method in all_methods:
                        count = pids_methods[pid].get(method, 0)
                        counts.append(count)
                        row += f"{count:<12}"

                    if all(c == 20 for c in counts if c > 0):
                        status = "[OK] complete"
                    elif any(c > 0 for c in counts):
                        status = f"[PART] partial ({sum(1 for c in counts if c > 0)}/{len(all_methods)} methods)"
                    else:
                        status = "[MISS] missing"

                    row += f"  {status}"
                    lines.append(row)

    return '\n'.join(lines)


def format_csv(coverage):
    """Format coverage as CSV."""
    lines = ['benchmark,suite,n_obj,problem_or_pid,method,seed_count,seeds']

    # WFG
    if 'wfg' in coverage and coverage['wfg'].get('wfg'):
        for n_obj in sorted(coverage['wfg']['wfg'].keys()):
            for problem in sorted(coverage['wfg']['wfg'][n_obj].keys()):
                for method, data in coverage['wfg']['wfg'][n_obj][problem].items():
                    seeds_str = ','.join(str(s) for s in data['seeds'])
                    lines.append(f'wfg,,{n_obj},{problem},{method},{data["count"]},"{seeds_str}"')

    # EvoXBench
    if 'evoxbench' in coverage:
        for suite in sorted(coverage['evoxbench'].keys()):
            for n_obj in sorted(coverage['evoxbench'][suite].keys()):
                for pid in sorted(coverage['evoxbench'][suite][n_obj].keys()):
                    for method, data in coverage['evoxbench'][suite][n_obj][pid].items():
                        seeds_str = ','.join(str(s) for s in data['seeds'])
                        lines.append(f'evoxbench,{suite},{n_obj},pid{pid},{method},{data["count"]},"{seeds_str}"')

    return '\n'.join(lines)


def format_json(coverage):
    """Format coverage as JSON (flatten for JSON compatibility)."""
    result = {
        'wfg': {},
        'evoxbench': {}
    }

    # WFG
    if 'wfg' in coverage and coverage['wfg'].get('wfg'):
        for n_obj in sorted(coverage['wfg']['wfg'].keys()):
            key = f'n_obj_{n_obj}'
            result['wfg'][key] = {}
            for problem in sorted(coverage['wfg']['wfg'][n_obj].keys()):
                result['wfg'][key][problem] = {
                    method: {
                        'count': data['count'],
                        'seeds': data['seeds']
                    }
                    for method, data in coverage['wfg']['wfg'][n_obj][problem].items()
                }

    # EvoXBench
    if 'evoxbench' in coverage:
        for suite in sorted(coverage['evoxbench'].keys()):
            result['evoxbench'][suite] = {}
            for n_obj in sorted(coverage['evoxbench'][suite].keys()):
                key = f'n_obj_{n_obj}'
                result['evoxbench'][suite][key] = {}
                for pid in sorted(coverage['evoxbench'][suite][n_obj].keys()):
                    result['evoxbench'][suite][key][f'pid_{pid}'] = {
                        method: {
                            'count': data['count'],
                            'seeds': data['seeds']
                        }
                        for method, data in coverage['evoxbench'][suite][n_obj][pid].items()
                    }

    return json.dumps(result, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Map seed coverage across experiments and algorithms.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--results_root', default='results/obj_ga',
                        help='Root results directory (default: results/obj_ga)')
    parser.add_argument('--format', choices=['table', 'csv', 'json'], default='table',
                        help='Output format (default: table)')
    parser.add_argument('--output', help='Output file path (default: stdout)')

    args = parser.parse_args()

    # Scan results
    coverage = scan_results(args.results_root)

    # Format output
    if args.format == 'table':
        output = format_table(coverage)
    elif args.format == 'csv':
        output = format_csv(coverage)
    elif args.format == 'json':
        output = format_json(coverage)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Seed coverage written to {args.output}", flush=True)
    else:
        print(f"\nSeed Coverage Report (scanning: {args.results_root})\n", flush=True)
        print(output, flush=True)


if __name__ == '__main__':
    main()
