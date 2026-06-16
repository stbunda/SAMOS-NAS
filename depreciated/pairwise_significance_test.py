"""Pairwise significance tests between algorithms.

Performs Wilcoxon rank-sum tests with optional Holm-Bonferroni correction
across multiple problems and algorithms.

**IMPORTANT**: For EvoXBench (C-10, IN-1K) problems, this script loads HV values
from the pre-computed cache (pareto_approx.pkl and hv_cache.pkl), which matches
the table in analyze_all_benchmarks.py. For WFG, it loads raw seed indicators.

Usage:
    python pairwise_significance_test.py \\
        --ref samos-xgb-c \\
        --compare samos-xgb ssa-xgb ssa-xgb-xgb \\
        --problems c10_1 c10_2 in1k_1 in1k_2 \\
        --alpha 0.05 \\
        --correction \\
        --max-seeds 20
"""

import argparse
import os
import pickle
import numpy as np
from scipy.stats import ranksums
from pathlib import Path

# Import from analysis module for proper EvoXBench HV loading
try:
    from analysis.convergence import build_pareto_approximation, recompute_final_indicators_seeds, compute_empirical_norm_bounds
    _ANALYSIS_AVAILABLE = True
except ImportError as e:
    _ANALYSIS_AVAILABLE = False
    _ANALYSIS_ERROR = str(e)


# ─── configuration ──────────────────────────────────────────────────────────

_WFG_FOLDER = {
    'random':    'random',
    'parego':    'parego',
    'mosmac':    'mosmac',
    'nsga2':     'nsga2',
    'gpsaf':     'gpsaf-default',
    'ssa-nsga2': 'ssa-nsga2-default',
    'ssa-nsga2-xgb': 'ssa-nsga2-xgb',
    'ssa-nsga2-xgb-cheap': 'ssa-nsga2-xgb-cheap',
    'samos-xgb': 'samos-xgb-i200-g20',
    'samos-xgb-c': 'samos-xgb-i200-g20',
}

_EVOX_FOLDER = {
    'random':    'random',
    'parego':    'parego',
    'mosmac':    'mosmac',
    'nsga2':     'nsga2',
    'gpsaf':     'gpsaf-default',
    'ssa-nsga2': 'ssa-nsga2',
    'ssa-nsga2-xgb': 'ssa-nsga2-xgb',
    'ssa-nsga2-xgb-cheap': 'ssa-nsga2-xgb-cheap',
    'samos-xgb': 'samos-xgb',
    'samos-xgb-c': 'samos-cheapreal',
}

_PROBLEM_MAP = {
    'wfg': {'suite': 'wfg', 'type': 'wfg', 'folder': _WFG_FOLDER},
    'c10': {'suite': 'c10mop', 'type': 'evox', 'folder': _EVOX_FOLDER},
    'in1k': {'suite': 'in1kmop', 'type': 'evox', 'folder': _EVOX_FOLDER},
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _parse_problem_id(problem_str: str) -> dict:
    """Parse problem string like 'c10_1', 'wfg4', 'in1k_3'."""
    parts = problem_str.lower().split('_')
    if len(parts) == 1:
        # Format: wfg4, wfg1, etc.
        suite_code = ''.join([c for c in parts[0] if not c.isdigit()])
        problem_num = ''.join([c for c in parts[0] if c.isdigit()])
    else:
        # Format: c10_1, in1k_2, etc.
        suite_code = parts[0]
        problem_num = parts[1]
    
    if suite_code not in _PROBLEM_MAP:
        raise ValueError(f"Unknown problem suite: {suite_code}. "
                        f"Must be one of {list(_PROBLEM_MAP.keys())}")
    
    meta = _PROBLEM_MAP[suite_code]
    return {
        'suite': meta['suite'],
        'type': meta['type'],
        'folder_map': meta['folder'],
        'num': int(problem_num),
        'display': problem_str,
    }


def _load_hv_from_seeds(seed_dir: str, max_seeds: int | None = None) -> np.ndarray | None:
    """Load HV values from seed pickle files in a directory."""
    if not os.path.isdir(seed_dir):
        return None
    
    hvs = []
    all_fnames = sorted(f for f in os.listdir(seed_dir) if f.endswith('.pkl'))
    if max_seeds is not None:
        all_fnames = all_fnames[:max_seeds]
    
    for fname in all_fnames:
        try:
            with open(os.path.join(seed_dir, fname), 'rb') as fh:
                data = pickle.load(fh)
        except Exception as e:
            print(f'  [WARN] Failed to load {fname}: {e}')
            continue
        
        indicators = data.get('indicators', [])
        if indicators:
            hvs.append(float(indicators[-1].get('hv', float('nan'))))
    
    if not hvs:
        return None
    return np.array(hvs)


def _load_hv_from_evox_cache(suite: str, problem_num: int, algorithm: str, 
                             n_gen: int = 60, pop_size: int = 20,
                             max_seeds: int | None = None) -> np.ndarray | None:
    """Load HV from cached indicators for EvoXBench (uses Pareto-approximation recomputed HV).
    
    Falls back to raw seed loading if cache cannot be loaded (e.g., numpy version mismatch).
    """
    folder_name = _EVOX_FOLDER.get(algorithm)
    if folder_name is None:
        return None
    
    cache_path = os.path.join(
        'results', 'evoxbench', suite, f'pid{problem_num}',
        f'B{n_gen * pop_size}_P{pop_size}', 'hv_cache.pkl'
    )
    
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache has the actual HV data
            if folder_name in cache_data:
                hv_arr = cache_data[folder_name].get('hv')
                if hv_arr is not None:
                    if max_seeds is not None:
                        hv_arr = hv_arr[:max_seeds]
                    return np.array(hv_arr)
        except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError):
            # Numpy version mismatch or other pickle incompatibility - silently fall back
            pass
        except Exception as e:
            print(f'  [DEBUG] Cache load failed ({type(e).__name__}), attempting recompute...')
    
    # Try to recompute using analysis module (matches analyze_all_benchmarks.py exactly)
    if _ANALYSIS_AVAILABLE:
        try:
            results_root = os.path.join(
                'results', 'evoxbench', suite, f'pid{problem_num}',
                f'B{n_gen * pop_size}_P{pop_size}'
            )
            
            # Build Pareto approximation from all algorithms
            all_evox_folders = list(_EVOX_FOLDER.values())
            approx = build_pareto_approximation(
                suite, problem_num, all_evox_folders, pop_size, n_gen,
                norm_bounds=None, results_root=results_root
            )
            
            if approx is not None:
                # Recompute final indicators for this specific algorithm
                seeds = recompute_final_indicators_seeds(
                    folder_name, results_root,
                    approx['ref_point'],
                    approx['pareto_approx'],
                    norm_bounds=None,
                )
                if seeds is not None:
                    hv_arr = seeds.get('hv')
                    if hv_arr is not None:
                        if max_seeds is not None:
                            hv_arr = hv_arr[:max_seeds]
                        return np.array(hv_arr)
        except Exception as e:
            print(f'  [DEBUG] Pareto recompute failed ({type(e).__name__}), falling back to raw seeds...')
    
    # Final fallback: raw seed loading (may give different HV than table)
    seed_dir = os.path.join(
        'results', 'evoxbench', suite, f'pid{problem_num}',
        f'B{n_gen * pop_size}_P{pop_size}', folder_name
    )
    return _load_hv_from_seeds(seed_dir, max_seeds=max_seeds)



def _get_wfg_hv(problem_num: int, algorithm: str, n_gen: int = 60, 
                pop_size: int = 20, max_seeds: int | None = None) -> np.ndarray | None:
    """Load HV for a WFG problem."""
    folder_name = _WFG_FOLDER.get(algorithm)
    if folder_name is None:
        return None
    
    problem_name = f'wfg{problem_num}'
    seed_dir = os.path.join(
        'results', 'pymoo_benchmark', '2_obj', problem_name,
        f'B{n_gen * pop_size}_P{pop_size}', folder_name
    )
    return _load_hv_from_seeds(seed_dir, max_seeds=max_seeds)


def _get_evox_hv(suite: str, problem_num: int, algorithm: str, 
                 n_gen: int = 60, pop_size: int = 20, 
                 max_seeds: int | None = None) -> np.ndarray | None:
    """Load HV for an EvoXBench problem (from cache when available)."""
    return _load_hv_from_evox_cache(suite, problem_num, algorithm, 
                                    n_gen=n_gen, pop_size=pop_size, 
                                    max_seeds=max_seeds)


def _load_hv(problem_meta: dict, algorithm: str, max_seeds: int | None = None) -> np.ndarray | None:
    """Load HV for any problem type."""
    if problem_meta['type'] == 'wfg':
        return _get_wfg_hv(problem_meta['num'], algorithm, max_seeds=max_seeds)
    else:  # evox
        return _get_evox_hv(problem_meta['suite'], problem_meta['num'], 
                           algorithm, max_seeds=max_seeds)


def _holm_correct(p_values: list[float], alpha: float) -> list[bool]:
    """Holm-Bonferroni correction. Returns a bool list: True = reject H0 (significant)."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    reject = [False] * n
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (n - rank):
            reject[idx] = True
        else:
            break
    return reject


def _raw_wilcoxon_p(ref_vals, other_vals) -> float:
    """Return the Wilcoxon rank-sum p-value, or 1.0 if data is insufficient."""
    if ref_vals is None or other_vals is None:
        return 1.0
    rv = np.asarray(ref_vals, dtype=float)
    ov = np.asarray(other_vals, dtype=float)
    rv = rv[np.isfinite(rv)]
    ov = ov[np.isfinite(ov)]
    if len(rv) < 3 or len(ov) < 3:
        return 1.0
    try:
        _, p = ranksums(rv, ov)
        return float(p)
    except Exception:
        return 1.0


def _direction_symbol(ref_arr, other_arr) -> str:
    """Return '+' or '-' based on mean direction."""
    rv = np.asarray(ref_arr, dtype=float)
    ov = np.asarray(other_arr, dtype=float)
    rv = rv[np.isfinite(rv)]
    ov = ov[np.isfinite(ov)]
    if len(rv) == 0 or len(ov) == 0:
        return '≈'
    return '+' if np.mean(ov) > np.mean(rv) else '-'


def pairwise_significance_test(
    ref_algorithm: str,
    list_of_algorithms: list[str],
    list_of_problems: list[str],
    significance: float = 0.05,
    correction: bool = True,
    max_seeds: int | None = None,
) -> dict:
    """Perform pairwise significance tests.
    
    Parameters
    ----------
    ref_algorithm
        Reference algorithm to compare against (e.g., 'samos-xgb-c').
    list_of_algorithms
        List of algorithms to compare (e.g., ['samos-xgb', 'ssa-xgb']).
    list_of_problems
        List of problems (e.g., ['c10_1', 'c10_2', 'in1k_1', 'wfg4']).
    significance
        Alpha level (default 0.05).
    correction
        Whether to apply Holm-Bonferroni correction (default True).
    max_seeds
        Max seeds to use per algorithm-problem combo (default None = all).
    
    Returns
    -------
    dict
        Nested dict: {problem: {algorithm: {'p': float, 'symbol': str,
                                            'ref_mean': float, 'ref_std': float,
                                            'alg_mean': float, 'alg_std': float,
                                            'n_seeds_ref': int, 'n_seeds_alg': int}}}
    """
    results = {}
    
    for problem_str in list_of_problems:
        print(f"Processing problem: {problem_str}")
        problem_meta = _parse_problem_id(problem_str)
        
        # Load reference HV
        ref_hv = _load_hv(problem_meta, ref_algorithm, max_seeds=max_seeds)
        if ref_hv is None:
            print(f"  ✗ No data for {ref_algorithm}")
            results[problem_str] = {}
            continue
        
        ref_hv_f = ref_hv[np.isfinite(ref_hv)]
        ref_mean = float(np.mean(ref_hv_f))
        ref_std = float(np.std(ref_hv_f))
        
        # Collect p-values for all comparisons (for Holm correction)
        p_values = []
        alg_data = {}
        for alg in list_of_algorithms:
            alg_hv = _load_hv(problem_meta, alg, max_seeds=max_seeds)
            if alg_hv is None:
                print(f"  ✗ No data for {alg}")
                p_values.append(1.0)
                alg_data[alg] = None
                continue
            
            alg_hv_f = alg_hv[np.isfinite(alg_hv)]
            alg_mean = float(np.mean(alg_hv_f))
            alg_std = float(np.std(alg_hv_f))
            p = _raw_wilcoxon_p(ref_hv, alg_hv)
            
            p_values.append(p)
            alg_data[alg] = {
                'hv': alg_hv,
                'mean': alg_mean,
                'std': alg_std,
                'n': len(alg_hv_f),
            }
        
        # Apply Holm correction if requested
        if correction and len(p_values) > 0:
            sig_flags = _holm_correct(p_values, significance)
        else:
            sig_flags = [p < significance for p in p_values]
        
        # Build result dict for this problem
        problem_results = {}
        for alg, p, sig in zip(list_of_algorithms, p_values, sig_flags):
            if alg_data[alg] is None:
                problem_results[alg] = {
                    'p': 1.0,
                    'p_raw': 1.0,
                    'significant': False,
                    'symbol': '–',
                    'ref_mean': ref_mean,
                    'ref_std': ref_std,
                    'ref_n': len(ref_hv_f),
                    'alg_mean': None,
                    'alg_std': None,
                    'alg_n': None,
                }
            else:
                symbol = _direction_symbol(ref_hv, alg_data[alg]['hv']) if sig else '≈'
                problem_results[alg] = {
                    'p': p,
                    'p_raw': p,
                    'significant': sig,
                    'symbol': symbol,
                    'ref_mean': ref_mean,
                    'ref_std': ref_std,
                    'ref_n': len(ref_hv_f),
                    'alg_mean': alg_data[alg]['mean'],
                    'alg_std': alg_data[alg]['std'],
                    'alg_n': alg_data[alg]['n'],
                }
        
        results[problem_str] = problem_results
        print(f"  ✓ Completed")
    
    return results


def print_results(results: dict, ref_algorithm: str, correction: bool = True,
                  alpha: float = 0.05) -> None:
    """Print results in a formatted table."""
    correction_str = f" (Holm-corrected, α={alpha})" if correction else f" (uncorrected, α={alpha})"
    print("\n" + "=" * 120)
    print(f"Pairwise Wilcoxon Rank-Sum Tests vs. {ref_algorithm}{correction_str}")
    print("=" * 120)
    
    # Header
    print(f"{'Problem':<15} {'Algorithm':<20} {'Symbol':<10} {'p-value':<12} "
          f"{'Ref Mean':<12} {'Alg Mean':<12} {'Diff %':<10}")
    print("-" * 120)
    
    for problem, algs in results.items():
        first_alg = True
        for alg, res in algs.items():
            if res['alg_mean'] is None:
                symbol = "NO DATA"
                p_str = "—"
                ref_mean_str = f"{res['ref_mean']:.4f}"
                alg_mean_str = "—"
                diff_str = "—"
            else:
                symbol = f"{res['symbol']}{' *' if res['significant'] else ''}"
                p_str = f"{res['p']:.6f}"
                ref_mean_str = f"{res['ref_mean']:.4f}"
                alg_mean_str = f"{res['alg_mean']:.4f}"
                diff_pct = ((res['alg_mean'] - res['ref_mean']) / res['ref_mean'] * 100) if res['ref_mean'] != 0 else 0
                diff_str = f"{diff_pct:+.1f}%"
            
            prob_str = problem if first_alg else ""
            print(f"{prob_str:<15} {alg:<20} {symbol:<10} {p_str:<12} "
                  f"{ref_mean_str:<12} {alg_mean_str:<12} {diff_str:<10}")
            first_alg = False
    
    print("=" * 120)
    print("Symbol: + = significantly better, - = significantly worse, ≈ = not significant")
    print("       * = statistically significant after correction")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pairwise significance tests between algorithms'
    )
    parser.add_argument('--ref', required=True, help='Reference algorithm')
    parser.add_argument('--compare', nargs='+', required=True, 
                       help='List of algorithms to compare')
    parser.add_argument('--problems', nargs='+', required=True,
                       help='List of problems (e.g., c10_1 c10_2 in1k_1 wfg4)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default 0.05)')
    parser.add_argument('--correction', action='store_true', default=True,
                       help='Apply Holm-Bonferroni correction (default True)')
    parser.add_argument('--no-correction', dest='correction', action='store_false',
                       help='Disable Holm-Bonferroni correction')
    parser.add_argument('--max-seeds', type=int, default=None,
                       help='Max seeds to use per problem (default None = all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file (optional)')
    
    args = parser.parse_args()
    
    # Run tests
    results = pairwise_significance_test(
        ref_algorithm=args.ref,
        list_of_algorithms=args.compare,
        list_of_problems=args.problems,
        significance=args.alpha,
        correction=args.correction,
        max_seeds=args.max_seeds,
    )
    
    # Print results
    print_results(results, args.ref, args.correction, args.alpha)
    
    # Optionally write CSV
    if args.output:
        import csv
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Problem', 'Algorithm', 'p-value', 'Significant', 'Symbol',
                           'Ref Mean', 'Ref Std', 'Alg Mean', 'Alg Std', 'Diff %'])
            for problem, algs in results.items():
                for alg, res in algs.items():
                    if res['alg_mean'] is not None:
                        diff_pct = ((res['alg_mean'] - res['ref_mean']) / res['ref_mean'] * 100) \
                                  if res['ref_mean'] != 0 else 0
                    else:
                        diff_pct = None
                    writer.writerow([
                        problem, alg, f"{res['p']:.6f}", res['significant'],
                        res['symbol'], f"{res['ref_mean']:.6f}", f"{res['ref_std']:.6f}",
                        f"{res['alg_mean']:.6f}" if res['alg_mean'] else '',
                        f"{res['alg_std']:.6f}" if res['alg_std'] else '',
                        f"{diff_pct:+.2f}%" if diff_pct else '',
                    ])
        print(f"\nResults written to {args.output}")
