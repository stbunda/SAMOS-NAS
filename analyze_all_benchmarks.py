"""analyze_all_benchmarks.py — Combined HV LaTeX table for WFG / C-10 / IN-1K.

Produces a single booktabs LaTeX table with three benchmark sections:
  • WFG 1–9  (pymoo, analytically-fixed reference point)
  • C-10 MOP 1–9  (EvoXBench, shared combined Pareto approximation)
  • IN-1K MOP 1–9  (EvoXBench, shared combined Pareto approximation)

Columns (in order):
  Problem | Random | ParEGO | MO-SMAC | NSGA-II | GPSAF | SAMOS (XGBoost)

Each cell shows mean ± std HV across seeds.  The best value per row is
bolded.  All non-reference columns carry a Wilcoxon rank-sum significance
marker (p < 0.05) vs. SAMOS (XGBoost):
  $(+)$  significantly better
  $(-)$  significantly worse
  $(\approx)$  no significant difference

Run example
-----------
python analyze_all_benchmarks.py

python analyze_all_benchmarks.py \\
    --out results/all_benchmarks_hv_table.tex \\
    --wfg_n_gen 60 --wfg_pop 20 \\
    --evox_n_gen 60 --evox_pop 20
"""

import argparse
import os
import pickle
import tempfile

import numpy as np
from scipy.stats import ranksums
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from analysis.convergence import build_pareto_approximation, recompute_final_indicators_seeds, compute_empirical_norm_bounds
from problem.evoxbench.benchmark_meta import BENCHMARK_META

# Search spaces backed by a full tabular database (no surrogate predictor needed).
# Problems using any other search space are marked with \dagger in tables.
_TABULAR_SEARCH_SPACES: frozenset = frozenset({'NB101', 'NATS', 'NB201'})

# ─── canonical method list ────────────────────────────────────────────────────

_METHODS = ['random', 'parego', 'mosmac', 'nsga2', 'gpsaf',
            'ssa-nsga2', 'ssa-nsga2-xgb', 'ssa-nsga2-xgb-cheap',
            'samos-xgb', 'samos-xgb-c']

_COLUMN_LABELS = {
    'random':    'Random',
    'parego':    'ParEGO',
    'mosmac':    'MO-SMAC',
    'nsga2':     'NSGA-II',
    'gpsaf':     'GPSAF',
    'ssa-nsga2': 'SSA-RBF',
    'ssa-nsga2-xgb': 'SSA-XGB',
    'ssa-nsga2-xgb-cheap': 'SSA-XGB-C',
    'samos-xgb': 'SAMOS',
    'samos-xgb-c': 'SAMOS-C',
}

# Filesystem directory name for WFG results
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
    'samos-xgb-c': 'samos-xgb-cheap',
}

# Filesystem directory name for EvoXBench results
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

_WILCOXON_REF    = 'samos-xgb'
_WILCOXON_ALPHA  = 0.05
_EXPECTED_SEEDS  = 30


# ─── path helpers ─────────────────────────────────────────────────────────────

def _results_root_wfg(problem: str, experiment_name: str, n_gen: int, pop_size: int) -> str:
    return os.path.join('results', experiment_name, problem,
                        f'B{n_gen * pop_size}_P{pop_size}')


def _results_root_evox(suite: str, pid: int, n_gen: int, pop_size: int,
                       root: str = 'results/evoxbench') -> str:
    return os.path.join(root, suite, f'pid{pid}', f'B{n_gen * pop_size}_P{pop_size}')


def _detect_available_evox_methods(
    root: str,
    suites: list,
    pids: list,
    n_gen: int,
    pop_size: int,
) -> list:
    """Return the ordered subset of _METHODS that have at least one seed file under *root*."""
    available: set = set()
    for suite in suites:
        for pid in pids:
            pid_root = _results_root_evox(suite, pid, n_gen, pop_size, root=root)
            for key in _METHODS:
                if key in available:
                    continue
                seed_dir = os.path.join(pid_root, _EVOX_FOLDER[key])
                if os.path.isdir(seed_dir) and any(
                    f.endswith('.pkl') for f in os.listdir(seed_dir)
                ):
                    available.add(key)
    detected = [k for k in _METHODS if k in available]
    print(f'  [evox_methods] Detected in {root}: {detected}')
    return detected


# ─── EvoXBench HV cache ──────────────────────────────────────────────────────
#
# Stores per-seed HV arrays (keyed by method folder name) next to the Pareto
# approximation cache.  Two-level staleness check:
#   1. If pareto_approx.pkl has been modified, all entries are invalid.
#   2. Per-method: if any contributing seed file has been modified, that
#      method's entry is recomputed.

def _indicators_cache_path(root: str) -> str:
    return os.path.join(root, 'indicators_cache.pkl')


def _load_indicators_cache(root: str, approx_mtime: float) -> dict:
    """Load valid entries from the on-disk indicators cache.

    Returns a dict ``{folder: {'hv': np.ndarray, 'igd_plus': np.ndarray}}``
    containing only entries that are still valid (Pareto approximation
    unchanged, no seed files modified).  Entries missing ``'igd_plus'`` (old
    hv-only format) are treated as stale and will be recomputed.
    """
    path = _indicators_cache_path(root)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'rb') as fh:
            cache = pickle.load(fh)
    except Exception as exc:
        print(f'  [indicators_cache] WARN: could not read {path}: {exc}')
        return {}

    # Level-1: Pareto approximation must not have changed.
    stored_approx_mt = cache.get('approx_mtime', -1)
    if abs(stored_approx_mt - approx_mtime) > 1e-3:
        print(f'  [indicators_cache] Stale: Pareto approximation has changed '
              f'(stored={stored_approx_mt:.6f}, current={approx_mtime:.6f}); '
              f'rebuilding all.')
        return {}

    valid: dict = {}
    for folder, entry in cache.get('methods', {}).items():
        stale = False
        stored_mtimes = entry.get('source_mtimes', {})
        for fpath, stored_mt in stored_mtimes.items():
            if not os.path.isfile(fpath):
                stale = True
                break
            if os.path.getmtime(fpath) > stored_mt + 1e-3:
                stale = True
                break
        if not stale:
            # Also check for new seed files not in the stored snapshot.
            seed_dir = os.path.join(root, folder)
            current_files = set(_seed_mtimes(seed_dir).keys())
            if current_files != set(stored_mtimes.keys()):
                stale = True
        if stale:
            print(f'  [indicators_cache] Stale entry for {folder}; will recompute.')
            continue
        # Old cache format stored only 'hv' — treat as stale so igd_plus is computed.
        if 'igd_plus' not in entry:
            print(f'  [indicators_cache] Entry for {folder} missing igd_plus; will recompute.')
            continue
        valid[folder] = {'hv': entry['hv'], 'igd_plus': entry['igd_plus']}

    n_hit = len(valid)
    if n_hit:
        print(f'  [indicators_cache] Cache hit: {path} ({n_hit} method(s) valid)')
    return valid


def _save_indicators_cache(root: str, approx_mtime: float, methods_data: dict) -> None:
    """Atomically write the indicators cache.

    Parameters
    ----------
    root
        Budget directory (same folder that holds ``pareto_approx.pkl``).
    approx_mtime
        ``os.path.getmtime`` of ``pareto_approx.pkl`` at build time.
    methods_data
        ``{folder: {'hv': np.ndarray, 'igd_plus': np.ndarray,
                    'source_mtimes': {path: mtime}}}``
    """
    path  = _indicators_cache_path(root)
    cache = {'approx_mtime': approx_mtime, 'methods': methods_data}
    dir_  = os.path.dirname(path) or '.'
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(f'  [indicators_cache] Saved -> {path}')


def _seed_mtimes(seed_dir: str) -> dict:
    """Return ``{abs_path: mtime}`` for every ``.pkl`` file in *seed_dir*."""
    if not os.path.isdir(seed_dir):
        return {}
    result = {}
    for fname in sorted(os.listdir(seed_dir)):
        if fname.endswith('.pkl'):
            p = os.path.abspath(os.path.join(seed_dir, fname))
            result[p] = os.path.getmtime(p)
    return result


# ─── data-loading helpers ─────────────────────────────────────────────────────

def _load_seeds_from_indicators(seed_dir: str, max_seeds: int | None = None) -> dict | None:
    """Load per-seed HV and IGD+ from stored ``indicators[-1]`` (WFG results).

    Each ``seed_*.pkl`` is expected to contain a list of per-generation
    indicator dicts under the ``'indicators'`` key.  Only the last
    generation values are used.

    Parameters
    ----------
    max_seeds
        When set, only the first *max_seeds* seed files (in sorted order)
        are loaded.  ``None`` (default) loads all available seeds.

    Returns ``{'hv': np.ndarray, 'igd_plus': np.ndarray}`` or ``None`` if
    the directory is absent or no valid seeds are found.
    """
    if not os.path.isdir(seed_dir):
        return None
    hvs, igds = [], []
    all_fnames = sorted(f for f in os.listdir(seed_dir) if f.endswith('.pkl'))
    if max_seeds is not None:
        all_fnames = all_fnames[:max_seeds]
    for fname in all_fnames:
        try:
            with open(os.path.join(seed_dir, fname), 'rb') as fh:
                data = pickle.load(fh)
        except Exception as exc:
            print(f'  [WARN] Failed to load {fname}: {exc}')
            continue
        indicators = data.get('indicators', [])
        if not indicators:
            continue
        hvs.append(float(indicators[-1].get('hv',       float('nan'))))
        igds.append(float(indicators[-1].get('igd_plus', float('nan'))))
    if not hvs:
        return None
    return {'hv': np.array(hvs), 'igd_plus': np.array(igds)}


def _collect_wfg_data(
    problems: list,
    experiment_name: str,
    n_gen: int,
    pop_size: int,
    methods: list | None = None,
    max_seeds: int | None = None,
) -> dict:
    """Return ``{problem: {method_key: ndarray | None}}`` for all WFG problems.

    Parameters
    ----------
    problems
        List of problem names, e.g. ``['wfg1', ..., 'wfg9']``.
    experiment_name
        Sub-tree relative to ``results/``, e.g. ``'pymoo_benchmark/2_obj'``.
    n_gen, pop_size
        Budget parameters used to locate the ``B{budget}_P{pop_size}`` folder.
    max_seeds
        When set, only the first *max_seeds* seed files are used.
    """
    result: dict = {}
    _m = methods if methods is not None else _METHODS
    for problem in problems:
        root      = _results_root_wfg(problem, experiment_name, n_gen, pop_size)
        prob_data = {}
        for key in _m:
            folder  = _WFG_FOLDER[key]
            metrics = _load_seeds_from_indicators(os.path.join(root, folder),
                                                  max_seeds=max_seeds)
            prob_data[key] = metrics
            if metrics is not None:
                hv_f = metrics['hv'][np.isfinite(metrics['hv'])]
                mean_str = f'{hv_f.mean():.4f}' if len(hv_f) else 'NaN'
                print(f'  [WFG] {problem}/{key}: {len(metrics["hv"])} seeds, '
                      f'mean HV = {mean_str}')
            else:
                print(f'  [WFG] {problem}/{key}: no data')
        result[problem] = prob_data
    return result



def _collect_evox_data(
    suite: str,
    pids: list,
    n_gen: int,
    pop_size: int,
    force: bool = False,
    evox_root: str = 'results/evoxbench',
    no_norm: bool = False,
    methods: list | None = None,
    max_seeds: int | None = None,
) -> dict:
    """Return ``{pid: {method_key: ndarray | None}}`` for an EvoXBench suite.

    HV is recomputed for each seed against a combined Pareto approximation
    built from *all* method folders present in the budget directory (using
    ``build_pareto_approximation`` from ``analysis.convergence``), which
    ensures a fair cross-method comparison.  Results are cached in
    ``hv_cache.pkl`` next to ``pareto_approx.pkl`` and reused on subsequent
    runs unless the approximation or the underlying seed files have changed.

    Parameters
    ----------
    suite
        EvoXBench suite name, e.g. ``'c10mop'`` or ``'in1kmop'``.
    pids
        List of problem IDs, e.g. ``[1, 2, ..., 9]``.
    n_gen, pop_size
        Budget parameters.
    force
        Ignore the on-disk cache and recompute everything.
    max_seeds
        When set, only the first *max_seeds* seeds (sorted order) are used
        for statistics.  The on-disk cache is read in full and then truncated.
    """
    result: dict = {}
    methods_list = methods if methods is not None else _METHODS
    evox_folders = [_EVOX_FOLDER[k] for k in methods_list]
    suite_tag    = suite.upper()

    for pid in pids:
        root          = _results_root_evox(suite, pid, n_gen, pop_size, root=evox_root)
        norm_bounds   = compute_empirical_norm_bounds(evox_folders, root) if no_norm else None
        approx        = build_pareto_approximation(
            suite, pid, evox_folders, pop_size, n_gen,
            norm_bounds=norm_bounds, results_root=root,
        )

        # ── resolve cache ──────────────────────────────────────────────────
        approx_path = os.path.join(root, 'pareto_approx.pkl')
        approx_mt   = os.path.getmtime(approx_path) if os.path.isfile(approx_path) else 0.0

        cached: dict = {} if force else _load_indicators_cache(root, approx_mt)
        updated: dict = {}   # accumulates entries to (re)write to cache

        pid_data: dict = {}
        for key in methods_list:
            folder = _EVOX_FOLDER[key]

            # ── cache hit ─────────────────────────────────────────────────
            if folder in cached:
                metrics        = cached[folder]   # {'hv': arr, 'igd_plus': arr}
                if max_seeds is not None:
                    metrics = {k: v[:max_seeds] for k, v in metrics.items()}
                pid_data[key]  = metrics
                hv_f           = metrics['hv'][np.isfinite(metrics['hv'])]
                mean_str       = f'{hv_f.mean():.4f}' if len(hv_f) else 'NaN'
                print(f'  [{suite_tag}] pid{pid}/{key}: {len(metrics["hv"])} seeds, '
                      f'mean HV = {mean_str} (cached)')
                seed_dir = os.path.join(root, folder)
                updated[folder] = {
                    'hv':            cached[folder]['hv'],
                    'igd_plus':      cached[folder]['igd_plus'],
                    'source_mtimes': _seed_mtimes(seed_dir),
                }
                continue

            # ── cache miss: compute ───────────────────────────────────────
            if approx is not None:
                seeds   = recompute_final_indicators_seeds(
                    folder, root,
                    approx['ref_point'],
                    approx['pareto_approx'],
                    norm_bounds=norm_bounds,
                )
                metrics = seeds   # {'hv': arr, 'igd_plus': arr} or None
            else:
                metrics = _load_seeds_from_indicators(os.path.join(root, folder))

            if metrics is not None and max_seeds is not None:
                metrics = {k: v[:max_seeds] for k, v in metrics.items()}
            pid_data[key] = metrics
            if metrics is not None:
                hv_f     = metrics['hv'][np.isfinite(metrics['hv'])]
                mean_str = f'{hv_f.mean():.4f}' if len(hv_f) else 'NaN'
                print(f'  [{suite_tag}] pid{pid}/{key}: {len(metrics["hv"])} seeds, '
                      f'mean HV = {mean_str}')
                seed_dir = os.path.join(root, folder)
                updated[folder] = {
                    'hv':            metrics['hv'],
                    'igd_plus':      metrics['igd_plus'],
                    'source_mtimes': _seed_mtimes(seed_dir),
                }
            else:
                print(f'  [{suite_tag}] pid{pid}/{key}: no data')

        # ── persist updated cache ─────────────────────────────────────────
        if updated:
            _save_indicators_cache(root, approx_mt, updated)

        result[pid] = pid_data
    return result


# ─── statistical helpers ──────────────────────────────────────────────────────

def _wilcoxon_marker(ref_vals, other_vals, higher_is_better: bool = True) -> str:
    """Wilcoxon rank-sum significance marker vs. the reference method.

    Returns ``$(+)$`` when *other_vals* is significantly better than
    *ref_vals*, ``$(-)$`` when significantly worse, or ``$(\\approx)$``
    when the difference is not significant (p >= _WILCOXON_ALPHA).
    An empty string is returned when either array is too small or
    consists entirely of non-finite values.
    """
    if ref_vals is None or other_vals is None:
        return ''
    rv = np.asarray(ref_vals, dtype=float)
    ov = np.asarray(other_vals, dtype=float)
    rv = rv[np.isfinite(rv)]
    ov = ov[np.isfinite(ov)]
    if len(rv) < 3 or len(ov) < 3:
        return ''
    try:
        _, p = ranksums(rv, ov)
    except Exception:
        return r'$(\approx)$'
    if p >= _WILCOXON_ALPHA:
        return r'$^{\approx}$'
    ref_med   = np.mean(rv)
    other_med = np.mean(ov)
    if higher_is_better:
        return r'$^{+}$' if other_med > ref_med else r'$^{-}$'
    else:
        return r'$^{+}$' if other_med < ref_med else r'$^{-}$'


def _fmt(mean: float, std: float, bold: bool, marker: str = '') -> str:
    """Format a ``mean_{std}`` cell, optionally bolding and appending a marker."""
    s    = f'{mean:.4f}$_{{\\text{{{std:.4f}}}}}$'
    cell = f'\\textbf{{{s}}}' if bold else s
    return f'{cell}{marker}'


# ─── section renderer ─────────────────────────────────────────────────────────

def _render_section(
    lines: list,
    section_header: str,
    row_labels: list,
    suite_data: dict,
    row_keys: list,
    metric: str = 'hv',
    higher_is_better: bool = True,
    methods: list | None = None,
    max_seeds: int | None = None,
) -> None:
    """Append one benchmark section (a header row + data rows) to *lines*.

    Parameters
    ----------
    lines
        Accumulator list for LaTeX source lines.
    section_header
        Human-readable section title, e.g. ``r'WFG ($m=2$, $d=12$)'``.
    row_labels
        Display labels for each problem row, e.g. ``['WFG/MOP1', ...]``.
    suite_data
        ``{row_key: {method_key: {'hv': ndarray, 'igd_plus': ndarray} | None}}``
        mapping from :func:`_collect_wfg_data` or :func:`_collect_evox_data`.
    row_keys
        Keys into *suite_data* (problems or pids), in display order.
    metric
        Which indicator to display: ``'hv'`` or ``'igd_plus'``.
    higher_is_better
        ``True`` for HV (bold = max), ``False`` for IGD+ (bold = min).
    """
    _m = methods if methods is not None else _METHODS
    n_cols = 1 + len(_m)
    lines.append(r'\midrule')
    lines.append(
        r'\multicolumn{' + str(n_cols) + r'}{l}{\small\textit{' + section_header + r'}} \\'
    )
    lines.append(r'\midrule')

    for row_key, row_label in zip(row_keys, row_labels):
        prob_data = suite_data.get(row_key, {})

        # Extract per-method arrays for the requested metric
        avail: dict = {}
        for k, v in prob_data.items():
            if v is None:
                continue
            arr = v[metric] if isinstance(v, dict) else v
            if len(arr) > 0 and np.isfinite(arr).any():
                avail[k] = arr

        # Identify best method (only among displayed methods)
        avail_m = {k: avail[k] for k in _m if k in avail}
        if higher_is_better:
            best_key = max(avail_m, key=lambda k: float(np.nanmean(avail_m[k]))) if avail_m else None
        else:
            best_key = min(avail_m, key=lambda k: float(np.nanmean(avail_m[k]))) if avail_m else None
        ref_arr = avail.get(_WILCOXON_REF)

        cells = [row_label]
        for key in _m:
            arr = avail.get(key)
            if arr is None:
                cells.append('--')
                continue
            arr_f  = arr[np.isfinite(arr)]
            mean   = float(np.mean(arr_f))
            std    = float(np.std(arr_f))
            bold   = (key == best_key)
            marker = '' if key == _WILCOXON_REF else _wilcoxon_marker(ref_arr, arr, higher_is_better)
            _seed_threshold = max_seeds if max_seeds is not None else _EXPECTED_SEEDS
            if len(arr) < _seed_threshold:
                marker += r'$^*$'
            cells.append(_fmt(mean, std, bold, marker))

        lines.append(' & '.join(cells) + r' \\')


# ─── table generation ─────────────────────────────────────────────────────────

def generate_combined_hv_table(
    out_path: str,
    wfg_data: dict,
    c10_data: dict,
    in1k_data: dict,
    wfg_problems: list,
    pids: list,
    n_obj_wfg: int = 2,
    n_var_wfg: int = 12,
    metric: str = 'hv',
    methods: list | None = None,
    caption_note: str = '',
    include_wfg: bool = True,
    c10_pids: list | None = None,
    max_seeds: int | None = None,
    rotate: bool = False,
) -> None:
    """Write a combined HV or IGD+ booktabs LaTeX table to *out_path*.

    Parameters
    ----------
    out_path
        Destination file path for the ``.tex`` output.
    wfg_data, c10_data, in1k_data
        Per-suite data dicts from :func:`_collect_wfg_data` /
        :func:`_collect_evox_data`.
    wfg_problems
        WFG problem name list in display order.
    pids
        EvoXBench problem-ID list used for IN-1K MOP rows (and C-10 MOP rows
        when *c10_pids* is not given).
    c10_pids
        Override the problem-ID list for the C-10 MOP section.  When ``None``
        the same *pids* list is used (original behaviour).
    n_obj_wfg, n_var_wfg
        WFG dimensionality metadata used in the caption.
    rotate
        When ``True`` wrap the table in a ``sidewaystable`` environment
        (requires ``\\usepackage{rotating}`` in the document preamble) so
        that it is rotated 90° to fit on a portrait page.
    """
    _METRIC_META = {
        'hv':       ('hypervolume',                    'tab:combined_hv',       True),
        'igd_plus': (r'IGD\textsuperscript{+}',        'tab:combined_igd_plus', False),
    }
    metric_name, label_key, higher_is_better = _METRIC_META[metric]
    _c10_pids = c10_pids if c10_pids is not None else pids
    _m       = methods if methods is not None else _METHODS
    n_cols   = 1 + len(_m)
    col_spec = 'l @{\\hspace{2em}} ' + ' '.join(['r'] * len(_m))

    col_header = ' & '.join(
        [r'\textbf{Problem}']
        + [r'\textbf{' + _COLUMN_LABELS[k] + r'}' for k in _m]
    )

    _seed_threshold = max_seeds if max_seeds is not None else _EXPECTED_SEEDS
    _seeds_str = (
        rf'{max_seeds}~seeds' if max_seeds is not None else r'seeds'
    )

    table_env = 'sidewaystable' if rotate else 'table*'
    lines = [
        r'\begin{' + table_env + r'}[t]',
        r'\centering',
        (
            r'\caption{Final ' + metric_name
            + r' (mean\,\textpm\,std over ' + _seeds_str + r') after 1200 evaluations on '
            + (
                r'WFG\,1\textendash{}9 , C-10\,MOP\,1\textendash{}9, and IN-1K\,MOP\,1\textendash{}9. '
                if include_wfg else
                r'C-10\,MOP\,8\textendash{}9 and IN-1K\,MOP\,1\textendash{}9. '
            )
            + r'\textbf{Bold}: best per problem. '
            r'Wilcoxon rank-sum vs. SAMOS, $p{<}0.05$: '
            r'$^{+}$\,better, $^{-}$\,worse, $^{\approx}$\,not significant.'
            + (' ' + caption_note if caption_note else '')
            + r'}'
        ),
        r'\label{' + label_key + r'}',
    ]
    if not rotate:
        lines += [
            r'\resizebox{\linewidth}{!}{%',
            r'\setlength\tabcolsep{5pt}%',
        ]
    lines += [
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        col_header + r' \\',
    ]

    # Problem-type markers appended to row labels
    _MK_SYNTHETIC  = r'$^{\diamond}$'   # WFG — analytically-defined synthetic
    _MK_TABULAR    = r'$^{\square}$'    # NB101 / NATS / NB201 — full tabular lookup
    _MK_SURROGATE  = r'$^{\dagger}$'     # DARTS / ResNet-50D / … — surrogate predictor

    def _evox_marker(suite: str, pid: int) -> str:
        ss = BENCHMARK_META.get(suite, {}).get(pid, {}).get('search_space', '')
        return _MK_TABULAR if ss in _TABULAR_SEARCH_SPACES else _MK_SURROGATE

    # WFG section
    if include_wfg:
        _render_section(
            lines,
            section_header=f'WFG (${n_obj_wfg}$ objectives, ${n_var_wfg}$ variables)',
            row_labels=[
                f'{i}{_MK_SYNTHETIC}'
                for i in range(1, len(wfg_problems) + 1)
            ],
            suite_data=wfg_data,
            row_keys=wfg_problems,
            metric=metric,
            higher_is_better=higher_is_better,
            methods=_m,
            max_seeds=max_seeds,
        )

    # C-10 MOP section
    _render_section(
        lines,
        section_header='C-10 MOP (M objectives, D variables)',
        row_labels=[
            '{}{} ({}, {})'.format(
                pid,
                _evox_marker('c10mop', pid),
                BENCHMARK_META.get('c10mop', {}).get(pid, {}).get('n_obj', '?'),
                BENCHMARK_META.get('c10mop', {}).get(pid, {}).get('n_var', '?'),
            )
            for pid in _c10_pids
        ],
        suite_data=c10_data,
        row_keys=_c10_pids,
        metric=metric,
        higher_is_better=higher_is_better,
        methods=_m,
        max_seeds=max_seeds,
    )

    # IN-1K MOP section
    _render_section(
        lines,
        section_header='IN-1K MOP (M objectives, D variables)',
        row_labels=[
            '{}{} ({}, {})'.format(
                pid,
                _evox_marker('in1kmop', pid),
                BENCHMARK_META.get('in1kmop', {}).get(pid, {}).get('n_obj', '?'),
                BENCHMARK_META.get('in1kmop', {}).get(pid, {}).get('n_var', '?'),
            )
            for pid in pids
        ],
        suite_data=in1k_data,
        row_keys=pids,
        metric=metric,
        higher_is_better=higher_is_better,
        methods=_m,
        max_seeds=max_seeds,
    )

    # Footnote row (before \bottomrule, inside the tabular body)
    lines += [
        r'\midrule',
        r'\multicolumn{' + str(n_cols) + r'}{l}{\footnotesize $^{+}$: significantly better than SAMOS; $^{-}$: significantly worse; $^{\approx}$: no significant difference (Wilcoxon rank-sum, $p{<}0.05$).} \\',
        r'\multicolumn{' + str(n_cols) + r'}{l}{\footnotesize $^*$: fewer than ' + str(_seed_threshold) + r' seeds evaluated.} \\',
        r'\multicolumn{' + str(n_cols) + r'}{l}{\footnotesize $^{\diamond}$: synthetic benchmark (WFG); $^{\square}$: tabular NAS benchmark (NB101/NATS/NB201); $^{\dagger}$: surrogate NAS benchmark (DARTS/ResNet-50D/\ldots).} \\',
        # r'\bottomrule',
        r'\end{tabular}',
    ]
    if not rotate:
        lines.append(r'}')
    lines.append(r'\end{' + table_env + r'}')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nLaTeX table saved -> {out_path}')


# ─── rank-pareto helpers ──────────────────────────────────────────────────────

def _load_true_pf_evox(suite: str, pids: list) -> dict:
    """Load and normalize the benchmark's pre-computed true Pareto front.

    Uses the evoxbench benchmark object's ``pareto_front`` attribute and
    normalizes it to the same ``[0, 1]`` scale as ``test_obj_archive`` entries.

    Returns
    -------
    ``{pid: np.ndarray | None}``
        Normalized Pareto front per problem, or ``None`` when unavailable.
    """
    from problem.evoxbench.utils import get_benchmark
    result: dict = {}
    for pid in pids:
        try:
            bm     = get_benchmark(suite, pid)
            pf_raw = bm.pareto_front
            if pf_raw is None or len(pf_raw) == 0:
                print(
                    f'  [rank_pareto] {suite}/pid{pid}: '
                    f'no true PF available, using combined pool only.'
                )
                result[pid] = None
                continue
            pf_norm = bm.normalize(np.asarray(pf_raw, dtype=float))
            pf_norm = np.where(np.isfinite(pf_norm), pf_norm, 1.0)
            result[pid] = pf_norm
            print(
                f'  [rank_pareto] {suite}/pid{pid}: '
                f'true PF loaded ({len(pf_norm)} pts)'
            )
        except Exception as exc:
            print(
                f'  [rank_pareto] {suite}/pid{pid}: '
                f'failed to load true PF: {exc}'
            )
            result[pid] = None
    return result


def _load_final_archives_evox(
    suite: str,
    pids: list,
    n_gen: int,
    pop_size: int,
    root: str = 'results/evoxbench',
) -> dict:
    """Load per-seed final test-objective archives for every (pid, method).

    For each ``seed_*.pkl`` inside
    ``results/evoxbench/<suite>/pid<pid>/B<budget>_P<pop>/\<method_folder>/``
    the function reads ``data['test_obj_archive'][-1]`` — the final-generation
    cumulative non-dominated archive in benchmark-normalized true-eval space.

    Returns
    -------
    ``{pid: {method_key: list[np.ndarray]}}``
        One ``np.ndarray`` per seed (shape ``(n_pts, n_obj)``).
        Missing or empty archives are silently skipped.
    """
    result: dict = {}
    for pid in pids:
        pid_root = _results_root_evox(suite, pid, n_gen, pop_size, root=root)
        pid_data: dict = {}
        for key in _METHODS:
            folder   = _EVOX_FOLDER[key]
            seed_dir = os.path.join(pid_root, folder)
            archives: list = []
            if not os.path.isdir(seed_dir):
                pid_data[key] = archives
                continue
            for fname in sorted(os.listdir(seed_dir)):
                if not fname.endswith('.pkl'):
                    continue
                fpath = os.path.join(seed_dir, fname)
                try:
                    with open(fpath, 'rb') as fh:
                        data = pickle.load(fh)
                except Exception as exc:
                    print(f'  [rank_pareto] WARN: failed to load {fpath}: {exc}')
                    continue
                test_arch = data.get('test_obj_archive', [])
                if not test_arch:
                    continue
                F = test_arch[-1]
                if F is None or (hasattr(F, '__len__') and len(F) == 0):
                    continue
                F = np.asarray(F, dtype=float)
                if F.ndim != 2 or F.shape[0] == 0:
                    continue
                finite = np.isfinite(F).all(axis=1)
                F = F[finite]
                if len(F) == 0:
                    continue
                archives.append(F)
            pid_data[key] = archives
        result[pid] = pid_data
    return result


def _assign_nd_layers(pool_F: np.ndarray, n_layers: int = 5) -> np.ndarray:
    """Assign a non-dominated rank to every point in *pool_F*.

    Parameters
    ----------
    pool_F : np.ndarray, shape ``(n_pts, n_obj)``
        Minimization objectives for the combined pool of all methods and seeds.
    n_layers : int
        Points in rank > *n_layers* are clamped to ``n_layers + 1``.

    Returns
    -------
    np.ndarray of int, shape ``(n_pts,)``
        Layer index per point, 1-based (layer 1 = Pareto front).
    """
    nds    = NonDominatedSorting()
    fronts = nds.do(pool_F, n_stop_if_ranked=None)
    layer_ids = np.full(len(pool_F), n_layers + 1, dtype=int)
    for rank, front in enumerate(fronts, start=1):
        layer = min(rank, n_layers + 1)
        layer_ids[front] = layer
    return layer_ids


def _compute_rank_pareto_counts(
    suite: str,
    pids: list,
    n_gen: int,
    pop_size: int,
    n_layers: int = 5,
    reference_fronts: dict | None = None,
    root: str = 'results/evoxbench',
) -> dict:
    """For each (pid, method, seed) count how many points fall in each ND layer.

    The combined pool is built by stacking ``test_obj_archive[-1]`` from *all*
    methods and *all* seeds for a given ``pid``.  When *reference_fronts* is
    provided (a ``{pid: np.ndarray}`` mapping of pre-normalized true Pareto
    fronts), those points are prepended to the pool so that NDS layer 1
    corresponds to genuinely Pareto-optimal solutions rather than the
    approximation set.

    Parameters
    ----------
    suite : str
        EvoXBench suite name, e.g. ``'in1kmop'``.
    pids : list[int]
        Problem IDs to process.
    n_gen, pop_size : int
        Budget parameters used to locate result folders.
    n_layers : int
        Number of non-dominated layers to track explicitly.
    reference_fronts : dict or None
        Optional ``{pid: np.ndarray(n_pts, n_obj)}`` of pre-normalized true
        Pareto fronts to include in the pool.  Points tagged with the
        ``'__ref__'`` sentinel are excluded from per-method counts.

    Returns
    -------
    ``{pid: {method_key: {'counts': np.ndarray(n_seeds, n_layers), 'n_pts': np.ndarray(n_seeds)}}}``
        ``counts[s, l]`` = number of points seed *s* contributed to layer *l+1*.
    """
    _REF_TAG = '__ref__'
    archives = _load_final_archives_evox(suite, pids, n_gen, pop_size, root=root)
    result: dict = {}

    for pid in pids:
        pid_archives = archives.get(pid, {})

        # ── build combined pool with (method, seed_idx) tags ──────────────
        pool_pts:  list = []   # accumulated rows for pool_F
        pool_tags: list = []   # (method_key, seed_idx) per row

        # Prepend true reference front if available
        if reference_fronts is not None:
            pf = reference_fronts.get(pid)
            if pf is not None and len(pf) > 0:
                for pt in pf:
                    pool_pts.append(pt)
                    pool_tags.append((_REF_TAG, -1))

        for key in _METHODS:
            for s_idx, F_seed in enumerate(pid_archives.get(key, [])):
                for pt in F_seed:
                    pool_pts.append(pt)
                    pool_tags.append((key, s_idx))

        if not pool_pts:
            print(f'  [rank_pareto] {suite}/pid{pid}: no data, skipping.')
            result[pid] = {k: {'counts': np.zeros((0, n_layers), dtype=int),
                               'n_pts':  np.zeros(0, dtype=float)} for k in _METHODS}
            continue

        pool_F    = np.array(pool_pts, dtype=float)           # (total_pts, n_obj)
        layer_ids = _assign_nd_layers(pool_F, n_layers)       # (total_pts,)

        # ── count per (method, seed, layer) ───────────────────────────────
        n_seeds_per_method = {k: len(pid_archives.get(k, [])) for k in _METHODS}

        # Per-seed archive sizes (denominator for percentage computation)
        n_pts_per_method: dict = {
            k: np.array([len(F) for F in pid_archives.get(k, [])], dtype=float)
            for k in _METHODS
        }

        pid_counts: dict = {}
        for key in _METHODS:
            n_s = n_seeds_per_method[key]
            counts = np.zeros((max(n_s, 1), n_layers), dtype=int)
            for row_idx, (m_key, s_idx) in enumerate(pool_tags):
                if m_key != key:
                    continue
                layer = layer_ids[row_idx]          # 1-based; > n_layers = clamped
                if layer <= n_layers:
                    counts[s_idx, layer - 1] += 1
            pid_counts[key] = {
                'counts': counts[:n_s] if n_s > 0 else np.zeros((0, n_layers), dtype=int),
                'n_pts':  n_pts_per_method[key],
            }

        # Actual pool-level layer sizes (unique pool points per layer)
        layer_sizes = {
            l + 1: int(np.sum(layer_ids == l + 1))
            for l in range(n_layers)
        }

        total_pts_info = ' | '.join(
            f'{k}:{n_seeds_per_method[k]}s'
            for k in _METHODS if n_seeds_per_method[k] > 0
        )
        n_ref = sum(1 for tag in pool_tags if tag[0] == _REF_TAG)
        ref_info = f' (incl. {n_ref} true-PF pts)' if n_ref > 0 else ''
        print(f'  [rank_pareto] {suite}/pid{pid}: pool={len(pool_F)} pts{ref_info} — {total_pts_info}')

        result[pid] = {'_meta': {'layer_sizes': layer_sizes}, **pid_counts}

    return result


# ─── rank-pareto table rendering ─────────────────────────────────────────────

def _rank_pareto_section(
    lines: list,
    section_header: str | None,
    pids: list,
    rank_data: dict,
    layer_idx: int,
    suite: str = 'in1kmop',
    top_k_bracket: int = 3,
) -> None:
    """Append one layer-section to the accumulator *lines*.

    Each cell shows the median percentage of a seed's archive that falls in
    the given non-dominated layer, followed in brackets by the median
    percentage in the top *top_k_bracket* layers combined.
    The row label shows ``MOP{pid}[†] (N1, N13)`` where *N1* is the total
    combined-pool count in layer~1, *N13* in layers~1–3, and the optional
    dagger marks surrogate (non-tabular) problems.

    Parameters
    ----------
    lines : list
        Accumulator for LaTeX source strings.
    section_header : str or None
        Section title, e.g. ``'Layer 1 (Pareto Front)'``. Pass ``None`` to
        suppress the header row (used when only one section is shown).
    pids : list[int]
        Problem IDs in display order.
    rank_data : dict
        Output of :func:`_compute_rank_pareto_counts`.
    layer_idx : int
        0-based index into the counts array column (0 = layer 1).
    suite : str
        EvoXBench suite name; used to look up the search space in
        BENCHMARK_META so surrogate problems are marked with ``†``.
    top_k_bracket : int
        Number of top layers to sum for the bracketed median (default: 3).
    """
    lines.append(r'\midrule')
    if section_header is not None:
        lines.append(
            r'\multicolumn{7}{l}{\small\textit{' + section_header + r'}} \\'
        )
        lines.append(r'\midrule')

    for pid in pids:
        pid_counts = rank_data.get(pid, {})

        # Determine if this problem uses a surrogate predictor
        meta         = BENCHMARK_META.get(suite, {}).get(pid, {})
        search_space = meta.get('search_space', '')
        dagger       = r'$^\dagger$' if search_space not in _TABULAR_SEARCH_SPACES else ''

        # Actual pool-level layer sizes (passed from _compute_rank_pareto_counts)
        layer_sizes = pid_counts.get('_meta', {}).get('layer_sizes', {})
        pool_l1   = layer_sizes.get(1, 0)
        pool_topk = sum(layer_sizes.get(l + 1, 0) for l in range(top_k_bracket))

        # Compute per-seed percentages for each method
        method_pct_layer: dict = {}   # pct in this layer
        method_pct_topk:  dict = {}   # pct in top-k layers combined
        for key in _METHODS:
            entry = pid_counts.get(key)   # {'counts': ndarray, 'n_pts': ndarray}
            if entry is None or len(entry.get('counts', [])) == 0:
                continue
            counts = entry['counts']      # (n_seeds, n_layers)
            n_pts  = entry['n_pts']       # (n_seeds,)
            with np.errstate(invalid='ignore', divide='ignore'):
                pct_layer = np.where(
                    n_pts > 0,
                    counts[:, layer_idx].astype(float) / n_pts * 100.0,
                    0.0,
                )
                k = min(top_k_bracket, counts.shape[1])
                pct_topk = np.where(
                    n_pts > 0,
                    counts[:, :k].sum(axis=1).astype(float) / n_pts * 100.0,
                    0.0,
                )
            method_pct_layer[key] = pct_layer
            method_pct_topk[key]  = pct_topk

        row_label = f'MOP{pid}{dagger} ({pool_l1}, {pool_topk})'

        best_key = (
            max(method_pct_layer, key=lambda k: float(np.mean(method_pct_layer[k])))
            if method_pct_layer else None
        )

        cells = [row_label]
        for key in _METHODS:
            pct_l = method_pct_layer.get(key)
            pct_t = method_pct_topk.get(key)
            if pct_l is None:
                cells.append('--')
                continue
            bold       = (key == best_key)
            mean_layer = float(np.mean(pct_l))
            mean_topk  = float(np.mean(pct_t))
            inner      = f'{mean_layer:.1f}\\% ({mean_topk:.1f}\\%)'
            cell       = r'\textbf{' + inner + r'}' if bold else inner
            cells.append(cell)

        lines.append(' & '.join(cells) + r' \\')


def generate_rank_pareto_tables(
    out_path: str,
    rank_data: dict,
    pids: list,
    n_layers: int = 5,
    suite: str = 'in1kmop',
    suite_label: str = 'IN-1K MOP',
    extended: bool = False,
    use_true_pf: bool = False,
) -> None:
    """Write a single booktabs LaTeX table for the rank-Pareto analysis.

    Each cell shows the mean percentage of a seed's final archive that
    falls in the respective non-dominated layer of the combined pool.

    Parameters
    ----------
    out_path : str
        Destination ``.tex`` file path.
    rank_data : dict
        Output of :func:`_compute_rank_pareto_counts`.
    pids : list[int]
        Problem IDs in display order.
    n_layers : int
        Total number of layers tracked (used only when *extended* is True).
    suite : str
        EvoXBench suite name for BENCHMARK_META look-ups.
    suite_label : str
        Human-readable suite name for the caption.
    extended : bool
        If ``False`` (default): one section showing layer-1 (Pareto front) only.
        If ``True``: *n_layers* sections, one per non-dominated layer.
    use_true_pf : bool
        When ``True`` the pool included the benchmark's pre-defined true Pareto
        front, so layer 1 = genuinely Pareto-optimal solutions.  This affects
        the caption wording and the LaTeX label suffix.
    """
    if extended:
        layer_indices = list(range(n_layers))
        layer_names   = [
            'Layer 1 (Pareto Front)' if i == 0 else f'Layer {i + 1}'
            for i in layer_indices
        ]
        caption_layers = f'top {n_layers} non-dominated layers'
        label_suffix   = '_extended'
    else:
        layer_indices = [0]
        layer_names   = [None]  # no header row for single-section table
        caption_layers = 'non-dominated layer\\,1 (Pareto front)'
        label_suffix   = ''

    # Distinguish true-PF-anchored (c10mop) from combined-pool (in1kmop)
    suite_slug    = suite.replace('mop', '').lower()   # 'c10' or 'in1k'
    label_suffix  = f'_{suite_slug}' + label_suffix

    if use_true_pf:
        pf_desc_l1 = (
            r'percentage of solutions (per seed) that are genuinely '
            r'Pareto-optimal (layer~1 = true Pareto front)'
        )
        pf_desc_ext = (
            r'percentage of solutions (per seed) in each non-dominated '
            r'layer relative to the benchmark\'s true Pareto front'
        )
        pool_desc = (
            r'The non-dominated layers are computed on the union of the true '
            r'Pareto front and all final cumulative Pareto archives from all '
            r'methods and seeds; layer~1 therefore equals the true Pareto front.'
        )
    else:
        pf_desc_l1 = (
            r'percentage of solutions (per seed) '
            r'in the Pareto front approximation (layer~1)'
        )
        pf_desc_ext = (
            r'percentage of solutions (per seed) reaching the '
            + caption_layers + r' of the combined Pareto pool'
        )
        pool_desc = (
            r'The pool is formed by pooling the final cumulative Pareto '
            r'archives from all methods and seeds and applying non-dominated '
            r'sorting to the union.'
        )

    row_label_desc = (
        r'Row labels show the total pool size for layer~1 '
        r'and layers~1--3 in parentheses. '
    )
    bold_desc_l1  = r'\textbf{Bold}: highest mean (layer~1) per problem row.'
    bold_desc_ext = r'\textbf{Bold}: highest mean per problem row.'

    caption_default = (
        r'\caption{Mean '
        + pf_desc_l1
        + r' and in brackets the top~3 non-dominated layers combined, for '
        + suite_label + r'. '
        + row_label_desc
        + pool_desc + r' '
        + bold_desc_l1 + r'}'
    )
    caption_extended = (
        r'\caption{Mean '
        + pf_desc_ext
        + r' for ' + suite_label + r'. '
        + row_label_desc
        + r'Values in brackets show the mean percentage in the top~3 layers combined. '
        + pool_desc + r' '
        + bold_desc_ext + r'}'
    )

    col_header = ' & '.join(
        [r'\textbf{Problem}']
        + [r'\textbf{' + _COLUMN_LABELS[k] + r'}' for k in _METHODS]
    )

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        caption_extended if extended else caption_default,
        r'\label{tab:rank_pareto' + label_suffix + r'}',
        r'\resizebox{\linewidth}{!}{%',
        r'\setlength\tabcolsep{10pt}%',
        r'\begin{tabular}{l @{\hspace{2em}} r r r r r r}',
        r'\toprule',
        col_header + r' \\',
    ]

    for l_idx, l_name in zip(layer_indices, layer_names):
        _rank_pareto_section(
            lines,
            section_header=l_name,
            pids=pids,
            rank_data=rank_data,
            layer_idx=l_idx,
            suite=suite,
        )

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'}',
        r'\end{table*}',
    ]

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nRank-Pareto LaTeX table saved -> {out_path}')


def generate_rank_pareto_combined_table(
    out_path: str,
    rank_data_c10: dict,
    rank_data_in1k: dict,
    c10_pids: list,
    in1k_pids: list,
    n_layers: int = 5,
    extended: bool = False,
) -> None:
    """Write a single LaTeX table combining C-10 MOP and IN-1K MOP sections.

    The table has two subsections separated by a ``\\midrule`` block header:
    one for C-10 MOP (with the true Pareto front anchoring layer~1) and one
    for IN-1K MOP (combined-pool approximation).

    Parameters
    ----------
    out_path : str
        Destination ``.tex`` file path.
    rank_data_c10 : dict
        Output of :func:`_compute_rank_pareto_counts` for ``'c10mop'``,
        computed with ``reference_fronts`` set.
    rank_data_in1k : dict
        Output of :func:`_compute_rank_pareto_counts` for ``'in1kmop'``.
    c10_pids, in1k_pids : list[int]
        Problem IDs for each suite.
    n_layers : int
        Number of non-dominated layers tracked.
    extended : bool
        Show all *n_layers* sections (one per layer) instead of layer~1 only.
    """
    if extended:
        layer_indices = list(range(n_layers))
        layer_names   = [
            'Layer 1 (Pareto Front)' if i == 0 else f'Layer {i + 1}'
            for i in layer_indices
        ]
        label_suffix = '_combined_extended'
        caption_pool_c10  = (
            r'true Pareto front anchors layer~1 for C-10 MOP; '
            r'IN-1K MOP uses a combined-pool approximation'
        )
    else:
        layer_indices = [0]
        layer_names   = [None]
        label_suffix = '_combined'
        caption_pool_c10 = (
            r'true Pareto front anchors layer~1 for C-10 MOP; '
            r'IN-1K MOP uses a combined-pool approximation'
        )

    col_header = ' & '.join(
        [r'\textbf{Problem}']
        + [r'\textbf{' + _COLUMN_LABELS[k] + r'}' for k in _METHODS]
    )

    caption = (
        r'\caption{Mean percentage of solutions (per seed) in the Pareto front '
        r'approximation (layer~1) and in brackets the top~3 non-dominated layers '
        r'combined. '
        + caption_pool_c10
        + r'. Row labels show the total pool size for layer~1 and layers~1--3 '
        r'in parentheses. '
        r'\textbf{Bold}: highest mean (layer~1) per problem row.}'
    )

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        caption,
        r'\label{tab:rank_pareto' + label_suffix + r'}',
        r'\resizebox{\linewidth}{!}{%',
        r'\setlength\tabcolsep{10pt}%',
        r'\begin{tabular}{l @{\hspace{2em}} r r r r r r}',
        r'\toprule',
        col_header + r' \\',
    ]

    for l_idx, l_name in zip(layer_indices, layer_names):
        # ── C-10 MOP subsection ───────────────────────────────────────────────
        lines.append(r'\midrule')
        lines.append(
            r'\multicolumn{7}{l}{\small\textbf{C-10 MOP}}'
            + (r' \quad {\scriptsize\textit{' + l_name + r'}}' if l_name else '')
            + r' \\'
        )
        lines.append(r'\midrule')
        _rank_pareto_section(
            lines,
            section_header=None,
            pids=c10_pids,
            rank_data=rank_data_c10,
            layer_idx=l_idx,
            suite='c10mop',
        )
        # ── IN-1K MOP subsection ──────────────────────────────────────────────
        lines.append(r'\midrule')
        lines.append(
            r'\multicolumn{7}{l}{\small\textbf{IN-1K MOP}}'
            + (r' \quad {\scriptsize\textit{' + l_name + r'}}' if l_name else '')
            + r' \\'
        )
        lines.append(r'\midrule')
        _rank_pareto_section(
            lines,
            section_header=None,
            pids=in1k_pids,
            rank_data=rank_data_in1k,
            layer_idx=l_idx,
            suite='in1kmop',
        )

    lines += [
        r'\bottomrule',
        r'\multicolumn{7}{l}{\footnotesize Surrogate-assisted NAS problems are marked~$^\dagger$.} \\',
        r'\end{tabular}',
        r'}',

        r'\end{table*}',
    ]

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nRank-Pareto combined LaTeX table saved -> {out_path}')


# ─── cross-benchmark convergence grid ────────────────────────────────────────

# Ordered linestyles: each method gets a unique style so the plot is
# distinguishable even in greyscale (colourblind-friendly).
# _LINESTYLES = [
#     '-',                    # solid
#     '--',                   # dashed
#     ':',                    # dotted
#     '-.',                   # dash-dot
#     (0, (3, 1, 1, 1)),      # densely dash-dotted
#     (0, (5, 5)),            # loosely dashed
#     (0, (1, 1)),            # densely dotted
#     (0, (5, 1, 1, 1, 1, 1)),  # dash dot dot
# ]

# Markers for experimentation — all lines are solid, methods distinguished by marker shape.
_MARKERS = [
    'o',    # circle
    's',    # square
    '^',    # triangle up
    'D',    # diamond
    'P',    # plus (filled)
    '*',    # star
    'X',    # x (filled)
    'v',    # triangle down
    'h',    # hexagon
    '<',    # triangle left
]

# Method colours (same palette used in plotter.py / analyze_evoxbench.py)
_CONV_COLOURS = {
    'random':              '#4e79a7',
    'parego':              '#9c755f',
    'mosmac':              '#76b7b2',
    'nsga2':               '#f28e2b',
    'gpsaf':               '#a0cbe8',
    'ssa-nsga2':           '#ff9da7',
    'ssa-nsga2-xgb':       '#e15759',
    'ssa-nsga2-xgb-cheap': '#59a14f',
    'samos-xgb':           '#b07aa1',
    'samos-xgb-c':         '#8cd17d',
}

# Display labels for the legend
_CONV_LABELS = {
    'random':              'Random',
    'parego':              'ParEGO',
    'mosmac':              'MO-SMAC',
    'nsga2':               'NSGA-II',
    'gpsaf':               'GPSAF',
    'ssa-nsga2':           'SSA-RBF',
    'ssa-nsga2-xgb':       'SSA-XGB',
    'ssa-nsga2-xgb-cheap': 'SSA-XGB-C',
    'samos-xgb':           'SAMOS',
    'samos-xgb-c':         'SAMOS-C',
}


def _parse_convergence_spec(spec: str):
    """Parse a problem specifier such as ``'wfg-1'``, ``'c10-8'``, ``'in1k-4'``.

    Returns ``(suite, pid_or_problem)`` where *suite* is one of
    ``'wfg'``, ``'c10mop'``, ``'in1kmop'`` and *pid_or_problem* is either
    a problem-name string (WFG) or an integer PID (EvoXBench).
    """
    s = spec.lower().strip()
    if s.startswith('wfg-'):
        return ('wfg', f'wfg{s[4:]}')
    if s.startswith('c10-'):
        return ('c10mop', int(s[4:]))
    if s.startswith('in1k-'):
        return ('in1kmop', int(s[5:]))
    raise ValueError(
        f'Unknown convergence spec {spec!r}. '
        'Expected one of: wfg-N, c10-N, in1k-N'
    )


def _spec_display_label(suite: str, pid_or_problem) -> str:
    """Human-readable row label, e.g. ``'C-10/MOP1 (2 obj)'``."""
    if suite == 'wfg':
        n = pid_or_problem[3:]   # 'wfg1' -> '1'
        return f'WFG/MOP{n}'
    meta = BENCHMARK_META.get(suite, {}).get(pid_or_problem, {})
    label = meta.get('label', f'MOP{pid_or_problem}')
    n_obj = meta.get('n_obj', '?')
    return f'{label} ({n_obj} obj)'


def _wfg_traj_cache_path(root: str) -> str:
    return os.path.join(root, '_conv_traj_cache.pkl')


def _load_wfg_traj_cache(root: str, methods: list, n_gen: int) -> dict:
    """Load cached WFG trajectory arrays, validating against seed file mtimes.

    Returns a dict ``{folder: (hv_mean, hv_std, igd_mean, igd_std)}`` for
    entries that are still fresh.  Missing or stale entries are omitted.
    """
    path = _wfg_traj_cache_path(root)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'rb') as fh:
            cache = pickle.load(fh)
    except Exception as exc:
        print(f'  [wfg_traj_cache] WARN: could not read {path}: {exc}')
        return {}

    if cache.get('n_gen') != n_gen:
        return {}

    valid: dict = {}
    for folder, entry in cache.get('methods', {}).items():
        stale = False
        for fpath, stored_mt in entry.get('source_mtimes', {}).items():
            if not os.path.isfile(fpath):
                stale = True
                break
            if os.path.getmtime(fpath) > stored_mt + 1e-3:
                stale = True
                break
        if not stale and 'traj' in entry:
            valid[folder] = entry['traj']
    if valid:
        print(f'  [wfg_traj_cache] Cache hit: {path} ({len(valid)} method(s))')
    return valid


def _save_wfg_traj_cache(root: str, n_gen: int, methods_data: dict) -> None:
    """Atomically write WFG trajectory cache.

    *methods_data* maps ``folder -> {'traj': tuple, 'source_mtimes': dict}``.
    """
    path  = _wfg_traj_cache_path(root)
    cache = {'n_gen': n_gen, 'methods': methods_data}
    dir_  = os.path.dirname(path) or '.'
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(f'  [wfg_traj_cache] Saved -> {path}')


def _load_convergence_for_spec(
    suite: str,
    pid_or_problem,
    methods: list,
    wfg_n_gen: int,
    wfg_pop: int,
    evox_n_gen: int,
    evox_pop: int,
    experiment_name: str,
    evox_root: str,
    n_obj_wfg: int = 2,
    no_norm: bool = False,
    evox_approx_cache: dict | None = None,
):
    """Load per-method convergence trajectories and HV ceiling for one problem.

    Parameters
    ----------
    evox_approx_cache : dict | None
        Pre-built ``{(suite, pid): approx_info}`` mapping from ``main()``
        (keyed as returned by :func:`build_pareto_approximation`).  When
        provided the EvoXBench branch reuses these objects directly, avoiding
        a full Pareto-approx rebuild and serving trajectories from the
        ``traj_cache`` already embedded in each approx dict.

    Returns
    -------
    trajectories : dict[str, tuple | None]
        ``{method_key: (hv_mean, hv_std, igd_mean, igd_std)}`` or ``None``
        when no data exists for that method.
    hv_ceiling : float
        Reference HV value (Pareto-front HV) to draw as a horizontal line.
    pop_size : int
        Population size used to scale the x-axis to evaluations.
    """
    from analysis.plotter import load_indicator_trajectories
    from pymoo.indicators.hv import HV as _HV

    if suite == 'wfg':
        from problem.pymoo.benchmark_utils import (
            get_pareto_front, default_ref_point, build_problem,
        )
        problem_name = pid_or_problem           # e.g. 'wfg1'
        root = _results_root_wfg(problem_name, experiment_name, wfg_n_gen, wfg_pop)

        # HV ceiling from the analytic Pareto front
        problem_obj = build_problem(problem_name, n_obj_wfg)
        pf         = get_pareto_front(problem_obj, n_obj_wfg)
        ref_pt     = default_ref_point(problem_name, n_obj_wfg)
        hv_ceiling = float(_HV(ref_point=ref_pt)(pf))

        # Load trajectory cache; only recompute stale / missing entries.
        cache_hit   = _load_wfg_traj_cache(root, methods, wfg_n_gen)
        updated: dict = {}
        trajectories: dict = {}
        for key in methods:
            folder = _WFG_FOLDER.get(key, key)
            if folder in cache_hit:
                trajectories[key] = cache_hit[folder]
                seed_dir = os.path.join(root, folder)
                updated[folder] = {
                    'traj': cache_hit[folder],
                    'source_mtimes': _seed_mtimes(seed_dir),
                }
            else:
                traj = load_indicator_trajectories(folder, wfg_n_gen, root)
                trajectories[key] = traj
                seed_dir = os.path.join(root, folder)
                if traj is not None:
                    updated[folder] = {
                        'traj': traj,
                        'source_mtimes': _seed_mtimes(seed_dir),
                    }
        if updated:
            _save_wfg_traj_cache(root, wfg_n_gen, updated)

        return trajectories, hv_ceiling, wfg_pop

    else:                                       # c10mop / in1kmop
        from analysis.convergence import get_or_recompute_trajectories, save_approx_cache

        pid = pid_or_problem
        root = _results_root_evox(suite, pid, evox_n_gen, evox_pop, root=evox_root)
        evox_folders = [_EVOX_FOLDER.get(k, k) for k in methods]

        # Reuse pre-built approx from main() when available (avoids rebuild).
        approx = (
            evox_approx_cache.get((suite, pid))
            if evox_approx_cache is not None
            else None
        )
        if approx is None:
            norm_bounds = compute_empirical_norm_bounds(evox_folders, root) if no_norm else None
            approx = build_pareto_approximation(
                suite, pid, evox_folders, evox_pop, evox_n_gen,
                norm_bounds=norm_bounds,
                results_root=root,
            )
        else:
            norm_bounds = approx.get('norm_bounds') if no_norm else None

        if approx is not None:
            hv_ceiling = float(_HV(ref_point=approx['ref_point'])(approx['pareto_approx']))
        else:
            hv_ceiling = 1.0    # fallback when no approximation available

        trajectories = {}
        for key in methods:
            folder = _EVOX_FOLDER.get(key, key)
            if approx is not None:
                # get_or_recompute_trajectories serves from traj_cache when available
                traj = get_or_recompute_trajectories(
                    folder, evox_n_gen, root, approx,
                    norm_bounds=norm_bounds,
                )
            else:
                traj = load_indicator_trajectories(folder, evox_n_gen, root)
            trajectories[key] = traj

        if approx is not None:
            save_approx_cache(root, approx)

        return trajectories, hv_ceiling, evox_pop


def plot_convergence_grid_combined(
    specs: list,
    methods: list,
    wfg_n_gen: int,
    wfg_pop: int,
    evox_n_gen: int,
    evox_pop: int,
    experiment_name: str,
    evox_root: str,
    out_path: str,
    n_obj_wfg: int = 2,
    no_norm: bool = False,
    font_scale: float = 1.0,
    evox_approx_cache: dict | None = None,
) -> None:
    """Multi-row convergence plot: one row per problem, HV left / IGD+ right.

    Parameters
    ----------
    specs : list[str]
        Problem specifiers, e.g. ``['wfg-1', 'c10-1', 'c10-8', 'in1k-4', 'in1k-9']``.
        Each element produces one row in the figure.
    methods : list[str]
        Canonical method keys (from ``_METHODS``).
    wfg_n_gen, wfg_pop :
        Budget params for WFG results.
    evox_n_gen, evox_pop :
        Budget params for EvoXBench results.
    experiment_name :
        WFG sub-tree under ``results/`` (e.g. ``'pymoo_benchmark/2_obj'``).
    evox_root :
        Root folder for EvoXBench results (e.g. ``'results/evoxbench'``).
    out_path :
        Where to save the PNG.
    n_obj_wfg :
        Number of WFG objectives (used to build the reference point).
    no_norm :
        Derive empirical normalization bounds for EvoXBench (no-norm mode).
    font_scale :
        Multiplier applied to all font sizes.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    n_rows = len(specs)

    plt.rcParams.update({
        'font.size':        12 * font_scale,
        'axes.titlesize':   20 * font_scale,
        'axes.labelsize':   14 * font_scale,
        'xtick.labelsize':  12 * font_scale,
        'ytick.labelsize':  12 * font_scale,
        'legend.fontsize':  12 * font_scale,
        'figure.dpi':       150,
    })

    # Assign one marker per method (consistent across all rows)
    method_mk = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(methods)}

    col_w    = 8 * font_scale**0.5   
    row_h    = 4 * font_scale**0.5
    legend_h = 0.8 * font_scale   # extra bottom space for shared legend

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(col_w * 2, row_h * n_rows),
        squeeze=False,
    )

    # Accumulate legend handles/labels in first pass (one entry per method)
    legend_handles: list = []
    legend_labels:  list = []
    seen_labels:    set  = set()
    lw = 1.8 * font_scale

    for row_idx, spec in enumerate(specs):
        suite, pid_or_problem = _parse_convergence_spec(spec)
        display_label = _spec_display_label(suite, pid_or_problem)

        print(f'  [convergence_grid] Loading {spec} ({display_label}) ...')
        trajectories, hv_ceiling, pop_size = _load_convergence_for_spec(
            suite, pid_or_problem, methods,
            wfg_n_gen, wfg_pop, evox_n_gen, evox_pop,
            experiment_name, evox_root, n_obj_wfg, no_norm,
            evox_approx_cache=evox_approx_cache,
        )

        ax_hv  = axes[row_idx, 0]
        ax_igd = axes[row_idx, 1]

        for key in methods:
            traj = trajectories.get(key)
            if traj is None:
                continue
            hv_mean, hv_std, igd_mean, igd_std = traj
            x      = np.arange(1, len(hv_mean) + 1) * pop_size
            colour = _CONV_COLOURS.get(key, '#555555')
            label  = _CONV_LABELS.get(key, key)
            mk     = method_mk[key]

            line, = ax_hv.plot(x, hv_mean, label=label,
                               color=colour, linestyle='-', linewidth=lw,
                               marker=mk, markevery=0.15, markersize=5 * font_scale)
            ax_hv.fill_between(x, hv_mean - hv_std, hv_mean + hv_std,
                               alpha=0.12, color=colour)

            ax_igd.plot(x, igd_mean, label=label,
                        color=colour, linestyle='-', linewidth=lw,
                        marker=mk, markevery=0.15, markersize=5 * font_scale)
            ax_igd.fill_between(x, np.maximum(0.0, igd_mean - igd_std),
                                igd_mean + igd_std, alpha=0.12, color=colour)

            if label not in seen_labels:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_labels.add(label)

        # HV ceiling: dashed line + inline annotation (per-row, not in legend)
        ax_hv.axhline(hv_ceiling, color='#333333', linestyle='--',
                      linewidth=0.9 * font_scale, alpha=0.85, zorder=1)
        trans = blended_transform_factory(ax_hv.transAxes, ax_hv.transData)
        ax_hv.text(0.99, hv_ceiling, f' PF HV={hv_ceiling:.4f}',
                   transform=trans, ha='right', va='bottom', color='#333333',
                   fontsize=12 * font_scale)
        ylo, _ = ax_hv.get_ylim()
        ax_hv.set_ylim(ylo, hv_ceiling + (hv_ceiling - ylo) * 0.06 * font_scale)

        # Row label on left y-axis of HV subplot
        ax_hv.set_ylabel(display_label, fontweight='bold')
        ax_igd.set_ylabel('')

        # Column headers on first row only
        if row_idx == 0:
            ax_hv.set_title('Hypervolume  ↑')
            ax_igd.set_title('IGD\u207a  ↓')

        # X-axis label on last row only
        if row_idx == n_rows - 1:
            ax_hv.set_xlabel('Evaluations')
            ax_igd.set_xlabel('Evaluations')

        ax_hv.tick_params(axis='both')
        ax_igd.tick_params(axis='both')
        ax_hv.grid(True, alpha=0.3)
        ax_igd.grid(True, alpha=0.3)

    # Shared legend: two rows below the figure
    import math
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=math.ceil(len(legend_labels) / 2),
        frameon=True,
        fontsize=10 * font_scale,
    )

    fig.tight_layout()
    # Reserve room for the two-row legend at the bottom
    bottom_frac = (legend_h * 2) / (row_h * n_rows + legend_h * 2)
    fig.subplots_adjust(bottom=bottom_frac + 0.01)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Convergence grid saved -> {out_path}')


def plot_hv_convergence_3x3(
    suite: str,
    methods: list,
    wfg_n_gen: int,
    wfg_pop: int,
    evox_n_gen: int,
    evox_pop: int,
    experiment_name: str,
    evox_root: str,
    out_path: str,
    n_obj_wfg: int = 2,
    no_norm: bool = False,
    font_scale: float = 1.0,
    evox_approx_cache: dict | None = None,
) -> None:
    """3×3 HV-only convergence grid for all 9 problems of a benchmark suite.

    Parameters
    ----------
    suite : str
        One of ``'wfg'``, ``'c10'``, or ``'in1k'``.  Determines which 9
        problems are plotted.
    methods : list[str]
        Canonical method keys (from ``_METHODS``).
    wfg_n_gen, wfg_pop :
        Budget params for WFG results.
    evox_n_gen, evox_pop :
        Budget params for EvoXBench results.
    experiment_name :
        WFG sub-tree under ``results/``.
    evox_root :
        Root folder for EvoXBench results.
    out_path :
        Where to save the PNG.
    n_obj_wfg :
        Number of WFG objectives.
    no_norm :
        Derive empirical normalisation bounds for EvoXBench (no-norm mode).
    font_scale :
        Multiplier applied to all font sizes.
    evox_approx_cache :
        Pre-built ``{(suite, pid): approx_info}`` mapping (avoids rebuild).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    suite_lc = suite.lower()
    if suite_lc == 'wfg':
        specs = [f'wfg-{i}' for i in range(1, 10)]
    elif suite_lc == 'c10':
        specs = [f'c10-{i}' for i in range(1, 10)]
    elif suite_lc == 'in1k':
        specs = [f'in1k-{i}' for i in range(1, 10)]
    else:
        raise ValueError(
            f'Unknown suite keyword {suite!r}. Expected one of: wfg, c10, in1k'
        )

    plt.rcParams.update({
        'font.size':        10 * font_scale,
        'axes.titlesize':   11 * font_scale,
        'axes.labelsize':   10 * font_scale,
        'xtick.labelsize':  9  * font_scale,
        'ytick.labelsize':  9  * font_scale,
        'legend.fontsize':  9  * font_scale,
        'figure.dpi':       150,
    })

    method_mk = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(methods)}
    lw = 1.5 * font_scale

    cell_w   = 4.5 * font_scale ** 0.5
    cell_h   = 3.5 * font_scale ** 0.5
    legend_h = 0.7 * font_scale

    fig, axes = plt.subplots(
        3, 3,
        figsize=(cell_w * 3, cell_h * 3),
        squeeze=False,
    )

    legend_handles: list = []
    legend_labels:  list = []
    seen_labels:    set  = set()

    for idx, spec in enumerate(specs):
        row, col = divmod(idx, 3)
        suite_name, pid_or_problem = _parse_convergence_spec(spec)
        display_label = _spec_display_label(suite_name, pid_or_problem)

        print(f'  [3x3_convergence] Loading {spec} ({display_label}) ...')
        trajectories, hv_ceiling, pop_size = _load_convergence_for_spec(
            suite_name, pid_or_problem, methods,
            wfg_n_gen, wfg_pop, evox_n_gen, evox_pop,
            experiment_name, evox_root, n_obj_wfg, no_norm,
            evox_approx_cache=evox_approx_cache,
        )

        ax = axes[row, col]

        for key in methods:
            traj = trajectories.get(key)
            if traj is None:
                continue
            hv_mean, hv_std, *_ = traj
            x      = np.arange(1, len(hv_mean) + 1) * pop_size
            colour = _CONV_COLOURS.get(key, '#555555')
            label  = _CONV_LABELS.get(key, key)
            mk     = method_mk[key]

            line, = ax.plot(
                x, hv_mean, label=label,
                color=colour, linestyle='-', linewidth=lw,
                marker=mk, markevery=0.2, markersize=4 * font_scale,
            )
            ax.fill_between(
                x, hv_mean - hv_std, hv_mean + hv_std,
                alpha=0.12, color=colour,
            )

            if label not in seen_labels:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_labels.add(label)

        # HV ceiling reference line (not added to legend)
        ax.axhline(
            hv_ceiling, color='#333333', linestyle='--',
            linewidth=0.8 * font_scale, alpha=0.85, zorder=1,
        )

        ax.set_title(display_label)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both')

        if col == 0:
            ax.set_ylabel('Hypervolume  ↑')
        if row == 2:
            ax.set_xlabel('Evaluations')

    # Shared legend below the grid (two rows)
    import math
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=math.ceil(max(1, len(legend_labels)) / 2),
        frameon=True,
        fontsize=9 * font_scale,
    )

    fig.tight_layout()
    bottom_frac = (legend_h * 2) / (cell_h * 3 + legend_h * 2)
    fig.subplots_adjust(bottom=bottom_frac + 0.01)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  3×3 HV convergence grid saved -> {out_path}')


def main(args) -> None:
    wfg_problems = args.wfg_problems
    skip_set     = set(args.skip_pids or [])
    pids         = [p for p in args.pids if p not in skip_set]
    if skip_set:
        print(f'Skipping EvoXBench PIDs: {sorted(skip_set)}')
    n_var_wfg    = 2 * (args.n_obj_wfg - 1) + 10

    caption_note = ''
    if args.evox_no_norm:
        # Log which methods have result files (informational only — all _METHODS
        # are always loaded and shown; missing ones render as '--').
        _detect_available_evox_methods(
            args.evox_root, ['c10mop', 'in1kmop'], pids, args.evox_n_gen, args.evox_pop,
        )
        caption_note = (
            r'C-10 MOP and IN-1K MOP values use empirically derived normalization bounds '
            r'(pooled min/max from all runs; evoxbench built-in bounds are inaccurate '
            r'for surrogate-assisted NAS problems). \textbf{Still under construction, not all seeds are in the table}}'
        )

    # ── resolve method filter ──────────────────────────────────────────────────
    active_methods = (
        [k for k in _METHODS if k in set(args.methods)]
        if args.methods
        else _METHODS
    )
    if args.methods:
        excluded = [k for k in _METHODS if k not in active_methods]
        print(f'Active methods: {active_methods}')
        if excluded:
            print(f'Excluded methods: {excluded}')

    print('=' * 60)
    print('  Collecting WFG data ...')
    print('=' * 60)
    wfg_data = _collect_wfg_data(
        wfg_problems, args.experiment_name, args.wfg_n_gen, args.wfg_pop,
        methods=active_methods,
        max_seeds=args.max_seeds,
    )

    print('\n' + '=' * 60)
    print('  Collecting C-10 MOP data ...')
    print('=' * 60)
    c10_data = _collect_evox_data('c10mop', pids, args.evox_n_gen, args.evox_pop,
                                   force=args.force, evox_root=args.evox_root,
                                   no_norm=args.evox_no_norm,
                                   methods=active_methods,
                                   max_seeds=args.max_seeds)

    print('\n' + '=' * 60)
    print('  Collecting IN-1K MOP data ...')
    print('=' * 60)
    in1k_data = _collect_evox_data('in1kmop', pids, args.evox_n_gen, args.evox_pop,
                                    force=args.force, evox_root=args.evox_root,
                                    no_norm=args.evox_no_norm,
                                    methods=active_methods,
                                    max_seeds=args.max_seeds)

    def _out_path_for(m: str) -> str:
        """Derive the output path for a given metric from ``args.out``."""
        if m == 'hv':
            return args.out
        base, ext = os.path.splitext(args.out)
        # Strip a trailing '_hv' from the base name if present, then append metric.
        clean = base[:-3] if base.endswith('_hv') else base
        return f'{clean}_{m}{ext}'

    metrics_to_run = ['hv', 'igd_plus'] if args.metric == 'both' else [args.metric]
    for m in metrics_to_run:
        generate_combined_hv_table(
            out_path=_out_path_for(m),
            wfg_data=wfg_data,
            c10_data=c10_data,
            in1k_data=in1k_data,
            wfg_problems=wfg_problems,
            pids=pids,
            n_obj_wfg=args.n_obj_wfg,
            n_var_wfg=n_var_wfg,
            metric=m,
            methods=active_methods,
            caption_note=caption_note,
            include_wfg=not args.evox_no_norm,
            c10_pids=[8, 9] if args.evox_no_norm else None,
            max_seeds=args.max_seeds,
            rotate=args.rotate,
        )

    # ── optional rank-Pareto analysis ─────────────────────────────────────────
    rank_data      = None   # IN-1K rank-Pareto data (populated below if requested)
    rank_data_c10  = None   # C-10  rank-Pareto data (populated below if requested)

    if args.rank_pareto:
        print('\n' + '=' * 60)
        print('  Computing rank-Pareto layer counts (IN-1K MOP) ...')
        print('=' * 60)
        rank_data = _compute_rank_pareto_counts(
            suite    = 'in1kmop',
            pids     = pids,
            n_gen    = args.evox_n_gen,
            pop_size = args.evox_pop,
            n_layers = args.rank_pareto_n_layers,
            root     = args.evox_root,
        )
        generate_rank_pareto_tables(
            out_path  = args.rank_pareto_out,
            rank_data = rank_data,
            pids      = pids,
            n_layers  = args.rank_pareto_n_layers,
            suite     = 'in1kmop',
            suite_label = 'IN-1K MOP',
            extended  = args.rank_pareto_extended,
            use_true_pf = False,
        )

    if args.rank_pareto_c10:
        c10_pids = args.rank_pareto_c10_pids
        print('\n' + '=' * 60)
        print(
            f'  Computing rank-Pareto layer counts '
            f'(C-10 MOP, PIDs {c10_pids}) ...'
        )
        print('=' * 60)
        print('  Loading true Pareto fronts from benchmark ...')
        true_pfs = _load_true_pf_evox('c10mop', c10_pids)
        rank_data_c10 = _compute_rank_pareto_counts(
            suite            = 'c10mop',
            pids             = c10_pids,
            n_gen            = args.evox_n_gen,
            pop_size         = args.evox_pop,
            n_layers         = args.rank_pareto_n_layers,
            reference_fronts = true_pfs,
            root             = args.evox_root,
        )
        generate_rank_pareto_tables(
            out_path    = args.rank_pareto_c10_out,
            rank_data   = rank_data_c10,
            pids        = c10_pids,
            n_layers    = args.rank_pareto_n_layers,
            suite       = 'c10mop',
            suite_label = 'C-10 MOP',
            extended    = args.rank_pareto_extended,
            use_true_pf = True,
        )

    if args.rank_pareto_combined:
        c10_pids   = args.rank_pareto_c10_pids
        in1k_pids  = pids
        print('\n' + '=' * 60)
        print('  Computing rank-Pareto combined table (C-10 + IN-1K) ...')
        print('=' * 60)
        # load c10 data (with true PF) if not already computed
        if not args.rank_pareto_c10:
            print('  Loading true Pareto fronts for C-10 MOP ...')
            true_pfs = _load_true_pf_evox('c10mop', c10_pids)
            rank_data_c10 = _compute_rank_pareto_counts(
                suite            = 'c10mop',
                pids             = c10_pids,
                n_gen            = args.evox_n_gen,
                pop_size         = args.evox_pop,
                n_layers         = args.rank_pareto_n_layers,
                reference_fronts = true_pfs,
                root             = args.evox_root,
            )
        # reuse already-computed in1k data; else fetch fresh
        if not args.rank_pareto:
            rank_data_in1k_comb = _compute_rank_pareto_counts(
                suite    = 'in1kmop',
                pids     = in1k_pids,
                n_gen    = args.evox_n_gen,
                pop_size = args.evox_pop,
                n_layers = args.rank_pareto_n_layers,
                root     = args.evox_root,
            )
        else:
            rank_data_in1k_comb = rank_data
        generate_rank_pareto_combined_table(
            out_path       = args.rank_pareto_combined_out,
            rank_data_c10  = rank_data_c10,
            rank_data_in1k = rank_data_in1k_comb,
            c10_pids       = c10_pids,
            in1k_pids      = in1k_pids,
            n_layers       = args.rank_pareto_n_layers,
            extended       = args.rank_pareto_extended,
        )

    # ── cross-benchmark convergence grid ──────────────────────────────────────
    if args.convergence_plot_content:
        _SUITE_KEYWORDS = {'wfg', 'c10', 'in1k'}
        regular_specs  = [s for s in args.convergence_plot_content
                          if s.lower() not in _SUITE_KEYWORDS]
        suite_keywords = [s.lower() for s in args.convergence_plot_content
                          if s.lower() in _SUITE_KEYWORDS]

        conv_out = getattr(args, 'convergence_plot_out',
                           os.path.join('results', 'figures', 'convergence_grid.png'))

        # Build a combined approx cache keyed by (suite, pid) from the data
        # already loaded above, so EvoXBench trajectories are served from the
        # traj_cache embedded in each approx_info dict (no rebuild needed).
        _evox_approx_cache: dict = {}
        for _pid in pids:
            for _suite in ('c10mop', 'in1kmop'):
                _root = _results_root_evox(_suite, _pid, args.evox_n_gen, args.evox_pop,
                                           root=args.evox_root)
                _approx = build_pareto_approximation(
                    _suite, _pid,
                    [_EVOX_FOLDER.get(k, k) for k in active_methods],
                    args.evox_pop, args.evox_n_gen,
                    results_root=_root,
                )
                if _approx is not None:
                    _evox_approx_cache[(_suite, _pid)] = _approx

        # Existing multi-row combined plot (individual specs like wfg-1, c10-3 …)
        if regular_specs:
            print('\n' + '=' * 60)
            print(f'  Convergence grid: {regular_specs}')
            print('=' * 60)
            plot_convergence_grid_combined(
                specs              = regular_specs,
                methods            = active_methods,
                wfg_n_gen          = args.wfg_n_gen,
                wfg_pop            = args.wfg_pop,
                evox_n_gen         = args.evox_n_gen,
                evox_pop           = args.evox_pop,
                experiment_name    = args.experiment_name,
                evox_root          = args.evox_root,
                out_path           = conv_out,
                n_obj_wfg          = args.n_obj_wfg,
                no_norm            = args.evox_no_norm,
                font_scale         = getattr(args, 'font_scale', 1.0),
                evox_approx_cache  = _evox_approx_cache,
            )

        # New 3×3 HV-only grids (one plot per suite keyword)
        for suite_kw in suite_keywords:
            base, ext = os.path.splitext(conv_out)
            suite_out = f'{base}_{suite_kw}_3x3{ext}'
            print('\n' + '=' * 60)
            print(f'  3×3 HV convergence grid: {suite_kw.upper()}')
            print('=' * 60)
            plot_hv_convergence_3x3(
                suite              = suite_kw,
                methods            = active_methods,
                wfg_n_gen          = args.wfg_n_gen,
                wfg_pop            = args.wfg_pop,
                evox_n_gen         = args.evox_n_gen,
                evox_pop           = args.evox_pop,
                experiment_name    = args.experiment_name,
                evox_root          = args.evox_root,
                out_path           = suite_out,
                n_obj_wfg          = args.n_obj_wfg,
                no_norm            = args.evox_no_norm,
                font_scale         = getattr(args, 'font_scale', 1.0),
                evox_approx_cache  = _evox_approx_cache,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Generate a combined HV LaTeX table for '
            'WFG / C-10 MOP / IN-1K MOP benchmarks.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--out',
        default=os.path.join('results', 'all_benchmarks_hv_table.tex'),
        help='Output path for the generated LaTeX table.',
    )
    parser.add_argument(
        '--experiment_name',
        default='pymoo_benchmark/2_obj',
        help='WFG results sub-tree root under results/.',
    )
    parser.add_argument(
        '--wfg_problems',
        nargs='+',
        default=[f'wfg{i}' for i in range(1, 10)],
        metavar='PROBLEM',
        help='WFG problem names.',
    )
    parser.add_argument(
        '--pids',
        nargs='+',
        type=int,
        default=list(range(1, 10)),
        metavar='PID',
        help='EvoXBench problem IDs.',
    )
    parser.add_argument(
        '--skip_pids',
        nargs='*',
        type=int,
        default=None,
        metavar='PID',
        help='EvoXBench problem IDs to skip (default: 7, slow HV computation).',
    )
    parser.add_argument('--wfg_n_gen',  type=int, default=60,
                        help='Number of generations for WFG runs.')
    parser.add_argument('--wfg_pop',    type=int, default=20,
                        help='Population size for WFG runs.')
    parser.add_argument('--evox_n_gen', type=int, default=60,
                        help='Number of generations for EvoXBench runs.')
    parser.add_argument('--evox_pop',   type=int, default=20,
                        help='Population size for EvoXBench runs.')
    parser.add_argument('--n_obj_wfg',  type=int, default=2,
                        help='Number of objectives for WFG (used in caption).')
    parser.add_argument('--force', action='store_true',
                        help='Ignore the indicators cache and recompute all EvoXBench values.')
    parser.add_argument('--rotate', action='store_true',
                        help='Wrap the table in a sidewaystable environment (requires \\usepackage{rotating}) to rotate it 90° and fit on a portrait page.')
    parser.add_argument(
        '--max_seeds', '--max-seeds',
        type=int,
        default=None,
        dest='max_seeds',
        metavar='N',
        help=(
            'Use only the first N seed files (sorted order) per method/problem. '
            'Default: use all available seeds.'
        ),
    )
    parser.add_argument(
        '--evox_root', '--evox-root',
        default='results/evoxbench',
        dest='evox_root',
        metavar='PATH',
        help='Root folder for EvoXBench results (default: results/evoxbench).',
    )
    parser.add_argument(
        '--evox_no_norm', '--evox-no-norm',
        action='store_true',
        dest='evox_no_norm',
        help=(
            'Treat EvoXBench results as raw (no_norm mode): derive empirical '
            'normalization bounds post-hoc from the pooled final archives.'
        ),
    )
    parser.add_argument(
        '--metric',
        choices=['hv', 'igd_plus', 'both'],
        default='both',
        help='Indicator(s) to tabulate (default: both).',
    )

    # ── rank-Pareto options ────────────────────────────────────────────────────
    parser.add_argument(
        '--rank_pareto',
        action='store_true',
        help='Compute per-method per-layer solution counts and write two LaTeX tables.',
    )
    parser.add_argument(
        '--rank_pareto_n_layers',
        type=int,
        default=5,
        metavar='K',
        help='Number of non-dominated layers to track (default: 5).',
    )
    parser.add_argument(
        '--rank_pareto_out',
        default=os.path.join('results', 'rank_pareto_in1k.tex'),
        metavar='PATH',
        help='Output path for the rank-Pareto LaTeX table.',
    )
    parser.add_argument(
        '--rank_pareto_extended',
        action='store_true',
        help=(
            'Show all --rank_pareto_n_layers non-dominated layers '
            'instead of layer 1 only.'
        ),
    )
    parser.add_argument(
        '--rank_pareto_c10',
        action='store_true',
        help=(
            'Compute rank-Pareto analysis for C-10 MOP using the '
            'benchmark\'s pre-defined true Pareto front as the pool reference.'
        ),
    )
    parser.add_argument(
        '--rank_pareto_c10_pids',
        nargs='+',
        type=int,
        default=list(range(1, 10)),
        metavar='PID',
        help=(
            'C-10 MOP problem IDs for rank-Pareto analysis '
            '(default: 1-7, the tabular benchmarks with a true PF).'
        ),
    )
    parser.add_argument(
        '--rank_pareto_c10_out',
        default=os.path.join('results', 'rank_pareto_c10.tex'),
        metavar='PATH',
        help='Output path for the C-10 MOP rank-Pareto LaTeX table.',
    )
    parser.add_argument(
        '--rank_pareto_combined',
        action='store_true',
        help=(
            'Produce a single combined table with C-10 MOP and IN-1K MOP '
            'subsections (implies loading data for both suites).'
        ),
    )
    parser.add_argument(
        '--rank_pareto_combined_out',
        default=os.path.join('results', 'rank_pareto_combined.tex'),
        metavar='PATH',
        help='Output path for the combined C-10 + IN-1K rank-Pareto LaTeX table.',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=None,
        metavar='METHOD',
        choices=list(_METHODS),
        help=(
            'Restrict output to these method keys (e.g. random mosmac nsga2 gpsaf samos-xgb). '
            'Default: all methods in _METHODS. Useful to exclude methods whose '
            'result files are not yet available (e.g. --methods random mosmac nsga2 gpsaf samos-xgb).'
        ),
    )
    parser.add_argument(
        '--convergence_plot_content', '--convergence-plot-content',
        nargs='+',
        default=None,
        dest='convergence_plot_content',
        metavar='SPEC',
        help=(
            'Generate a combined convergence grid (HV left, IGD+ right, one row per problem). '
            'Each SPEC is one of: wfg-N, c10-N, in1k-N  '
            '(e.g. --convergence_plot_content wfg-1 c10-1 c10-8 in1k-4 in1k-9). '
            'The number of rows equals the number of specs provided.'
        ),
    )
    parser.add_argument(
        '--convergence_plot_out', '--convergence-plot-out',
        default=os.path.join('results', 'figures', 'convergence_grid.png'),
        dest='convergence_plot_out',
        metavar='PATH',
        help='Output path for the convergence grid PNG (default: results/figures/convergence_grid.png).',
    )
    parser.add_argument(
        '--font_scale', '--font-scale',
        type=float,
        default=1.0,
        dest='font_scale',
        help='Font size multiplier for convergence plot outputs (default: 1.0).',
    )

    arguments = parser.parse_args()
    print(f'Arguments: {arguments}')
    main(arguments)
