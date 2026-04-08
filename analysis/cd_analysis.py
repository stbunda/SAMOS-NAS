"""analysis/cd_analysis.py — Robust ranking / critical difference analysis.

Collects final per-seed HV and IGD+ values across WFG, C10MOP and IN1KMOP
benchmarks, caches the resulting DataFrame on disk with mtime-based staleness
invalidation, and wires the data into robustranking's BootstrapComparison for
confidence-interval ranking plots.

Public API
----------
build_indicators_df(benchmark_name, methods, pop_size, n_gen, force, **kw)
    -> pd.DataFrame  with columns: algorithm, instance, hv, igd_plus

build_robustranking_benchmark(df)
    -> robustranking.Benchmark

run_comparison(bm, alpha, bootstrap_runs, minimise)
    -> BootstrapComparison (already computed)

plot_cd_panels(comparisons_dict, out_dir, labels)
    One figure per metric, one subplot per benchmark.

plot_cd_grid(comparisons_dict, out_dir, labels)
    Single 2 × N figure: rows = metrics, cols = benchmarks.
"""

import json
import os
import pickle
import tempfile
import time

import numpy as np
import pandas as pd

# ─── canonical method → filesystem folder maps ────────────────────────────────

# WFG results live under results/pymoo_benchmark/2_obj/<problem>/B*_P*/<folder>/
_WFG_FOLDER_MAP: dict[str, str] = {
    'random':    'random',
    'nsga2':     'nsga2',
    'parego':    'parego',
    'mosmac':    'mosmac',
    'gpsaf':     'gpsaf-default',
    'samos-xgb': 'samos-xgb-i200-g20',
}

# EvoXBench results live under results/evoxbench/<suite>/pid<p>/B*_P*/<folder>/
_EVOX_FOLDER_MAP: dict[str, str] = {
    'random':    'random',
    'nsga2':     'nsga2',
    'parego':    'parego',
    'mosmac':    'mosmac',
    'gpsaf':     'gpsaf-default',
    'samos-xgb': 'samos-xgb',
}

# ─── display labels ───────────────────────────────────────────────────────────

DISPLAY_LABELS: dict[str, str] = {
    'random':    'Random',
    'nsga2':     'NSGA-II',
    'parego':    'ParEGO',
    'mosmac':    'MO-SMAC',
    'gpsaf':     'GPSAF',
    'samos-xgb': 'SAMOS-XGB',
}

# ─── on-disk cache helpers ────────────────────────────────────────────────────

_CACHE_ROOT = os.path.join('results', 'cd_analysis')


def _cache_paths(benchmark_name: str, budget: int, pop_size: int) -> tuple[str, str]:
    d = os.path.join(_CACHE_ROOT, benchmark_name, f'B{budget}_P{pop_size}')
    return os.path.join(d, 'indicators_df.csv'), os.path.join(d, 'indicators_meta.json')


def _load_cached_df(benchmark_name: str, budget: int, pop_size: int):
    """Return (DataFrame, meta) or (None, None) if cache is absent or stale."""
    pkl_path, meta_path = _cache_paths(benchmark_name, budget, pop_size)
    if not os.path.isfile(pkl_path) or not os.path.isfile(meta_path):
        return None, None
    try:
        with open(meta_path, 'r') as fh:
            meta = json.load(fh)
    except Exception as exc:
        print(f'  [cd_analysis] WARN: could not read meta {meta_path}: {exc}')
        return None, None

    for path, stored_mtime in meta.get('source_mtimes', {}).items():
        if not os.path.isfile(path):
            print(f'  [cd_analysis] Cache stale: {path} no longer exists.')
            return None, None
        if os.path.getmtime(path) > stored_mtime + 1e-3:
            print(f'  [cd_analysis] Cache stale: {path} has been modified.')
            return None, None

    try:
        df = pd.read_csv(pkl_path)
    except Exception as exc:
        print(f'  [cd_analysis] WARN: could not load cached DataFrame {pkl_path}: {exc}')
        return None, None

    print(f'  [cd_analysis] Cache hit: {pkl_path} ({len(df)} rows)')
    return df, meta


def _save_cached_df(
    df: pd.DataFrame,
    source_mtimes: dict,
    benchmark_name: str,
    budget: int,
    pop_size: int,
) -> None:
    csv_path, meta_path = _cache_paths(benchmark_name, budget, pop_size)
    dir_ = os.path.dirname(csv_path)
    os.makedirs(dir_, exist_ok=True)

    # atomic write for the DataFrame CSV
    fd, tmp_csv = tempfile.mkstemp(dir=dir_, suffix='.csv.tmp')
    with os.fdopen(fd, 'w', newline='') as fh:
        df.to_csv(fh, index=False)
    os.replace(tmp_csv, csv_path)

    meta = {'source_mtimes': source_mtimes, 'computed_at': time.time()}
    fd2, tmp_json = tempfile.mkstemp(dir=dir_, suffix='.json.tmp')
    with os.fdopen(fd2, 'w') as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp_json, meta_path)

    print(f'  [cd_analysis] Cache saved -> {csv_path} ({len(df)} rows)')


# ─── WFG data loading ─────────────────────────────────────────────────────────

def _load_wfg_rows(
    methods: list[str],
    problems: list[str],
    experiment_name: str,
    pop_size: int,
    n_gen: int,
):
    """Yield (algorithm, instance, hv, igd_plus, abs_path) for WFG benchmarks.

    Reads pre-stored ``data['indicators'][-1]`` from each seed pickle.
    WFG benchmarks use an analytically fixed reference point that is identical
    across all methods and runs, so the stored values are directly comparable.
    """
    budget_folder = f'B{n_gen * pop_size}_P{pop_size}'
    for problem in problems:
        root = os.path.join('results', experiment_name, problem, budget_folder)
        for canonical in methods:
            folder = _WFG_FOLDER_MAP.get(canonical, canonical)
            seed_dir = os.path.join(root, folder)
            if not os.path.isdir(seed_dir):
                print(f'  [cd_analysis] WARN: directory not found: {seed_dir}')
                continue
            for fname in sorted(os.listdir(seed_dir)):
                if not fname.endswith('.pkl'):
                    continue
                seed_id  = fname.removeprefix('seed_').removesuffix('.pkl')
                abs_path = os.path.abspath(os.path.join(seed_dir, fname))
                try:
                    with open(abs_path, 'rb') as fh:
                        data = pickle.load(fh)
                except Exception as exc:
                    print(f'  [cd_analysis] WARN: failed to load {abs_path}: {exc}')
                    continue
                indicators = data.get('indicators', [])
                if not indicators:
                    continue
                last = indicators[-1]
                hv  = last.get('hv',       float('nan'))
                igd = last.get('igd_plus', float('nan'))
                instance = f'{problem}_seed_{seed_id}'
                yield canonical, instance, hv, igd, abs_path


# ─── EvoXBench data loading ───────────────────────────────────────────────────

def _load_evox_rows(
    suite: str,
    pids: list[int],
    methods: list[str],
    pop_size: int,
    n_gen: int,
):
    """Yield (algorithm, instance, hv, igd_plus, abs_path) for EvoXBench suites.

    For each PID a shared Pareto approximation is built from ALL method folders
    present in that budget directory (so the reference point is fair across
    methods).  HV and IGD+ are recomputed per seed against that shared reference.
    """
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd_plus import IGDPlus
    from analysis.convergence import build_pareto_approximation

    budget_folder = f'B{n_gen * pop_size}_P{pop_size}'

    for pid in pids:
        root = os.path.join('results', 'evoxbench', suite, f'pid{pid}', budget_folder)
        if not os.path.isdir(root):
            print(f'  [cd_analysis] WARN: results dir not found: {root}')
            continue

        # Build shared Pareto approximation from ALL available method folders
        all_folders = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
        approx = build_pareto_approximation(suite, pid, all_folders, pop_size, n_gen)
        if approx is None:
            print(f'  [cd_analysis] WARN: no Pareto approx for {suite}/pid{pid}, skipping.')
            continue

        ref_point = approx['ref_point']
        pf        = approx['pareto_approx']
        hv_ind    = HV(ref_point=ref_point)
        igd_ind   = IGDPlus(pf)

        for canonical in methods:
            folder   = _EVOX_FOLDER_MAP.get(canonical, canonical)
            seed_dir = os.path.join(root, folder)
            if not os.path.isdir(seed_dir):
                print(f'  [cd_analysis] WARN: directory not found: {seed_dir}')
                continue
            for fname in sorted(os.listdir(seed_dir)):
                if not fname.endswith('.pkl'):
                    continue
                seed_id  = fname.removeprefix('seed_').removesuffix('.pkl')
                abs_path = os.path.abspath(os.path.join(seed_dir, fname))
                try:
                    with open(abs_path, 'rb') as fh:
                        data = pickle.load(fh)
                except Exception as exc:
                    print(f'  [cd_analysis] WARN: {abs_path}: {exc}')
                    continue
                archives = data.get('test_obj_archive', [])
                if not archives:
                    continue
                F = archives[-1]
                if F is None or (hasattr(F, '__len__') and len(F) == 0):
                    continue
                F = np.asarray(F, dtype=float)
                F = np.where(np.isfinite(F), F, 1.0)
                if len(F) == 0:
                    continue
                hv  = float(hv_ind(F))
                igd = float(igd_ind(F))
                instance = f'pid{pid}_seed_{seed_id}'
                yield canonical, instance, hv, igd, abs_path


# ─── main DataFrame builder ───────────────────────────────────────────────────

def build_indicators_df(
    benchmark_name: str,
    methods: list[str],
    pop_size: int = 20,
    n_gen: int = 60,
    force: bool = False,
    *,
    # WFG-specific keyword arguments
    experiment_name: str = 'pymoo_benchmark/2_obj',
    problems: list[str] | None = None,
    # EvoXBench-specific keyword arguments
    pids: list[int] | None = None,
) -> pd.DataFrame:
    """Build (or load from disk cache) the per-seed HV/IGD+ DataFrame.

    Parameters
    ----------
    benchmark_name
        One of ``'wfg'``, ``'c10mop'``, ``'in1kmop'``.
    methods
        Canonical method names (keys in ``_WFG_FOLDER_MAP`` / ``_EVOX_FOLDER_MAP``).
    pop_size, n_gen
        Budget parameters used to locate ``B{budget}_P{pop_size}`` folders.
    force
        Skip the disk cache and rebuild from raw pkl files.
    experiment_name
        WFG only: sub-tree root relative to ``results/``
        (default ``'pymoo_benchmark/2_obj'``).
    problems
        WFG only: list of problem names (default all wfg1-9).
    pids
        EvoXBench only: list of problem IDs (default 1-9).

    Returns
    -------
    pd.DataFrame with columns ``['algorithm', 'instance', 'hv', 'igd_plus']``.

    Notes
    -----
    robustranking requires a value for every (algorithm × instance × objective)
    combination.  After loading, instances that lack data for at least one
    requested method are dropped so the Benchmark is complete.
    """
    budget = n_gen * pop_size

    # ── cache lookup ──────────────────────────────────────────────────────────
    if not force:
        cached_df, _ = _load_cached_df(benchmark_name, budget, pop_size)
        if cached_df is not None:
            cached_methods = sorted(cached_df['algorithm'].unique().tolist())
            if sorted(methods) == cached_methods:
                return cached_df
            print(
                f'  [cd_analysis] Cache method list mismatch '
                f'(cached={cached_methods} vs requested={sorted(methods)}); rebuilding.'
            )

    # ── collect raw rows ──────────────────────────────────────────────────────
    rows: list[tuple]  = []
    source_mtimes: dict = {}

    if benchmark_name == 'wfg':
        _problems = problems or [f'wfg{i}' for i in range(1, 10)]
        for algo, inst, hv, igd, path in _load_wfg_rows(
            methods, _problems, experiment_name, pop_size, n_gen
        ):
            rows.append((algo, inst, hv, igd))
            source_mtimes[path] = os.path.getmtime(path)

    elif benchmark_name in ('c10mop', 'in1kmop'):
        _pids = pids or list(range(1, 10))
        for algo, inst, hv, igd, path in _load_evox_rows(
            benchmark_name, _pids, methods, pop_size, n_gen
        ):
            rows.append((algo, inst, hv, igd))
            source_mtimes[path] = os.path.getmtime(path)

    else:
        raise ValueError(
            f"Unknown benchmark_name {benchmark_name!r}. "
            "Choose from 'wfg', 'c10mop', 'in1kmop'."
        )

    if not rows:
        raise RuntimeError(
            f'No indicator data found for benchmark={benchmark_name!r}. '
            f'Check that result directories exist for the requested methods.'
        )

    df = pd.DataFrame(rows, columns=['algorithm', 'instance', 'hv', 'igd_plus'])

    # ── completeness filtering ────────────────────────────────────────────────
    # Drop rows with NaN indicators first.
    df = df.dropna(subset=['hv', 'igd_plus'])

    # Drop instances not covered by every requested method.
    pivot = df.pivot_table(
        index='instance', columns='algorithm', values='hv', aggfunc='count'
    )
    complete_instances = pivot.dropna().index
    n_total    = df['instance'].nunique()
    n_complete = len(complete_instances)
    if n_complete < n_total:
        print(
            f'  [cd_analysis] Dropping {n_total - n_complete} of {n_total} instances '
            f'missing data for at least one method.'
        )
    df = df[df['instance'].isin(complete_instances)].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            f'After completeness filtering, no instances remain for {benchmark_name!r}. '
            f'Check that all requested methods have result files.'
        )

    print(
        f'  [cd_analysis] {benchmark_name}: {len(df)} rows, '
        f'{df["instance"].nunique()} instances, '
        f'{df["algorithm"].nunique()} algorithms.'
    )

    # ── persist ───────────────────────────────────────────────────────────────
    _save_cached_df(df, source_mtimes, benchmark_name, budget, pop_size)
    return df


# ─── robustranking wrappers ───────────────────────────────────────────────────

def build_robustranking_benchmark(df: pd.DataFrame):
    """Convert an indicators DataFrame into a robustranking Benchmark object."""
    from robustranking.benchmark import Benchmark
    bm = Benchmark()
    bm.from_pandas(
        df,
        algorithm_key='algorithm',
        instance_key='instance',
        objective_keys=['hv', 'igd_plus'],
    )
    return bm


def run_comparison(
    bm,
    alpha: float = 0.05,
    bootstrap_runs: int = 10_000,
    minimise: dict | None = None,
):
    """Build and compute a BootstrapComparison.

    Parameters
    ----------
    bm
        robustranking Benchmark object (from ``build_robustranking_benchmark``).
    alpha
        Significance level for Holm–Bonferroni corrected pairwise tests.
    bootstrap_runs
        Number of bootstrap resamples.
    minimise
        Per-objective minimisation flags.
        Default: ``{'hv': False, 'igd_plus': True}``
        (HV is maximised, IGD+ is minimised).
    """
    from robustranking.comparison.bootstrap_comparison import BootstrapComparison
    if minimise is None:
        minimise = {'hv': False, 'igd_plus': True}
    comp = BootstrapComparison(
        bm,
        minimise=minimise,
        alpha=alpha,
        bootstrap_runs=bootstrap_runs,
        aggregation_method=np.mean,
    )
    comp.compute()
    return comp


# ─── plotting helpers ─────────────────────────────────────────────────────────

def _apply_labels(ax, labels: dict | None) -> None:
    """Rename y-axis tick labels using *labels* dict (in-place, best-effort)."""
    if not labels:
        return
    ax.figure.canvas.draw()   # force tick label rendering
    yticks = ax.get_yticklabels()
    ax.set_yticklabels([labels.get(t.get_text(), t.get_text()) for t in yticks])


def plot_cd_panels(
    benchmarks_dict: dict,
    out_dir: str,
    labels: dict | None = None,
    dpi: int = 150,
    font_size: int = 12,
) -> None:
    """Save one figure per metric (HV, IGD+), with one subplot per benchmark.

    Parameters
    ----------
    benchmarks_dict
        ``{title: pd.DataFrame}``, ordered left-to-right.  Each DataFrame must
        have columns ``['algorithm', 'instance', 'hv', 'igd_plus']``.
    out_dir
        Directory where output PDF files are written.
    labels
        Optional ``{canonical_name: display_name}`` map for y-tick relabelling.
    dpi
        Resolution for rasterised elements.
    font_size
        Base font size in points (applies to all text including robustranking
        internals).  Increase if the figure will be scaled down in the paper.
    """
    import matplotlib.pyplot as plt
    from robustranking.benchmark import Benchmark
    from robustranking.comparison.ranked_comparison import RankedComparison
    from robustranking.utils.plots import plot_critical_difference

    os.makedirs(out_dir, exist_ok=True)
    benchmarks = list(benchmarks_dict.items())
    n = len(benchmarks)

    rc = {
        'font.size':        font_size,
        'axes.titlesize':   font_size + 2,
        'axes.labelsize':   font_size,
        'xtick.labelsize':  font_size - 1,
        'ytick.labelsize':  font_size - 1,
        'figure.titlesize': font_size + 4,
    }

    for metric, metric_label, minimise in [
        ('hv',       'HV',   False),
        ('igd_plus', 'IGD+', True),
    ]:
        metric_full = 'Hypervolume (HV)' if metric == 'hv' else 'IGD+  (Inverted Generational Distance+)'
        direction   = 'higher is better' if not minimise else 'lower is better'
        with plt.rc_context(rc):
            cd_width     = 6.0
            cd_textspace = 1.5
            fig, axes = plt.subplots(1, n, figsize=(cd_width * n, 2.5))
            if n == 1:
                axes = [axes]
            fig.subplots_adjust(wspace=0.0, left=0.02, right=0.98)
            for ax, (title, df) in zip(axes, benchmarks):
                bm = Benchmark()
                bm.from_pandas(
                    df,
                    algorithm_key='algorithm',
                    instance_key='instance',
                    objective_keys=[metric],
                )
                comp = RankedComparison(bm, minimise=minimise, aggregation_method=np.mean)
                comp.compute()
                plot_critical_difference(
                    comp, ax=ax,
                    width=cd_width, textspace=cd_textspace,
                )
                ax.set_title(title)
                _apply_labels(ax, labels)
            fig.suptitle(
                f'Critical Difference Diagram  --  {metric_full}  ({direction})\n'
                f'Mean rank over instances  |  Holm\u2013Bonferroni corrected',
                fontsize=font_size + 1,
                fontweight='bold',
            )
            plt.tight_layout()
            out_path = os.path.join(out_dir, f'cd_ranking_{metric}.pdf')
            plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        print(f'  Saved -> {out_path}')


def plot_cd_grid(
    benchmarks_dict: dict,
    out_dir: str,
    labels: dict | None = None,
    dpi: int = 150,
    font_size: int = 12,
) -> None:
    """Save a single 2 × N grid figure (rows = metrics, cols = benchmarks).

    Parameters
    ----------
    benchmarks_dict
        ``{title: pd.DataFrame}``, ordered left-to-right.  Each DataFrame must
        have columns ``['algorithm', 'instance', 'hv', 'igd_plus']``.
    out_dir
        Directory for the output PDF file.
    labels
        Optional ``{canonical_name: display_name}`` map for y-tick relabelling.
    dpi
        Resolution for rasterised elements.
    font_size
        Base font size in points (applies to all text including robustranking
        internals).  Increase if the figure will be scaled down in the paper.
    """
    import matplotlib.pyplot as plt
    from robustranking.benchmark import Benchmark
    from robustranking.comparison.ranked_comparison import RankedComparison
    from robustranking.utils.plots import plot_critical_difference

    os.makedirs(out_dir, exist_ok=True)
    benchmarks = list(benchmarks_dict.items())
    n = len(benchmarks)
    metrics = [
        ('hv',       'HV',   False),
        ('igd_plus', 'IGD+', True),
    ]

    rc = {
        'font.size':        font_size,
        'axes.titlesize':   font_size,
        'axes.labelsize':   font_size,
        'xtick.labelsize':  font_size - 1,
        'ytick.labelsize':  font_size - 1,
        'figure.titlesize': font_size + 1,
    }

    # Layout: rows = benchmarks, cols = metrics (HV | IGD+)
    n_metrics = len(metrics)

    with plt.rc_context(rc):
        # CD diagram internal geometry (in inches):
        #   width=6, textspace=1.5 → scalewidth=3, text extends ~0.6 beyond
        #   the scale on each side → total overflow ≈ 1.1 in from each edge.
        # Each subplot is sized to exactly match these internal dimensions so
        # no overflow occurs.  Figure width = n_metrics * cd_width (exact).
        cd_width     = 6.0   # passed to __graph_ranks as `width`
        cd_textspace = 2  # passed to __graph_ranks as `textspace`
        subplot_w    = cd_width*2         # one subplot per metric column
        subplot_h    = 6.5               # compact height per row
        fig_w        = subplot_w * n_metrics
        fig_h        = subplot_h * n * 0.7

        fig, axes = plt.subplots(
            n, n_metrics,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )
        # No wspace needed when subplot width == cd_width: text fills axes exactly.
        fig.subplots_adjust(hspace=0.05, wspace=0.0, left=0.1, right=0.95)

        for row, (title, df) in enumerate(benchmarks):
            for col, (metric, metric_label, minimise) in enumerate(metrics):
                ax = axes[row][col]
                bm = Benchmark()
                bm.from_pandas(
                    df,
                    algorithm_key='algorithm',
                    instance_key='instance',
                    objective_keys=[metric],
                )
                comp = RankedComparison(bm, minimise=minimise, aggregation_method=np.mean)
                comp.compute()
                plot_critical_difference(
                    comp, ax=ax,
                    width=cd_width, textspace=cd_textspace,
                )
                # Column headers: metric name (top row only)
                if row == 0:
                    metric_full = ('Hypervolume (HV)  [higher is better]'
                                   if metric == 'hv'
                                   else 'IGD\u207a  [lower is better]')
                    ax.set_title(metric_full)
                _apply_labels(ax, labels)

            # Bold benchmark title rotated on the left of each row
            ax_left = axes[row][0]
            ax_left.text(
                -0.08, 0.5, title,
                transform=ax_left.transAxes,
                ha='right', va='center',
                fontsize=font_size, fontweight='bold',
                rotation=90, clip_on=False,
            )

        fig.suptitle(
            'Critical Difference Diagrams  --  Mean rank over instances  |  Holm\u2013Bonferroni corrected',
            y=1.02,
            fontweight='bold',
            fontsize=font_size + 1,
        )
        out_path = os.path.join(out_dir, 'cd_ranking_grid.pdf')
        plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    print(f'  Saved -> {out_path}')
