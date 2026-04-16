"""analyze_cd_plots.py — Critical difference (CI ranking) plots using robustranking.

Produces confidence-interval ranking plots for three benchmark groups:

  Synthetic  — WFG 1–9
  Tabular    — C10MOP 1–7
  Surrogate  — C10MOP 8–9 + IN1KMOP 1–9

Collects final per-seed HV and IGD+ values, caches them as DataFrames, and
feeds them into robustranking's BootstrapComparison.

Run examples
------------
# All three groups with default settings (n_gen=60, pop_size=20)
python analyze_cd_plots.py

# Synthetic only, force cache rebuild
python analyze_cd_plots.py --benchmarks wfg --force

# Custom method list and budget
python analyze_cd_plots.py \\
    --benchmarks c10mop surrogate \\
    --methods random nsga2 mosmac gpsaf samos \\
    --pop_size 20 --n_gen 60

# Reduce bootstrap samples for a quick preview
python analyze_cd_plots.py --bootstrap_runs 1000

Output
------
results/cd_analysis/
    wfg/B1200_P20/indicators_df.csv          (cached DataFrame)
    wfg/B1200_P20/indicators_meta.json       (staleness metadata)
    c10mop/B1200_P20/…
    surrogate/B1200_P20/…
    cd_ranking_hv.pdf                        (one subplot per benchmark)
    cd_ranking_igd_plus.pdf
    cd_ranking_grid.pdf                      (2 × N combined figure)
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from analysis.cd_analysis import (
    DISPLAY_LABELS,
    build_indicators_df,
    build_robustranking_benchmark,
    run_comparison,
    plot_cd_panels,
    plot_cd_grid,
)

# ─── defaults ─────────────────────────────────────────────────────────────────
_METHODS = ['random', 'parego', 'mosmac', 'nsga2', 'gpsaf', 
            'ssa-nsga2',
            'samos']

# _DEFAULT_METHODS = ['random', 'nsga2', 'parego', 'mosmac', 'gpsaf', 'ssa-nsga2', 'samos']
# _DEFAULT_BENCHMARKS = ['wfg', 'c10mop', 'in1kmop']

# Human-readable benchmark titles used in plot headers
_BENCHMARK_TITLES = {
    'wfg':      'Synthetic\nWFG 1–9',
    'c10mop':   'Tabular\nC10MOP 1–7',
    'in1kmop':  'IN1KMOP 1–9',
    'surrogate': 'Surrogate\nC10MOP 8–9\nIN1KMOP 1–9',
}


# ─── argument parsing ─────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Generate critical difference (CI ranking) plots via robustranking.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--benchmarks', nargs='+',
        default=['wfg', 'c10mop', 'surrogate'],
        choices=['wfg', 'c10mop', 'in1kmop', 'surrogate'],
        metavar='BENCH',
        help=(
            'Benchmark groups to analyse. '
            'wfg=Synthetic (WFG 1–9), '
            'c10mop=Tabular (C10MOP 1–7 by default), '
            'surrogate=C10MOP 8–9 + IN1KMOP 1–9, '
            'in1kmop=IN1KMOP 1–9 standalone.'
        ),
    )
    p.add_argument(
        '--methods', nargs='+',
        default=_METHODS,
        metavar='METHOD',
        help=(
            'Canonical method names. '
            'Supported: random, nsga2, parego, mosmac, gpsaf, ssa-nsga2, samos.'
        ),
    )
    p.add_argument('--pop_size',  type=int, default=20,  help='Population size.')
    p.add_argument('--n_gen',     type=int, default=60,  help='Number of generations.')
    p.add_argument(
        '--experiment_name', default='pymoo_benchmark/2_obj',
        help='WFG sub-tree root under results/ (used only for --benchmarks wfg).',
    )
    p.add_argument(
        '--problems', nargs='+', default=None,
        metavar='PROBLEM',
        help='WFG problem names. Default: wfg1 … wfg9.',
    )
    p.add_argument(
        '--pids', nargs='+', type=int, default=None,
        metavar='PID',
        help='EvoXBench problem IDs. Default: 1 … 9.',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Ignore on-disk cache and rebuild DataFrames from raw pkl files.',
    )
    p.add_argument(
        '--alpha', type=float, default=0.05,
        help='Significance level for Holm–Bonferroni pairwise tests.',
    )
    p.add_argument(
        '--bootstrap_runs', type=int, default=10_000,
        help='Number of bootstrap resamples.',
    )
    p.add_argument(
        '--output_dir', default=os.path.join('results', 'cd_analysis'),
        help='Directory where plots are saved.',
    )
    p.add_argument(
        '--no_grid', action='store_true',
        help='Skip the combined 2×N grid figure.',
    )
    p.add_argument(
        '--font_size', type=int, default=23,
        help='Base font size (pt) for all plot text. Increase when the figure will be scaled down in a paper.',
    )
    return p.parse_args(argv)


# ─── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = _parse_args(argv)

    comparisons: dict = {}   # title -> BootstrapComparison

    for bench in args.benchmarks:
        print(f'\n{"=" * 60}')
        print(f'  Benchmark: {bench}')
        print(f'{"=" * 60}')

        # ── build / load DataFrame ────────────────────────────────────────────
        kwargs = {}
        if bench == 'wfg':
            kwargs['experiment_name'] = args.experiment_name
            if args.problems:
                kwargs['problems'] = args.problems
        elif bench == 'c10mop':
            # Tabular group: C10MOP 1–7 only (8–9 belong to Surrogate group)
            kwargs['pids'] = args.pids if args.pids else list(range(1, 8))
        elif bench in ('in1kmop', 'surrogate'):
            if args.pids:
                kwargs['pids'] = args.pids

        df = build_indicators_df(
            benchmark_name=bench,
            methods=args.methods,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            force=args.force,
            **kwargs,
        )

        print(f'\n  DataFrame shape : {df.shape}')
        print(f'  Algorithms      : {sorted(df["algorithm"].unique().tolist())}')
        print(f'  Instances       : {df["instance"].nunique()}')
        print(f'  HV  range       : [{df["hv"].min():.4f}, {df["hv"].max():.4f}]')
        print(f'  IGD+ range      : [{df["igd_plus"].min():.4f}, {df["igd_plus"].max():.4f}]')

        # ── build robustranking Benchmark ─────────────────────────────────────
        bm = build_robustranking_benchmark(df)
        print(f'\n  robustranking Benchmark stats:\n{bm.show_stats().to_string()}')

        # ── run bootstrap comparison ──────────────────────────────────────────
        print(f'\n  Running BootstrapComparison '
              f'(alpha={args.alpha}, bootstrap_runs={args.bootstrap_runs}) …')
        comp = run_comparison(
            bm,
            alpha=args.alpha,
            bootstrap_runs=args.bootstrap_runs,
        )

        # ── print ranking tables ──────────────────────────────────────────────
        for metric, metric_label in [('hv', 'HV'), ('igd_plus', 'IGD+')]:
            ranking = comp.get_ranking()
            ci_df   = comp.get_confidence_intervals()
            ci_obj  = ci_df[ci_df.index.get_level_values(1) == metric].copy()
            ci_obj.index = ci_obj.index.get_level_values(0)

            print(f'\n  -- {bench.upper()} ranking by {metric_label} --')
            print(f'  {"Algorithm":<20}  {"Group":>5}  {"Median":>8}  '
                  f'{"95%-CI lb":>10}  {"95%-CI ub":>10}')
            for algo, row in ranking.iterrows():
                group = row.get('group', '—')
                ci    = ci_obj.loc[algo] if algo in ci_obj.index else None
                med   = f'{ci["median"]:.4f}' if ci is not None else '—'
                lb    = f'{ci["lb"]:.4f}'     if ci is not None else '—'
                ub    = f'{ci["ub"]:.4f}'     if ci is not None else '—'
                display = DISPLAY_LABELS.get(algo, algo)
                print(f'  {display:<20}  {str(group):>5}  {med:>8}  {lb:>10}  {ub:>10}')

        comparisons[_BENCHMARK_TITLES[bench]] = df

    # ── save plots ────────────────────────────────────────────────────────────
    if not comparisons:
        print('\nNo benchmarks produced comparisons; nothing to plot.')
        return

    print(f'\n{"=" * 60}')
    print('  Saving plots ...')
    print(f'{"=" * 60}')

    plot_cd_panels(
        comparisons,
        out_dir=args.output_dir,
        labels=DISPLAY_LABELS,
        font_size=args.font_size,
    )

    if not args.no_grid:
        plot_cd_grid(
            comparisons,
            out_dir=args.output_dir,
            labels=DISPLAY_LABELS,
            font_size=args.font_size,
        )


if __name__ == '__main__':
    main()
