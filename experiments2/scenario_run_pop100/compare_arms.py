"""experiments2/scenario_run_pop100/compare_arms.py -- population-size arm comparison.

Compares the two budget-matched arms of the scenario_run campaign:

  pop20  : pop_size=20,  60 outer generations   (results2/scenario_run)
  pop100 : pop_size=100, 12 outer generations   (results2/scenario_run_pop100)

Both spend the same 1200 real evaluations per run and differ only in how that
budget is split between population and generations -- samos' surrogate-side
inner GA is pinned to 200 in BOTH arms (run_scenario.py --inner_pop_size), so
the outer population is the only moving part. Any difference here is therefore
attributable to the pop/gen split rather than to spend or to surrogate search
width.

Input is each arm's per-seed ``scenario_metrics.csv`` (written by
analyse_scenarios.py), paired on (sid, mode, method, handler, seed), plus each
arm's ``plot_data.pkl`` for the critical-difference ranks. Run
analyse_scenarios.py on both roots first.

Two things about the comparison that are not obvious from the numbers:

1. CRITICAL DIFFERENCE IS RECOMPUTED, NOT READ. analyse_scenarios.py's
   ``critical_difference.csv`` is EMPTY for both arms: its COMPARISON_ROWS
   carries two SSA-NSGA-II configs and ``_cd_matrix`` drops any block missing
   any config, so an arm without SSA-NSGA-II has no complete block left. This
   script drops those two rows and calls the analysis module's own
   ``compute_critical_difference``, giving the same statistic (Friedman +
   Nemenyi, force_mode='nonparametric', 12 blocks) over the 8 configs both
   arms ran. The reported CD applies to every rank shift in the table -- a
   shift smaller than CD is not separable.

2. hv_auc IS COMPARABLE BUT CONSERVATIVE. It integrates against EVALUATIONS
   over a common [0, budget] window anchored at (0, 0) (analyse_scenarios
   ``_hv_auc``), so pop100 is fairly charged for its later first reading
   rather than having the early stretch skipped. But the trapezoid samples
   that window at 13 points instead of 61, and a trapezoid under a concave
   curve understates area, so pop100's anytime deficit is an UPPER bound.

Examples
--------
  python experiments2/scenario_run_pop100/compare_arms.py
  python experiments2/scenario_run_pop100/compare_arms.py --no_csv
  python experiments2/scenario_run_pop100/compare_arms.py \\
      --pop20_analysis results2/scenario_run/analysis_4methods \\
      --output_dir /tmp/arm_compare
"""

import argparse
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_SCEN_DIR  = os.path.join(_REPO_ROOT, 'experiments2', 'scenario_run')
for _p in (_REPO_ROOT, _SCEN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_POP100 = os.path.join('results2', 'scenario_run_pop100', 'analysis')
_DEFAULT_POP20  = os.path.join('results2', 'scenario_run_pop100',
                                'analysis_pop20_baseline')

# Methods the pop100 arm ran (SSA-NSGA-II and IOC-SAMO-COBRA are not in it).
METHODS = ['random', 'nsga2', 'samos', 'ctaea']
# Methods carrying the full 7-handler row -- the only ones with a handler axis.
ROW_METHODS = ['nsga2', 'samos']
KEY = ['sid', 'mode', 'method', 'handler', 'seed']
ALPHA = 0.05

# (column, higher_is_better) for the headline panel.
HEADLINE_METRICS = [('hv_run', True), ('hv_ratio', True), ('hv_auc', True),
                    ('best_err', False), ('n_feasible_final', True)]


# ══════════════════════════════════════════════════════════════════════════
# paired statistics
# ══════════════════════════════════════════════════════════════════════════

def paired(sub, col):
    """(median_pop20, median_pop100, p, n) for one paired column.

    Wilcoxon signed-rank on the per-seed differences. p is NaN when the arms
    are identical or too few pairs survive -- both read as 'tie' downstream
    rather than as a missing test."""
    x = sub[f'{col}_20'].to_numpy(dtype=float)
    y = sub[f'{col}_100'].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.allclose(x, y):
        return (np.median(x) if len(x) else np.nan,
                np.median(y) if len(y) else np.nan, np.nan, len(x))
    return np.median(x), np.median(y), wilcoxon(x, y).pvalue, len(x)


def verdict(md20, md100, p, higher_better=True):
    """'pop20' | 'pop100' | 'tie' at ALPHA."""
    if not np.isfinite(p) or p >= ALPHA:
        return 'tie'
    better100 = (md100 > md20) if higher_better else (md100 < md20)
    return 'pop100' if better100 else 'pop20'


def _row(scope, name, col, sub, higher_better=True):
    """One comparison row as a flat dict."""
    md20, md100, p, n = paired(sub, col)
    return dict(scope=scope, name=name, metric=col, n_pairs=n,
                median_pop20=md20, median_pop100=md100,
                delta=md100 - md20,
                delta_pct=100 * (md100 - md20) / md20 if md20 else np.nan,
                p=p, significant=bool(np.isfinite(p) and p < ALPHA),
                better=verdict(md20, md100, p, higher_better))


# ══════════════════════════════════════════════════════════════════════════
# comparison panels
# ══════════════════════════════════════════════════════════════════════════

def compute_overall(m):
    """Headline: every metric in HEADLINE_METRICS, all cells pooled."""
    return [_row('overall', 'all', col, m, hb) for col, hb in HEADLINE_METRICS]


def compute_by_method(m):
    """hv_run per method, then per (method, mode)."""
    rows = []
    for meth in METHODS:
        sub = m[m['method'] == meth]
        if len(sub):
            rows.append(_row('method', meth, 'hv_run', sub))
    for meth in METHODS:
        for mode in ('hard', 'soft'):
            sub = m[(m['method'] == meth) & (m['mode'] == mode)]
            if len(sub):
                rows.append(_row('method_mode', f'{meth}/{mode}', 'hv_run', sub))
    return rows


def compute_by_handler(m):
    """hv_run per (method, handler) for the full-row methods."""
    rows = []
    for meth in ROW_METHODS:
        for h in sorted(m[m['method'] == meth]['handler'].unique()):
            sub = m[(m['method'] == meth) & (m['handler'] == h)]
            rows.append(_row('handler', f'{meth}/{h}', 'hv_run', sub))
    return rows


def compute_handler_ranks(a20, a100):
    """Mean handler rank across the 12 (scenario, mode) blocks, per arm.

    Rank 1 = best per-block median hv_run. The Spearman correlation between
    the arms' rank vectors is the answer to 'does the handler ranking survive
    the pop/gen split' -- carried on every row of that method."""
    rows = []
    for meth in ROW_METHODS:
        ranks = {}
        for arm, df in (('pop20', a20), ('pop100', a100)):
            sub = df[df['method'] == meth]
            piv = (sub.groupby(['sid', 'mode', 'handler'])['hv_run'].median()
                      .unstack('handler'))
            ranks[arm] = piv.rank(axis=1, ascending=False).mean()
        tbl = pd.DataFrame(ranks).sort_values('pop20')
        rho, p = spearmanr(tbl['pop20'], tbl['pop100'])
        for h, r in tbl.iterrows():
            rows.append(dict(method=meth, handler=h,
                             mean_rank_pop20=r['pop20'],
                             mean_rank_pop100=r['pop100'],
                             shift=r['pop20'] - r['pop100'],
                             spearman_rho=rho, spearman_p=p))
    return rows


def _head_to_head(df):
    """{(sid, mode, handler): 'nsga2' | 'samos' | 'tie'} within one arm."""
    out = {}
    for (sid, mode, h), sub in df.groupby(['sid', 'mode', 'handler']):
        n = sub[sub['method'] == 'nsga2'].set_index('seed')['hv_run']
        s = sub[sub['method'] == 'samos'].set_index('seed')['hv_run']
        idx = n.index.intersection(s.index)
        if len(idx) < 5:
            continue
        n, s = n.loc[idx].to_numpy(), s.loc[idx].to_numpy()
        if np.allclose(n, s):
            out[(sid, mode, h)] = 'tie'
            continue
        p = wilcoxon(n, s).pvalue
        out[(sid, mode, h)] = ('tie' if p >= ALPHA else
                               ('samos' if np.median(s) > np.median(n) else 'nsga2'))
    return out


def compute_head_to_head(a20, a100):
    """Per-cell nsga2-vs-samos verdict in each arm, and whether it changed.

    'reversed' marks a sign flip (nsga2 <-> samos); a change through 'tie' is
    a significance-threshold move, not a reversal, and is marked 'changed'."""
    v20, v100 = _head_to_head(a20), _head_to_head(a100)
    rows = []
    for k in sorted(set(v20) & set(v100)):
        sid, mode, h = k
        change = ('unchanged' if v20[k] == v100[k] else
                  ('reversed' if {v20[k], v100[k]} == {'nsga2', 'samos'} else 'changed'))
        rows.append(dict(sid=sid, mode=mode, handler=h,
                         winner_pop20=v20[k], winner_pop100=v100[k],
                         change=change))
    return rows


def compute_cost(m):
    """What the split costs in realised generations, feasibility and waste.

    h1-rejection and hard-gate DOE retries spend evaluations on infeasible
    probes, so their realised generation count sits below the nominal
    budget/pop_size and is the direct read on how badly a handler scales with
    population size."""
    cols = [c for c in ('n_gen', 'n_eval_realised', 'n_feasible_final',
                        'm3_final_infeasible_waste', 'h1_rejected_count')
            if f'{c}_20' in m.columns]
    rows = []
    for meth in ROW_METHODS:
        for h in sorted(m[m['method'] == meth]['handler'].unique()):
            sub = m[(m['method'] == meth) & (m['handler'] == h)]
            row = dict(method=meth, handler=h)
            for c in cols:
                row[f'{c}_pop20'] = sub[f'{c}_20'].median()
                row[f'{c}_pop100'] = sub[f'{c}_100'].median()
            rows.append(row)
    return rows


def compute_critical_difference(pop20_pkl, pop100_pkl):
    """Friedman + Nemenyi mean ranks per arm over the configs both arms ran.

    Calls analyse_scenarios' own implementation with the SSA-NSGA-II rows
    dropped from COMPARISON_ROWS -- see this module's docstring for why the
    shipped critical_difference.csv is empty."""
    import analyse_scenarios as A

    A.COMPARISON_ROWS = [r for r in A.COMPARISON_ROWS if r[0] != 'ssansga2']
    arms = {}
    for arm, path in (('pop20', pop20_pkl), ('pop100', pop100_pkl)):
        with open(path, 'rb') as fh:
            arms[arm] = A.compute_critical_difference(pickle.load(fh)['metrics_rows'])

    rows = []
    for key in sorted(set(arms['pop20']) & set(arms['pop100'])):
        scope, metric = key
        r20, r100 = arms['pop20'][key], arms['pop100'][key]
        order = r20['ranks'].sort_values().index
        rho, p = spearmanr(r20['ranks'][order].to_numpy(),
                           r100['ranks'][order].to_numpy())
        for lbl in order:
            rows.append(dict(
                scope=scope, metric=metric, config=lbl,
                mean_rank_pop20=r20['ranks'][lbl],
                mean_rank_pop100=r100['ranks'][lbl],
                shift=r20['ranks'][lbl] - r100['ranks'][lbl],
                median_pop20=r20['frame'][lbl].median(),
                median_pop100=r100['frame'][lbl].median(),
                cd=r20['cd'], n_blocks=r20['n_blocks'],
                omnibus_p_pop20=r20['pvalue'], omnibus_p_pop100=r100['pvalue'],
                spearman_rho=rho, spearman_p=p))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# reporting
# ══════════════════════════════════════════════════════════════════════════

def _banner(title):
    print('\n' + '=' * 96)
    print(title)
    print('=' * 96)


def print_comparison(rows, title, label='name'):
    """Comparison rows (from _row) as an aligned table."""
    _banner(title)
    print(f"{label:<26} {'metric':<17} {'pop20':>10} {'pop100':>10} "
          f"{'delta':>10} {'p':>10}  better")
    for r in rows:
        print(f"{r[label]:<26} {r['metric']:<17} {r['median_pop20']:>10.4f} "
              f"{r['median_pop100']:>10.4f} {r['delta']:>+10.4f} "
              f"{r['p']:>10.2e}  {r['better']}")


def print_handler_ranks(rows):
    _banner('HANDLER RANKING -- mean rank over the 12 (scenario, mode) blocks '
            '(1 = best hv_run)')
    for meth in ROW_METHODS:
        sub = [r for r in rows if r['method'] == meth]
        if not sub:
            continue
        print(f'\n  -- {meth} --')
        print(f"     {'handler':<20} {'pop20':>7} {'pop100':>7} {'shift':>7}")
        for r in sub:
            print(f"     {r['handler']:<20} {r['mean_rank_pop20']:>7.2f} "
                  f"{r['mean_rank_pop100']:>7.2f} {r['shift']:>+7.2f}")
        print(f"     Spearman between arms: rho={sub[0]['spearman_rho']:+.3f} "
              f"p={sub[0]['spearman_p']:.4f}")


def print_head_to_head(rows):
    _banner('nsga2 vs samos PER (scenario, mode, handler) -- does the verdict hold?')
    tally = pd.Series([f"{r['winner_pop20']} -> {r['winner_pop100']}"
                       for r in rows]).value_counts()
    print(f'  {len(rows)} cells compared\n')
    print(f"  {'pop20 -> pop100':<26} {'cells':>6}")
    for k, c in tally.items():
        a, b = k.split(' -> ')
        print(f'  {k:<26} {c:>6}' + ('' if a == b else '   <-- changed'))
    same = sum(1 for r in rows if r['change'] == 'unchanged')
    rev = sum(1 for r in rows if r['change'] == 'reversed')
    print(f'\n  unchanged: {same}/{len(rows)} ({100 * same / len(rows):.0f}%)')
    print(f'  reversed (sign flip, not via tie): {rev}/{len(rows)}')


def print_cost(rows):
    _banner('COST SIDE -- median per cell, pop20 -> pop100')
    cols = [c[:-6] for c in rows[0] if c.endswith('_pop20')]
    print(f"{'method':<8} {'handler':<20} " + ' '.join(f'{c[:13]:>16}' for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            a, b = r[f'{c}_pop20'], r[f'{c}_pop100']
            cells.append(f'{a:>6.0f} ->{b:>7.0f}' if np.isfinite(a) and np.isfinite(b)
                         else f'{"-":>16}')
        print(f"{r['method']:<8} {r['handler']:<20} " + ' '.join(cells))


def print_critical_difference(rows):
    _banner('CRITICAL DIFFERENCE -- mean ranks per arm (shifts below CD are '
            'not separable)')
    for key in dict.fromkeys((r['scope'], r['metric']) for r in rows):
        sub = [r for r in rows if (r['scope'], r['metric']) == key]
        print(f'\n  -- scope={key[0]}  metric={key[1]}  '
              f"n_blocks={sub[0]['n_blocks']}  CD={sub[0]['cd']:.3f} --")
        print(f"     {'config':<21} {'pop20':>7} {'pop100':>7} {'shift':>7} "
              f"{'med20':>9} {'med100':>9}")
        for r in sub:
            print(f"     {r['config']:<21} {r['mean_rank_pop20']:>7.2f} "
                  f"{r['mean_rank_pop100']:>7.2f} {r['shift']:>+7.2f} "
                  f"{r['median_pop20']:>9.4f} {r['median_pop100']:>9.4f}")
        best20 = min(sub, key=lambda r: r['mean_rank_pop20'])['config']
        best100 = min(sub, key=lambda r: r['mean_rank_pop100'])['config']
        print(f"     Spearman between arms: rho={sub[0]['spearman_rho']:+.3f} "
              f"p={sub[0]['spearman_p']:.4f}")
        print(f'     top-ranked: pop20={best20} | pop100={best100}'
              + ('' if best20 == best100 else '   <-- CHANGED'))


def write_csv(rows, path):
    with open(path, 'w', newline='') as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f'[compare] wrote {len(rows)} rows -> {path}')


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════

def load_arm(analysis_dir):
    """One arm's per-seed metrics, restricted to the shared method set."""
    path = os.path.join(analysis_dir, 'scenario_metrics.csv')
    if not os.path.exists(path):
        raise SystemExit(f'missing {path} -- run analyse_scenarios.py on this '
                         f'arm first.')
    df = pd.read_csv(path)
    return df[df['method'].isin(METHODS)].copy()


def main(args):
    a20, a100 = load_arm(args.pop20_analysis), load_arm(args.pop100_analysis)
    m = a20.merge(a100, on=KEY, suffixes=('_20', '_100'))
    if m.empty:
        raise SystemExit('no paired rows -- do the two arms share scenarios, '
                         'methods and seeds?')

    print(f'[compare] pop20  {args.pop20_analysis}: {len(a20)} rows')
    print(f'[compare] pop100 {args.pop100_analysis}: {len(a100)} rows')
    print(f'[compare] paired: {len(m)} per-seed rows over '
          f'{m.groupby(KEY[:4]).ngroups} cells')
    unpaired = len(a20) - len(m)
    if unpaired:
        print(f'[compare] {unpaired} pop20 row(s) unpaired (missing seeds) -- '
              f'dropped from every paired test')

    overall = compute_overall(m)
    by_method = compute_by_method(m)
    by_handler = compute_by_handler(m)
    ranks = compute_handler_ranks(a20, a100)
    h2h = compute_head_to_head(a20, a100)
    cost = compute_cost(m)

    print_comparison(overall, 'OVERALL -- budget-matched effect of '
                              'pop 20x60 -> pop 100x12', label='scope')
    print_comparison(by_method, 'PER METHOD and PER METHOD x MODE -- final '
                                'feasible HV')
    print_comparison(by_handler, 'PER HANDLER (within method) -- final feasible HV')
    print_handler_ranks(ranks)
    print_head_to_head(h2h)
    print_cost(cost)

    cd = []
    pkls = [os.path.join(d, 'plot_data.pkl')
            for d in (args.pop20_analysis, args.pop100_analysis)]
    if all(os.path.exists(p) for p in pkls):
        cd = compute_critical_difference(*pkls)
        if cd:
            print_critical_difference(cd)
    else:
        print('\n[compare] critical difference skipped: plot_data.pkl missing '
              'in one arm.')

    if args.no_csv:
        return 0
    os.makedirs(args.output_dir, exist_ok=True)
    for name, rows in (('overall.csv', overall),
                       ('method_effect.csv', by_method),
                       ('handler_effect.csv', by_handler),
                       ('handler_ranks.csv', ranks),
                       ('head_to_head.csv', h2h),
                       ('cost.csv', cost),
                       ('critical_difference.csv', cd)):
        write_csv(rows, os.path.join(args.output_dir, name))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pop100_analysis', default=_DEFAULT_POP100,
                    help='Analysis dir of the pop-100 arm.')
    p.add_argument('--pop20_analysis', default=_DEFAULT_POP20,
                    help='Analysis dir of the pop-20 baseline. Must cover the '
                         'SAME methods as the pop-100 arm -- an analysis that '
                         'also carries SSA-NSGA-II makes the rank statistics '
                         'incomparable.')
    p.add_argument('--output_dir', default=None,
                    help='Default: {pop100_analysis}/comparison.')
    p.add_argument('--no_csv', action='store_true',
                    help='Print the tables without writing CSVs.')
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.pop100_analysis, 'comparison')
    sys.exit(main(args))
