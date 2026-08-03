"""Stage 3 - collate cached analyzer outputs into the eligibility matrix and the
two-tier LaTeX deliverables.

Tier A (appendix): \\input every generated table + \\includegraphics every figure.
Tier B (paper): eligibility_matrix.{csv,tex}, a scenario-definition table, two
headline figures, a compact correlation panel, and (if any antagonistic pair
survives) a joint-ratio table.

Reads only cached CSV/manifests; no recompute, no EvoXBench.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _adapter as A
import _cache as IO
import _style as ST
import analyzers as AN
import matplotlib.pyplot as plt

# Verdict thresholds (exposed as raw columns too, so authors can re-threshold).
TAIL_MIN = 100          # >= this many samples behind the 1% quantile => tail-trustworthy
REDUNDANT_RHO = 0.8     # |Spearman(constraint, an objective)| at/above => near-duplicate
SHAPING_SURVIVAL = 0.8  # survival fraction at 20% below this => front-reshaping
SHAPING_VALUABLE = 0.05 # infeasible-valuable mass fraction at 10% above => shaping
TOP_N_PER_SPACE = 5     # paper table cap: most interesting rows per space
VERDICT_RANK = {'eligible-shaping': 0, 'eligible-mild': 1, 'redundant': 2, 'poorly-sampled': 3}


def _spaces(full_root):
    # Only spaces whose Stage 2 finished (quantile table is the last-but-figures
    # artifact every downstream column needs).
    return [d for d in sorted(os.listdir(full_root))
            if os.path.isdir(os.path.join(full_root, d))
            and os.path.exists(os.path.join(full_root, d, 'manifest.json'))
            and os.path.exists(os.path.join(full_root, d, 'quantile_thresholds.csv'))]


def build_matrix(full_root):
    rows = []
    for space in _spaces(full_root):
        sdir = os.path.join(full_root, space)
        manifest = json.load(open(os.path.join(sdir, 'manifest.json')))
        metrics = manifest['metrics']
        exact = manifest.get('complete', manifest['enumerable'])
        tabular = A.is_tabular(space)
        quant = pd.read_csv(os.path.join(sdir, 'quantile_thresholds.csv'))
        spear = pd.read_csv(os.path.join(sdir, 'corr_spearman.csv'), index_col=0)
        hv = _maybe_csv(os.path.join(sdir, 'hv_curve.csv'))
        region = _maybe_csv(os.path.join(sdir, 'region_counts.csv'))
        neigh = _maybe_csv(os.path.join(sdir, 'neighbour_feasibility.csv'))

        for objset in A.objective_sets(metrics):
            objkey = '+'.join(objset)
            for c_tuple in A.constraint_candidates(metrics, objset):
                c = '+'.join(c_tuple)
                # quant/spear are keyed per single metric (unaffected by pairing);
                # for a joint pair take the more conservative (min tail count /
                # max correlation) across its members.
                tail_n1 = min(
                    int(quant[(quant.metric == m) & (quant.percentile == 1)]['n_behind'].iloc[0])
                    for m in c_tuple)
                rho = max((abs(spear.loc[m, o]) for m in c_tuple for o in objset
                           if m in spear.index and o in spear.columns and np.isfinite(spear.loc[m, o])),
                          default=np.nan)
                surv20 = _lookup(hv, objkey, c, 20, 'survival_frac')
                hvfrac20 = _lookup(hv, objkey, c, 20, 'hv_frac_of_unconstrained')
                # 10% is the scenario suite's operating point; 20% stays as the
                # broader screening level the verdict thresholds are tuned on.
                surv10 = _lookup(hv, objkey, c, 10, 'survival_frac')
                val10 = _valuable_frac(region, objkey, c, 10)
                repair50 = _repair(neigh, c) if len(c_tuple) == 1 else np.nan

                well_sampled = tail_n1 >= TAIL_MIN
                redundant = np.isfinite(rho) and rho >= REDUNDANT_RHO
                shaping = ((np.isfinite(surv20) and surv20 < SHAPING_SURVIVAL)
                           or (np.isfinite(val10) and val10 > SHAPING_VALUABLE))
                if not well_sampled:
                    verdict = 'poorly-sampled'
                elif redundant:
                    verdict = 'redundant'
                elif shaping:
                    verdict = 'eligible-shaping'
                else:
                    verdict = 'eligible-mild'

                rows.append({
                    'space': space, 'objective_set': objkey, 'constraint': c,
                    'n_constraints': len(c_tuple),
                    'tabular': tabular, 'exact_reference_set': exact,
                    'tail_n_at_1pct': tail_n1, 'well_sampled_tail': well_sampled,
                    'max_rho_with_objective': round(rho, 3) if np.isfinite(rho) else np.nan,
                    'survival_frac_10pct': _r(surv10),
                    'survival_frac_20pct': _r(surv20), 'hv_frac_20pct': _r(hvfrac20),
                    'valuable_mass_10pct': _r(val10),
                    'repair_neigh_feas_50pct': _r(repair50),
                    'verdict': verdict,
                })
    return pd.DataFrame(rows)


def _maybe_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _lookup(hv, objkey, c, pct, col):
    if hv is None or len(hv) == 0:
        return np.nan
    m = hv[(hv.objset == objkey) & (hv.constraint == c) & (hv.percentile == pct)]
    return float(m[col].iloc[0]) if len(m) else np.nan


_REGION_CATS = ['infeasible-dominated', 'feasible-dominated',
                'infeasible-valuable', 'feasible-nondominated']


def _valuable_frac(region, objkey, c, pct):
    # normalize by the analyzed population (row's own four region categories),
    # so it tracks any analysis-time row filtering rather than the raw n_rows.
    if region is None or len(region) == 0:
        return np.nan
    m = region[(region.objset == objkey) & (region.constraint == c) & (region.percentile == pct)]
    if not len(m):
        return np.nan
    n = float(m[_REGION_CATS].iloc[0].sum())
    return float(m['infeasible-valuable'].iloc[0]) / n if n > 0 else np.nan


def _repair(neigh, c):
    if neigh is None or len(neigh) == 0:
        return np.nan
    m = neigh[(neigh.metric == c) & (neigh.percentile == 50)]
    return float(m['mean_neighbour_feasibility'].iloc[0]) if len(m) else np.nan


def _r(v):
    return round(float(v), 3) if v is not None and np.isfinite(v) else np.nan


# ─────────────────────────────── Tier A appendix ─────────────────────────────

def build_appendix(full_root, paper_dir):
    lines = ['% Auto-generated appendix. Compile from results2/constraint_analysis/full/.',
             '\\section{Constraint-analysis appendix}']
    for space in _spaces(full_root):
        sdir = os.path.join(full_root, space)
        rel = os.path.relpath(sdir, paper_dir).replace('\\', '/')
        lines.append(f'\\subsection{{{space}}}')
        for tex in sorted(glob.glob(os.path.join(sdir, '*.tex'))):
            lines.append(f'\\input{{{rel}/{os.path.basename(tex)}}}')
        for png in sorted(glob.glob(os.path.join(sdir, '*.png'))):
            cap = os.path.splitext(os.path.basename(png))[0].replace('_', ' ')
            lines += ['\\begin{figure}[htbp]\\centering',
                      f'\\includegraphics[width=\\linewidth]{{{rel}/{os.path.basename(png)}}}',
                      f'\\caption{{{space}: {cap}.}}\\end{{figure}}']
    path = os.path.join(full_root, 'appendix.tex')
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    return path


# ─────────────────────────────── Tier B paper ───────────────────────────────

def _top_per_space(matrix, n):
    # most interesting rows first (shaping > mild > redundant > poorly-sampled),
    # well-sampled tail as tiebreak; caps each space at n rows for the paper table.
    m = matrix.assign(_rank=matrix.verdict.map(VERDICT_RANK))
    m = m.sort_values(['space', '_rank', 'tail_n_at_1pct'], ascending=[True, True, False])
    return m.groupby('space', group_keys=False).head(n).drop(columns='_rank')


def _grid_axes(n, panel_w, panel_h, ncol_max=4):
    ncol = min(ncol_max, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes, nrow


def _find_region_cache(sdir, objset, c):
    """New (objset, c)-keyed path first; else the legacy `_region_cache_{c}.npz`
    (pre-fix naming, only correct when its stored objset happens to match --
    spaces never rerun under the new key still only have the legacy file)."""
    new_path = AN._region_cache_path(sdir, objset, c)
    if os.path.exists(new_path):
        return new_path
    legacy_path = os.path.join(sdir, f'_region_cache_{c}.npz')
    if os.path.exists(legacy_path):
        _, extra = IO.load_categorical(legacy_path)
        if tuple(extra['objset'].tolist()) == tuple(objset):
            return legacy_path
    return None


def build_headline_figures(full_root, paper_dir, matrix):
    """Redraw the four headline figures as one small-multiple grid each, one
    panel per search space, using every space's own most-eligible row (not a
    single "representative" space). Redrawn from cached CSV/parquet/npz --
    same cached artifacts the per-space Tier A figures were built from."""
    top1 = _top_per_space(matrix, 1).reset_index(drop=True)
    spaces = top1['space'].tolist()

    for value_col, fname, ylab in [
        ('hv_frac_of_unconstrained', 'headline_hv.png', 'HV(tau) / HV(inf)'),
        ('survival_frac', 'headline_survival.png', 'surviving-front fraction')]:
        fig, axes, nrow = _grid_axes(len(spaces), 3.2, 2.6)
        for ax, row in zip(axes, top1.itertuples()):
            hv = pd.read_csv(os.path.join(full_root, row.space, 'hv_curve.csv'))
            sub = hv[(hv.objset == row.objective_set) & (hv.constraint == row.constraint)]
            sub = sub.sort_values('feas_ratio')
            ax.plot(sub['feas_ratio'], sub[value_col], marker='o', color=ST.OKABE_ITO[5])
            ax.set_xlabel('feasible ratio'); ax.set_ylabel(ylab)
            ax.set_title(f'{row.space}\n{row.objective_set} | {row.constraint}', fontsize=7)
        fig.suptitle('Most-eligible constraint scenario per search space')
        ST.apply_row_spacing(fig, nrow)
        ST.savefig(fig, os.path.join(paper_dir, fname))

    fig, axes, nrow = _grid_axes(len(spaces), 3.4, 3.0)
    for ax, row in zip(axes, top1.itertuples()):
        sdir = os.path.join(full_root, row.space)
        objset_tuple = tuple(row.objective_set.split('+'))
        npz_path = _find_region_cache(sdir, objset_tuple, row.constraint)
        msg = 'no 2D scatter cached\nfor this scenario'
        if npz_path is not None:
            region, _ = IO.load_categorical(npz_path)
            df = IO.apply_row_filter(pd.read_parquet(os.path.join(sdir, 'samples.parquet')),
                                     row.space)
            if len(region) != len(df):
                # samples.parquet grew (HPC download landed) after this cache
                # was written -- stale, not corrupt; needs a Stage 2 rerun.
                msg = 'stale region cache\n(samples.parquet grew --\nrerun Stage 2 for this space)'
            else:
                manifest = json.load(open(os.path.join(sdir, 'manifest.json')))
                Mobj = df[list(objset_tuple)].to_numpy(dtype=float)
                Fn, _ = AN._normalize_min(Mobj, list(objset_tuple), manifest['families'])
                AN._region_scatter(row.space, Fn, region, objset_tuple, row.constraint,
                                   ax=ax, legend=False)
                msg = None
        if msg:
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=7, transform=ax.transAxes)
            ax.set_title(row.space, fontsize=7)
    fig.suptitle('Region classification (most-eligible scenario per search space)')
    ST.apply_row_spacing(fig, nrow)
    # one shared legend for the whole grid (category colours/markers are the
    # same across every panel) instead of repeating it in each subplot.
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   fontsize=8, markerscale=2, bbox_to_anchor=(0.5, -0.02))
    ST.savefig(fig, os.path.join(paper_dir, 'headline_region.png'))

    # panel height grows with the largest metric count in this batch: spaces
    # like NB201/MoSegNAS (7-8 metrics) need much more room for tick labels
    # than the 3-4 metric spaces, or their labels bleed into neighbouring panels.
    max_metrics = max(len(pd.read_csv(os.path.join(full_root, s, 'corr_spearman.csv'),
                                       index_col=0))
                       for s in spaces)
    fig, axes, nrow = _grid_axes(len(spaces), 3.0, 0.5 * max_metrics + 1.5)
    for ax, space in zip(axes, spaces):
        spear = pd.read_csv(os.path.join(full_root, space, 'corr_spearman.csv'), index_col=0)
        AN._heatmap(spear, title=space, ax=ax)
    fig.suptitle('Spearman metric correlations per search space')
    fig.subplots_adjust(hspace=0.9, wspace=0.6, top=0.93)
    ST.savefig(fig, os.path.join(paper_dir, 'headline_corr.png'))


def build_paper(full_root, paper_dir, matrix):
    os.makedirs(paper_dir, exist_ok=True)
    paper_matrix = _top_per_space(matrix, TOP_N_PER_SPACE)
    paper_matrix.to_csv(os.path.join(paper_dir, 'eligibility_matrix.csv'), index=False)
    IO.write_tex(paper_matrix.set_index(['space', 'objective_set', 'constraint']),
                 os.path.join(paper_dir, 'eligibility_matrix.tex'),
                 caption=f'Eligibility matrix: top {TOP_N_PER_SPACE} (space, objective set, '
                         'constraint) rows per search space by verdict interest -- tail '
                         'adequacy, redundancy, reference-set availability, repair cost, and '
                         'verdict. Full matrix in the appendix.',
                 label='tab:eligibility')

    # representative space: the one with the most eligible-shaping verdicts
    # (ties broken by eligible-mild count); falls back to the first space if
    # every verdict everywhere is redundant/poorly-sampled.
    elig = matrix[matrix.verdict.isin(['eligible-shaping', 'eligible-mild'])]
    if len(elig):
        counts = elig.groupby('space').verdict.value_counts().unstack(fill_value=0)
        for col in ('eligible-shaping', 'eligible-mild'):
            if col not in counts.columns:
                counts[col] = 0
        rep = counts.sort_values(['eligible-shaping', 'eligible-mild'], ascending=False).index[0]
    else:
        rep = matrix.space.iloc[0]

    # headline figures: one panel per search space, each built from that
    # space's own most-eligible row (top_per_space with n=1), not a single
    # "representative" space.
    build_headline_figures(full_root, paper_dir, matrix)

    # scenario-definition table from (c) for the representative space
    quant = pd.read_csv(os.path.join(full_root, rep, 'quantile_thresholds.csv'))
    piv = quant.pivot(index='metric', columns='percentile', values='threshold')
    IO.write_tex(piv, os.path.join(paper_dir, 'scenario_table.tex'),
                 caption=f'Scenario-seed thresholds tau (feasibility ratio = percentile) '
                         f'for {rep}.', label='tab:scenarios')

    # joint-ratio table only if any antagonistic pair (ratio < 1) survives anywhere
    antagonistic = []
    for space in _spaces(full_root):
        jp = os.path.join(full_root, space, 'joint_ratio.csv')
        if os.path.exists(jp):
            j = pd.read_csv(jp)
            j = j[j.ratio < 1]
            if len(j):
                j.insert(0, 'space', space)
                antagonistic.append(j)
    if antagonistic:
        jt = pd.concat(antagonistic, ignore_index=True).sort_values('ratio')
        IO.write_tex(jt, os.path.join(paper_dir, 'joint_ratio_antagonistic.tex'),
                     caption='Antagonistic constraint pairs (observed joint feasibility '
                             '$<$ product of marginals) -- the only pairs justifying a '
                             'multi-constraint scenario.', label='tab:joint-antag', index=False)

    # section.tex wiring it all together
    figs = []
    for f, cap in [('headline_hv.png', 'Constraint cost: HV$(\\tau)$ relative to the '
                    'unconstrained front, most-eligible scenario per search space.'),
                   ('headline_region.png', 'Region classification in objective space at the '
                    '10\\% feasibility level, most-eligible scenario per search space.'),
                   ('headline_corr.png', 'Spearman metric correlations per search space.')]:
        if os.path.exists(os.path.join(paper_dir, f)):
            figs += ['\\begin{figure}[htbp]\\centering',
                     f'\\includegraphics[width=\\columnwidth]{{{f}}}',
                     f'\\caption{{{cap}}}\\end{{figure}}']
    sec = ['% Auto-generated paper section.',
           '\\section{Constraint eligibility}',
           '\\input{eligibility_matrix.tex}',
           '\\input{scenario_table.tex}']
    sec += figs
    if antagonistic:
        sec.append('\\input{joint_ratio_antagonistic.tex}')
    open(os.path.join(paper_dir, 'section.tex'), 'w', encoding='utf-8').write('\n'.join(sec) + '\n')
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_root', default='results2/constraint_analysis/full')
    ap.add_argument('--paper_dir', default='results2/constraint_analysis/paper')
    args = ap.parse_args()
    matrix = build_matrix(args.full_root)
    matrix.to_csv(os.path.join(args.full_root, 'eligibility_matrix.csv'), index=False)
    app = build_appendix(args.full_root, args.paper_dir)
    rep = build_paper(args.full_root, args.paper_dir, matrix)
    # curated constraint-handling scenario suite (reads the matrix just written)
    import build_scenarios as SC
    SC.build(args.full_root, args.paper_dir)
    print(f'Matrix rows: {len(matrix)}; verdict counts:')
    print(matrix['verdict'].value_counts().to_string())
    print(f'Appendix: {app}')
    print(f'Representative space for Tier B: {rep}')
    print(matrix.to_string(index=False))


if __name__ == '__main__':
    main()
