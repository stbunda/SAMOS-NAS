"""Stage 3 companion: curated constraint-handling scenario-suite table.

Selects a small, judgement-based set of (space, objective set, constraint)
scenarios spanning tabular/surrogate benchmarks and hard/soft constraint modes,
grouped by budget theme so a tabular and a surrogate benchmark test the same
kind of constraint (the realism-gap comparison). Emits scenarios.{csv,tex} and
scenarios_map.png into the paper dir.

Survival/valuable numbers are read at the suite's 10% operating point straight from
the Stage-2 hv_curve.csv / region_counts.csv; kind and rho from the eligibility
matrix. Re-run after any Stage 2/3 refresh to pick up updated numbers.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import _style as ST          # sets the Agg backend on import
import _cache as IO
import _fronts as FR
import analyzers as AN
import matplotlib.pyplot as plt

# Curated suite: (theme, space, objective_set, constraint, what-we-look-for).
SCENARIOS = [
    ('Compute budget', 'NATS', 'err+params', 'flops',
     'High soft potential at moderate hardness, on an exact reference front.'),
    ('Compute budget', 'ResNet-50D', 'err+flops', 'params',
     'A hard constraint that still leaves moderate soft potential, on a '
     'surrogate front.'),
    ('Compute budget', 'MobileNetV3', 'err+flops', 'params',
     'An extreme, near-independent constraint leaving nothing for soft handling '
     'to recover (control case).'),
    ('Compute budget', 'NB201', 'err+edgegpu_latency', 'params',
     "Near-maximal hardness with the suite's largest soft potential: half of "
     'all architectures are infeasible yet undominated by the feasible front.'),
    ('GPU-latency budget', 'NB201', 'err+params', 'edgegpu_latency',
     "Moderate hardness, limited soft potential. S4's metric pair with roles "
     'swapped: the assignment, not the metrics, sets the geometry.'),
    ('Accelerator intensity', 'NB201', 'err+eyeriss_latency', 'eyeriss_arithmetic_intensity',
     "The suite's only lower-bound budget, a utilisation floor: hard, but the "
     'low-intensity side is already dominated.'),
]

# raw metric name -> interpretable display name
DISPLAY = {
    'err': 'Error', 'params': '#Params', 'flops': 'FLOPs', 'latency': 'Latency',
    'edgegpu_latency': 'GPU (Jetson TX2) latency', 'edgegpu_energy': 'GPU (Jetson TX2) energy',
    'eyeriss_latency': 'Eyeriss latency', 'eyeriss_energy': 'Eyeriss energy',
    'eyeriss_arithmetic_intensity': 'Arith. intensity',
    'h1_latency': 'GPU (RTX 4090) latency', 'h2_latency': 'Atlas AI processor latency',
    'h1_energy_consumption': 'GPU (RTX 4090) energy', 'h2_energy_consumption': 'Atlas AI processor energy',
}

# concise constraint labels for the scatter (the full device names above are
# for the table; on the figure they would overrun, especially when merged).
FIG_NAME = {
    'edgegpu_latency': 'GPU-lat (Jetson)', 'h1_latency': 'GPU-lat (4090)',
    'h2_latency': 'Atlas lat', 'eyeriss_latency': 'Eyeriss lat',
    'edgegpu_energy': 'GPU-en (Jetson)', 'h1_energy_consumption': 'GPU-en (4090)',
    'h2_energy_consumption': 'Atlas en', 'eyeriss_energy': 'Eyeriss en',
    'eyeriss_arithmetic_intensity': 'Arith. int.', 'flops': 'FLOPs',
    'params': '#Params', 'latency': 'Latency',
}

# metric columns to report: (kind, percentile, csv label, tex header). `hard` =
# fraction of the reference front removed by a p% budget (= 1 - survivability),
# the figure x-axis; `soft` = infeasible non-dominated mass under a p% budget,
# the figure y-axis. Reported at 10% only -- the suite's single operating point.
# 20% was dropped because soft potential collapses above 10% (S5 falls from 0.17
# to 0.01 by 15%), while hardness barely moves: five of the eight scenarios have
# identical hardness at 10/15/20 since their reference fronts are 21-25 points.
_PCTS = [
    ('hard', 10, 'hard@10%', r'hard$_{10}$'),
    ('soft', 10, 'soft@10%', r'soft$_{10}$'),
]
_REGION_CATS = ['infeasible-dominated', 'feasible-dominated',
                'infeasible-valuable', 'feasible-nondominated']

# constraint metric -> budget type; each budget owns one colour, shared by the
# selected scenarios and the greyed candidate cloud (selection shows via size +
# black border, not colour). Colours are Okabe-Ito.
BUDGET_COLOR = {'compute': '#56B4E9', 'latency': '#E69F00',
                'accelerator intensity': '#009E73', 'energy': '#CC79A7'}
BUDGET_ORDER = ['compute', 'latency', 'energy', 'accelerator intensity']

# spaces excluded from the suite, and therefore from the candidate cloud too --
# showing candidates that are not under consideration would misread as options.
EXCLUDE_SPACES = {'MoSegNAS'}


def _budget(constraint):
    c = constraint.lower()
    if 'arithmetic_intensity' in c:
        return 'accelerator intensity'
    if 'latency' in c:
        return 'latency'
    if 'energy' in c:
        return 'energy'
    if 'flops' in c or 'param' in c:
        return 'compute'
    return 'compute'


def _disp(name):
    return DISPLAY.get(name, name.replace('_', ' '))


def _fig_name(c):
    return FIG_NAME.get(c, _disp(c))


def _disp_objs(objset):
    return ', '.join(_disp(o) for o in objset.split('+'))


def _hv_col(hv, objset, c, pct, col):
    if hv is None:
        return np.nan
    m = hv[(hv.objset == objset) & (hv.constraint == c) & (hv.percentile == pct)]
    return float(m[col].iloc[0]) if len(m) else np.nan


def _surv(hv, objset, c, pct):
    return _hv_col(hv, objset, c, pct, 'survival_frac')


def _val(region, objset, c, pct):
    if region is None:
        return np.nan
    m = region[(region.objset == objset) & (region.constraint == c) & (region.percentile == pct)]
    if not len(m):
        return np.nan
    n = float(m[_REGION_CATS].iloc[0].sum())
    return float(m['infeasible-valuable'].iloc[0]) / n if n > 0 else np.nan


def _maybe_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _ref_front_size(full_root, space, objset, cache):
    """Size of the unconstrained reference front |PF|, the denominator behind
    survival_frac. Not stored by (e), so it is recomputed here along exactly
    (e)'s path -- same normalization, and the same deterministic FRONT_CAP
    subsample on non-exact spaces -- or the two would not be comparable."""
    key = (space, objset)
    if key in cache:
        return cache[key]
    sdir = os.path.join(full_root, space)
    try:
        manifest = json.load(open(os.path.join(sdir, 'manifest.json')))
        data = IO.apply_row_filter(pd.read_parquet(os.path.join(sdir, 'samples.parquet')), space)
    except (OSError, ValueError):
        cache[key] = np.nan
        return np.nan
    cols = list(objset.split('+'))
    Fn, _ = AN._normalize_min(data[cols].to_numpy(float), cols, manifest['families'])
    exact = manifest.get('complete', manifest['enumerable'])
    idx = np.arange(len(Fn)) if exact else AN._subsample(len(Fn), AN.FRONT_CAP)
    cache[key] = int(len(FR.first_front(Fn[idx])))
    return cache[key]


def build(full_root, paper_dir):
    matrix = pd.read_csv(os.path.join(full_root, 'eligibility_matrix.csv'))
    hv_cache, rg_cache, pf_cache = {}, {}, {}
    rows = []
    for i, (theme, space, objset, constraint, looking_for) in enumerate(SCENARIOS):
        hit = matrix[(matrix.space == space) & (matrix.objective_set == objset)
                     & (matrix.constraint == constraint)]
        if not len(hit):
            print(f'WARNING: scenario not in matrix -> {space} {objset} {constraint}')
            continue
        r = hit.iloc[0]
        hv = hv_cache.setdefault(space, _maybe_csv(os.path.join(full_root, space, 'hv_curve.csv')))
        rg = rg_cache.setdefault(space, _maybe_csv(os.path.join(full_root, space, 'region_counts.csv')))
        row = {
            'sid': f'S{i + 1}',
            'theme': theme,
            'kind': 'tabular' if r['exact_reference_set'] else 'surrogate',
            'space': space, 'objectives': objset, 'constraint': constraint,
            'objectives_disp': _disp_objs(objset), 'constraint_disp': _disp(constraint),
        }
        for kind, pct, label, _ in _PCTS:
            row[label] = (1.0 - _surv(hv, objset, constraint, pct)) if kind == 'hard' \
                else _val(rg, objset, constraint, pct)
        # realised feasible fraction: nominally 10%, but a tied constraint
        # sweeps in extra architectures at the threshold (S4 lands at 14%).
        row['feas@10%'] = _hv_col(hv, objset, constraint, 10, 'feas_ratio')
        # front sizes: |PF| also sets the granularity of hard@10%, which can
        # only move in steps of 1/|PF|.
        row['n_ref_front'] = _ref_front_size(full_root, space, objset, pf_cache)
        row['n_front@10%'] = _hv_col(hv, objset, constraint, 10, 'n_front')
        row['rho'] = r['max_rho_with_objective']
        row['what_we_look_for'] = looking_for
        rows.append(row)
    df = pd.DataFrame(rows)

    os.makedirs(paper_dir, exist_ok=True)
    df.to_csv(os.path.join(paper_dir, 'scenarios.csv'), index=False)
    _write_tex(df, os.path.join(paper_dir, 'scenarios.tex'))
    _plot_map(df, os.path.join(paper_dir, 'scenarios_map.png'))
    # context variant: same map with every other single-constraint candidate
    # greyed in behind the selected suite.
    bg = _background(matrix, {(s, o, c) for _, s, o, c, _ in SCENARIOS})
    _plot_map(df, os.path.join(paper_dir, 'scenarios_map_context.png'), bg=bg)
    # per-scenario headline figures (one panel per S1..S8)
    _plot_scenario_regions(df, full_root, os.path.join(paper_dir, 'headline2_region.png'))
    _plot_scenario_curves(df, full_root, os.path.join(paper_dir, 'headline2_curves.png'))
    show = ['theme', 'kind', 'space', 'feas@10%', 'n_ref_front', 'n_front@10%'] \
        + [l for _, _, l, _ in _PCTS] + ['rho']
    print(df[show].to_string(index=False))
    print(f'\nWrote scenarios.{{csv,tex}} + scenarios_map{{,_context}}.png to {paper_dir}')
    return df


def _background(matrix, exclude):
    """Single-constraint, two-objective candidate scenarios not in the suite,
    placed on the same (hardness, payoff) axes for the greyed context layer."""
    m = matrix[(matrix.n_constraints == 1)
               & (matrix.objective_set.str.count(r'\+') == 1)
               & (~matrix.space.isin(EXCLUDE_SPACES))].copy()
    keys = list(zip(m.space, m.objective_set, m.constraint))
    m = m[[k not in exclude for k in keys]]
    m['hardness'] = 1.0 - m['survival_frac_10pct']
    m['payoff'] = m['valuable_mass_10pct']
    m['kind'] = m['exact_reference_set'].map({True: 'tabular', False: 'surrogate'})
    return m.dropna(subset=['hardness', 'payoff'])


def _region_cache(sdir, objset, c):
    """Path to the (objset, constraint) region cache, falling back to the legacy
    constraint-only name for spaces not re-run since the key fix."""
    new = AN._region_cache_path(sdir, objset, c)
    if os.path.exists(new):
        return new
    legacy = os.path.join(sdir, f'_region_cache_{c}.npz')
    if os.path.exists(legacy):
        _, extra = IO.load_categorical(legacy)
        if tuple(extra['objset'].tolist()) == tuple(objset):
            return legacy
    return None


def _grid(n, panel_w, panel_h, ncol=4):
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes, nrow


def _plot_scenario_regions(df, full_root, path):
    """One panel per scenario: objective-space region classification at a 10%
    budget (feasible / infeasible x dominated / non-dominated)."""
    fig, axes, nrow = _grid(len(df), 3.7, 3.4)
    for ax, (_, r) in zip(axes, df.iterrows()):
        space, c = r['space'], r['constraint']
        objset = tuple(r['objectives'].split('+'))
        sdir = os.path.join(full_root, space)
        npz = _region_cache(sdir, objset, c)
        drawn = False
        if npz is not None:
            region, _ = IO.load_categorical(npz)
            data = IO.apply_row_filter(pd.read_parquet(os.path.join(sdir, 'samples.parquet')), space)
            if len(region) == len(data):
                fams = json.load(open(os.path.join(sdir, 'manifest.json')))['families']
                Fn, _ = AN._normalize_min(data[list(objset)].to_numpy(float), list(objset), fams)
                AN._region_scatter(space, Fn, region, objset, c, ax=ax, legend=False)
                drawn = True
        if not drawn:
            ax.text(0.5, 0.5, 'no region cache', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{r['sid']}: {space}\n{r['objectives_disp']} | {_fig_name(c)}",
                     fontsize=8.5)
    fig.suptitle('Scenario region classification at a 10% budget', fontsize=13)
    ST.apply_row_spacing(fig, nrow)
    # one shared legend below the panels (as in headline_region.png)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=10,
                   markerscale=2, bbox_to_anchor=(0.5, -0.03))
    ST.savefig(fig, path)


def _plot_scenario_curves(df, full_root, path):
    """One panel per scenario: hard and soft versus the feasibility budget."""
    pcts = [1, 5, 10, 20, 50]
    hv_cache, rg_cache = {}, {}
    fig, axes, nrow = _grid(len(df), 3.3, 3.0)
    for ax, (_, r) in zip(axes, df.iterrows()):
        space, objset, c = r['space'], r['objectives'], r['constraint']
        hv = hv_cache.setdefault(space, _maybe_csv(os.path.join(full_root, space, 'hv_curve.csv')))
        rg = rg_cache.setdefault(space, _maybe_csv(os.path.join(full_root, space, 'region_counts.csv')))
        hard = [1.0 - _surv(hv, objset, c, p) for p in pcts]
        soft = [_val(rg, objset, c, p) for p in pcts]
        ax.plot(pcts, hard, marker='o', color='#333333', label='hard')
        ax.plot(pcts, soft, marker='s', color=BUDGET_COLOR['latency'], label='soft')
        ax.set_xscale('log'); ax.set_xticks(pcts); ax.set_xticklabels(pcts)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{r['sid']}: {space}", fontsize=9)
        ax.set_xlabel('feasibility budget (%)', fontsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc='center right')
    fig.suptitle('Scenario hard and soft versus feasibility budget', fontsize=13)
    ST.apply_row_spacing(fig, nrow)
    ST.savefig(fig, path)


def _tex_escape(s):
    return str(s).replace('%', r'\%').replace('#', r'\#')


def _write_tex(df, path):
    # Theme is a column-spanning subheader row (not a column); the whole tabular
    # is wrapped in \resizebox so it fits the text width. Needs graphicx +
    # booktabs. Switch \textwidth -> \linewidth if the target is single-column.
    metric_heads = [r'feas$_{10}$', r'$|PF|$', r'$|PF_{10}|$'] \
        + [h for _, _, _, h in _PCTS] + [r'$\rho_{max}$']
    heads = ['', 'Kind', 'Space', 'Objectives', 'Constraint'] + metric_heads \
        + [r'Scenario description']
    ncol = len(heads)
    colspec = 'lll' + 'p{1.2cm}p{3.4cm}' + 'r' * len(metric_heads) + 'p{6cm}'
    lines = [
        r'\begin{table*}[htbp]', r'\centering',
        r'\caption{Selected constraint handling scenarios, grouped by budget. All '
        r'metrics are at a 10\% feasibility budget, each constraint held at its own '
        r'10th percentile: $\mathrm{feas}_{10}$ is the resulting share of the space '
        r'that is actually feasible, $|PF|$ the size of the unconstrained Pareto '
        r'front and $|PF_{10}|$ that of the constrained one (so $\mathrm{hard}_{10}$ '
        r'moves only in steps of $1/|PF|$), $\mathrm{hard}_{10}$ the '
        r'share of the unconstrained Pareto front the constraint removes, '
        r'$\mathrm{soft}_{10}$ the share of architectures that are infeasible yet '
        r'undominated by the feasible front, and $\rho_{max}$ the largest absolute '
        r'Spearman correlation between a constraint and an objective. '
        r'For each scenario, a small description states what it tests.}',
        r'\label{tab:scenarios-suite}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{' + colspec + '}', r'\toprule',
        ' & '.join(heads) + r' \\', r'\midrule',
    ]
    prev = None
    for _, r in df.iterrows():
        if r['theme'] != prev:
            if prev is not None:
                lines.append(r'\midrule')
            lines.append(r'\multicolumn{%d}{l}{\textbf{%s}} \\' % (ncol, r['theme']))
            prev = r['theme']
        cells = [r['sid'], _tex_escape(r['kind']), _tex_escape(r['space']),
                 _tex_escape(r['objectives_disp']), _tex_escape(r['constraint_disp'])]
        f = r['feas@10%']
        cells.append(f'{100 * float(f):.1f}\\%' if pd.notna(f) else '--')
        for label in ('n_ref_front', 'n_front@10%'):
            cells.append(f'{int(r[label])}' if pd.notna(r[label]) else '--')
        for _, _, label, _h in _PCTS:
            v = r[label]
            cells.append(f'{float(v):.2f}' if pd.notna(v) else '--')
        cells.append(f'{float(r["rho"]):.2f}' if pd.notna(r['rho']) else '--')
        cells.append(_tex_escape(r['what_we_look_for']))
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}', '']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def _plot_map(df, path, bg=None):
    """Scatter each scenario by how hard the constraint is (x = share of the
    Pareto front it removes at a 10% budget; further right = harder) and how
    much a soft method could recover (y = infeasible solutions the feasible
    front does not dominate; higher = more payoff). Both axes at the suite's
    10% operating point. Marker = tabular/surrogate; colour = theme;
    a dashed line links the tabular and surrogate benchmark within a theme, so
    the realism gap reads as the length of that line. Constraints on the same
    benchmark that coincide (differ only in correlation, not an axis here) share
    one marker, labelled with both names and both rho values. If `bg` is given,
    every other candidate scenario is drawn greyed behind the selected suite."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import _style as ST

    d = df.copy()
    d['hardness'] = d['hard@10%']   # further right = harder constraint
    d['payoff'] = d['soft@10%']
    theme_order = list(dict.fromkeys(d['theme']))
    markers = {'tabular': 'o', 'surrogate': 's'}

    # merge coincident points on the same benchmark into one marker; colour is by
    # budget type (shared with the candidate cloud). The label lists every
    # constraint + its rho.
    d['_hx'], d['_py'] = d['hardness'].round(2), d['payoff'].round(2)
    pts = []
    for _, g in d.groupby(['space', '_hx', '_py'], sort=False):
        g = g.sort_values('rho')
        cons = ' / '.join(_fig_name(c) for c in g['constraint'])
        pts.append(dict(space=g['space'].iloc[0], budget=_budget(g['constraint'].iloc[0]),
                        kind=g['kind'].iloc[0], x=g['hardness'].iloc[0], y=g['payoff'].iloc[0],
                        cons=cons, obj=g['objectives_disp'].iloc[0], sids=set(g['sid']),
                        rho=', '.join(f'{v:.2f}' for v in g['rho'])))
    pts = pd.DataFrame(pts)

    fig, ax = plt.subplots(figsize=(6.9, 5.2))
    # candidate cloud: same budget colours, small + faded + no border. The
    # selected suite reads as the choices purely by size + black border.
    if bg is not None and len(bg):
        bg = bg.copy()
        bg['budget'] = bg['constraint'].map(_budget)
        bg['elig'] = bg['verdict'].str.startswith('eligible')
        for budget in BUDGET_ORDER:
            for kind, mk in markers.items():
                for elig, alpha, sz in [(True, 0.65, 34), (False, 0.22, 24)]:
                    pv = bg[(bg['budget'] == budget) & (bg['kind'] == kind) & (bg['elig'] == elig)]
                    if len(pv):
                        ax.scatter(pv['hardness'], pv['payoff'], s=sz, marker=mk,
                                   c=BUDGET_COLOR[budget], alpha=alpha, edgecolor='none', zorder=0)
    # realism-gap dashed links: tabular<->surrogate sharing a budget
    for budget in pts['budget'].unique():
        gt = pts[pts['budget'] == budget]
        for _, a in gt[gt['kind'] == 'tabular'].iterrows():
            for _, b in gt[gt['kind'] == 'surrogate'].iterrows():
                ax.plot([a['x'], b['x']], [a['y'], b['y']], color=BUDGET_COLOR[budget],
                        lw=1.0, ls='--', alpha=0.5, zorder=1)
    # markers are labelled with their scenario IDs; the table carries
    # the detail. Merged points list every ID they cover.
    placed = []
    for _, r in pts.iterrows():
        ax.scatter(r['x'], r['y'], s=110, c=BUDGET_COLOR[r['budget']], marker=markers[r['kind']],
                   edgecolor='black', linewidth=0.8, zorder=3)
        dx, ha = ((-9, 'right') if r['x'] >= 0.5 else (9, 'left'))
        dy = 8
        for (px, py) in placed:            # push below if it would sit on a placed label
            if abs(r['x'] - px) < 0.09 and abs(r['y'] - py) < 0.06:
                dy = -16
        placed.append((r['x'], r['y']))
        ax.annotate(', '.join(sorted(r['sids'])), (r['x'], r['y']),
                    textcoords='offset points', xytext=(dx, dy), ha=ha,
                    fontsize=12, fontweight='bold')
    ytop = max(0.78, pts['y'].max() + 0.15,
               (bg['payoff'].max() + 0.1) if bg is not None and len(bg) else 0)
    ax.set_xlim(-0.05, 1.12); ax.set_ylim(-0.06, ytop)
    # two-line axis captions: larger first line, italic second line. On the
    # y-axis the italic description sits to the right of the main label.
    ax.tick_params(labelsize=13)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.text(0.5, -0.12, r'Constraint hardness  (hard$_{10}$)', transform=ax.transAxes,
            ha='center', va='top', fontsize=16)
    ax.text(0.5, -0.195, 'share of Pareto front removed at a 10% budget',
            transform=ax.transAxes, ha='center', va='top', fontsize=12, style='italic')
    ax.text(-0.175, 0.5, r'Soft potential  (soft$_{10}$)', transform=ax.transAxes,
            ha='center', va='center', rotation=90, fontsize=16)
    ax.text(-0.105, 0.5, 'infeasible solutions the feasible front does not dominate',
            transform=ax.transAxes, ha='center', va='center', rotation=90, fontsize=12,
            style='italic')
    # legend: budget colours, then the realism-gap link, then benchmark shapes.
    present = [bt for bt in BUDGET_ORDER
               if bt in set(pts['budget']) or (bg is not None and len(bg)
                                               and bt in set(bg['budget']))]
    handles = [Patch(facecolor=BUDGET_COLOR[bt], edgecolor='black', label=f'{bt} budget')
               for bt in present]
    handles += [Line2D([0], [0], color='#555555', ls='--', lw=1.2, label='realism gap')]
    handles += [Line2D([0], [0], marker=markers[k], ls='', mfc='gray', mec='black', label=k)
                for k in ('tabular', 'surrogate')]
    has_bg = bg is not None and len(bg)
    if has_bg:
        handles += [Patch(facecolor='0.3', alpha=0.7, edgecolor='none', label='other: eligible'),
                    Patch(facecolor='0.3', alpha=0.22, edgecolor='none', label='other: redundant')]
    ax.legend(handles=handles, title='budget / benchmark', fontsize=9, title_fontsize=10,
              loc='upper left', ncol=2 if has_bg else 1)
    # ax.set_title('Scenario map: constraint hardness vs soft handling payoff', fontsize=15)
    ST.savefig(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_root', default='results2/constraint_analysis/full')
    ap.add_argument('--paper_dir', default='results2/constraint_analysis/paper')
    args = ap.parse_args()
    build(args.full_root, args.paper_dir)


if __name__ == '__main__':
    main()
