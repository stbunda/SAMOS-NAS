"""Stage 2 analyzers (a)-(g).

Each analyzer is a pure function of (space, df, manifest, out_dir): it reads the
cached frame, writes numeric CSV + LaTeX + figure(s) under out_dir, and returns
a compact dict of headline numbers for Stage-3 collation. No analyzer touches
EvoXBench.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import _adapter as A
import _fronts as FR
import _cache as IO
import _style as ST
import matplotlib.pyplot as plt

QG = A.QUANTILE_GRID
TAU = A.TAU_RATIOS


# ─────────────────────────── (a) marginal summaries ──────────────────────────

def analyze_a(space, df, manifest, out_dir):
    """Per-metric univariate summary: quantile grid, min/max/mean, non-positive
    counts, plus a histogram grid (log-scaled for positive resource metrics)."""
    metrics = IO.metric_cols(manifest)
    fams = manifest['families']
    rows = []
    for m in metrics:
        x = df[m].to_numpy(dtype=float)
        finite = np.isfinite(x)
        xv = x[finite]
        n_nonpos = int(np.sum(xv <= 0))
        fam = fams[m]
        log_ok = fam == 'resource' and n_nonpos == 0
        qs = np.percentile(xv, QG) if xv.size else [np.nan] * len(QG)
        rows.append({
            'metric': m, 'family': fam, 'direction': A.metric_direction(m),
            'n_nonnull': int(finite.sum()), 'n_nonpos': n_nonpos,
            'log_plot': log_ok, 'min': xv.min() if xv.size else np.nan,
            'max': xv.max() if xv.size else np.nan,
            'mean': xv.mean() if xv.size else np.nan,
            **{f'q{p}': qs[i] for i, p in enumerate(QG)},
        })
    tab = pd.DataFrame(rows).set_index('metric')
    IO.write_csv(tab, os.path.join(out_dir, 'marginals.csv'))
    IO.write_tex(tab.drop(columns=['log_plot']), os.path.join(out_dir, 'marginals.tex'),
                 caption=f'Marginal metric summaries -- {space}.',
                 label=f'tab:marg-{space}')

    ncol = min(4, len(metrics))
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        x = df[m].to_numpy(dtype=float)
        xv = x[np.isfinite(x)]
        log_ok = fams[m] == 'resource' and (xv > 0).all()
        if log_ok:
            ax.hist(np.log10(xv), bins=60, color=ST.OKABE_ITO[5])
            ax.set_xlabel(f'log10({m})')
        else:
            ax.hist(xv, bins=60, color=ST.OKABE_ITO[1])
            ax.set_xlabel(m)
        ax.set_ylabel('count')
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(f'{space} marginals (n={len(df):,})')
    ST.apply_row_spacing(fig, nrow)
    ST.savefig(fig, os.path.join(out_dir, 'marginals.png'))
    return {'n_rows': int(len(df)), 'metrics': metrics,
            'nonpos_metrics': [r['metric'] for r in rows if r['n_nonpos'] > 0]}


# ─────────────────────────── (b) pairwise correlations ───────────────────────

def _corr_matrix(M, cols, method):
    k = len(cols)
    out = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i, k):
            xi, xj = M[:, i], M[:, j]
            good = np.isfinite(xi) & np.isfinite(xj)
            if good.sum() < 3 or np.std(xi[good]) == 0 or np.std(xj[good]) == 0:
                r = np.nan
            elif method == 'pearson':
                r = pearsonr(xi[good], xj[good])[0]
            else:
                r = spearmanr(xi[good], xj[good])[0]
            out[i, j] = out[j, i] = r
    return pd.DataFrame(out, index=cols, columns=cols)


def _heatmap(mat, path=None, title='', ax=None):
    # ax=None: standalone figure saved to `path` (Stage 2 usage). ax given:
    # draw onto it and let the caller save the enclosing figure (Stage 3
    # headline grids, redrawn from the same cached correlation CSV).
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(0.6 * len(mat) + 2, 0.6 * len(mat) + 1.5))
    im = ax.imshow(mat.to_numpy(), cmap=ST.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(mat))); ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(len(mat)):
        for j in range(len(mat)):
            v = mat.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if abs(v) < 0.6 else 'white')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    if own_fig:
        ST.savefig(fig, path)


def analyze_b(space, df, manifest, out_dir):
    """Pairwise metric correlations: Pearson (log-resource columns) and Spearman
    (all metrics, and Spearman restricted to the unconstrained Pareto front) --
    surfaces redundant/aligned metrics via heatmaps."""
    metrics = IO.metric_cols(manifest)
    fams = manifest['families']
    M = df[metrics].to_numpy(dtype=float)
    # drop constant columns
    keep = [i for i, m in enumerate(metrics) if np.nanstd(M[:, i]) > 0]
    cols = [metrics[i] for i in keep]
    M = M[:, keep]
    # Pearson on log-resource columns (strictly-positive resources only).
    Mlog = M.copy()
    for j, m in enumerate(cols):
        if fams[m] == 'resource':
            col = M[:, j]
            if np.all(col[np.isfinite(col)] > 0):
                Mlog[:, j] = np.log10(col)
    pear = _corr_matrix(Mlog, cols, 'pearson')
    spear = _corr_matrix(M, cols, 'spearman')

    # Spearman on the (all-metric) first front too.
    ff = FR.first_front(_normalize_min(M, cols, fams)[0])
    spear_pf = _corr_matrix(M[ff], cols, 'spearman') if len(ff) > 3 else spear * np.nan

    IO.write_csv(pear, os.path.join(out_dir, 'corr_pearson.csv'))
    IO.write_csv(spear, os.path.join(out_dir, 'corr_spearman.csv'))
    IO.write_csv(spear_pf, os.path.join(out_dir, 'corr_spearman_front.csv'))
    IO.write_tex(spear, os.path.join(out_dir, 'corr_spearman.tex'),
                 caption=f'Spearman metric correlations -- {space} '
                         f'({"tabular" if A.is_tabular(space) else "surrogate (attenuated acc.)"}).',
                 label=f'tab:corr-{space}')
    _heatmap(spear, os.path.join(out_dir, 'corr_spearman.png'), f'{space} Spearman')
    _heatmap(pear, os.path.join(out_dir, 'corr_pearson_log.png'), f'{space} Pearson (log-resource)')

    return {'spearman': spear, 'tabular': A.is_tabular(space), 'metrics': cols}


# ─────────────────────────── (c) quantile thresholds ─────────────────────────

def analyze_c(space, df, manifest, out_dir):
    """Per-metric quantile-threshold table (value at each percentile in QG) plus
    CDF plots with tau markers -- turns 'top X%' into concrete metric values."""
    metrics = IO.metric_cols(manifest)
    rows = []
    for m in metrics:
        x = df[m].to_numpy(dtype=float)
        xv = x[np.isfinite(x)]
        n = xv.size
        # tau is the threshold leaving p% feasible in the metric's own
        # direction, so `threshold` is comparable across min/max metrics;
        # raw quantiles stay available in marginals.csv.
        for p in QG:
            rows.append({'metric': m, 'direction': A.metric_direction(m),
                         'percentile': p, 'threshold': A.feasibility_tau(xv, p, m),
                         'n_behind': int(round(n * p / 100.0)), 'n_total': n})
    tab = pd.DataFrame(rows)
    IO.write_csv(tab, os.path.join(out_dir, 'quantile_thresholds.csv'))
    piv = tab.pivot(index='metric', columns='percentile', values='threshold')
    IO.write_tex(piv, os.path.join(out_dir, 'quantile_thresholds.tex'),
                 caption=f'Quantile thresholds tau (feasibility ratio = percentile) -- {space}.',
                 label=f'tab:quant-{space}')

    ncol = min(4, len(metrics)); nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        xv = np.sort(df[m].to_numpy(dtype=float))
        xv = xv[np.isfinite(xv)]
        cdf = np.arange(1, xv.size + 1) / xv.size
        ax.plot(xv, cdf, color=ST.OKABE_ITO[2])
        for p in TAU:
            ax.axvline(np.percentile(xv, p), color=ST.OKABE_ITO[6], lw=0.6, ls='--')
        if manifest['families'][m] == 'resource' and (xv > 0).all():
            ax.set_xscale('log')
        ax.set_xlabel(m); ax.set_ylabel('CDF')
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(f'{space} CDFs with tau markers')
    ST.apply_row_spacing(fig, nrow)
    ST.savefig(fig, os.path.join(out_dir, 'quantile_cdf.png'))
    return {'quantiles': tab}


# ─────────────────────────── (d) feasibility landscape ───────────────────────

def analyze_d(space, df, manifest, out_dir):
    """Enumerable spaces only: for each resource metric and tau, the fraction of
    a feasible point's 1-mutation neighbours that are also feasible, and the
    fraction of all points within edit-distance 1 of feasibility -- a proxy for
    how repairable/walkable the feasible region is for local search."""
    if not manifest['enumerable']:
        return {'skipped': 'non-enumerable'}
    dcols = IO.decision_cols(df)
    X = df[dcols].to_numpy(dtype=int)
    # enumerable => the cached frame is the full grid, so each decision column's
    # observed values ARE its categorical support (1-mutation neighbourhood),
    # recovered without re-entering EvoXBench.
    cats = [np.unique(X[:, vi]) for vi in range(len(dcols))]
    key_to_row = {tuple(r): i for i, r in enumerate(X)}
    metrics = [m for m in IO.metric_cols(manifest) if manifest['families'][m] == 'resource']

    # precompute neighbour row-index lists once (feasibility-independent)
    neigh = [[] for _ in range(len(X))]
    for i, row in enumerate(X):
        for vi in range(len(dcols)):
            for val in cats[vi]:
                if val == row[vi]:
                    continue
                r = list(row); r[vi] = val
                j = key_to_row.get(tuple(r))
                if j is not None:
                    neigh[i].append(j)
    neigh = [np.asarray(n, dtype=int) for n in neigh]

    rows = []
    for m in metrics:
        v = df[m].to_numpy(dtype=float)
        for p in TAU:
            tau = A.feasibility_tau(v, p, m)
            feas = A.feasible_mask(v, tau, m)
            feas_idx = np.where(feas)[0]
            if feas_idx.size == 0:
                continue
            # mean fraction of feasible neighbours over feasible points
            fr_list = []
            for i in feas_idx:
                nb = neigh[i]
                if nb.size:
                    fr_list.append(feas[nb].mean())
            mean_nf = float(np.mean(fr_list)) if fr_list else np.nan
            # fraction of ALL points within edit-distance 1 of feasibility
            within1 = feas.copy()
            for i in range(len(X)):
                if not within1[i] and neigh[i].size and feas[neigh[i]].any():
                    within1[i] = True
            frac_within1 = float(within1.mean())
            rows.append({'metric': m, 'percentile': p, 'feas_ratio': float(feas.mean()),
                         'mean_neighbour_feasibility': mean_nf,
                         'frac_within_edit1': frac_within1})
    tab = pd.DataFrame(rows)
    IO.write_csv(tab, os.path.join(out_dir, 'neighbour_feasibility.csv'))
    IO.write_tex(tab, os.path.join(out_dir, 'neighbour_feasibility.tex'),
                 caption=f'1-mutation neighbour feasibility vs tau -- {space}.',
                 label=f'tab:neigh-{space}', index=False)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for k, m in enumerate(metrics):
        sub = tab[tab.metric == m].sort_values('feas_ratio')
        ax.plot(sub['feas_ratio'], sub['mean_neighbour_feasibility'],
                marker='o', label=m, color=ST.OKABE_ITO[(k + 1) % len(ST.OKABE_ITO)])
    ax.set_xlabel('feasible ratio (tau)'); ax.set_ylabel('mean neighbour feasibility')
    ax.legend(fontsize=6); ax.set_title(f'{space} repair-cost proxy')
    ST.savefig(fig, os.path.join(out_dir, 'neighbour_feasibility.png'))
    return {'table': tab}


# ─────────────────── shared objective-space normalization ────────────────────

def _normalize_min(M, cols, fams):
    """Log-transform positive resources, then min-max normalize each objective
    to [0,1] using the (unconstrained) column range, flipping maximize-better
    metrics so every column is minimized. Returns (Fn, ref_point)."""
    Fn = M.astype(float).copy()
    for j, m in enumerate(cols):
        if fams[m] == 'resource':
            col = Fn[:, j]
            if np.all(col[np.isfinite(col)] > 0):
                Fn[:, j] = np.log10(col)
    lo = np.nanmin(Fn, axis=0)
    hi = np.nanmax(Fn, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Fn = (Fn - lo) / span
    for j, m in enumerate(cols):
        if A.metric_direction(m) == 'max':
            Fn[:, j] = 1.0 - Fn[:, j]
    ref = np.full(Fn.shape[1], 1.1)
    return Fn, ref


FRONT_CAP = 200_000


def _subsample(n, cap, seed=0):
    if n <= cap:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=cap, replace=False)


def _joint_feasibility(df, c_tuple, p):
    """Feasibility mask for a single constraint or a joint AND-pair, each
    member held at its OWN marginal p%-feasible threshold in that metric's
    direction (mirrors (g)'s hold-each-marginal-constant convention, swept
    across the tau grid instead of fixed at 20%). Returns (feas_mask, taus)
    with one tau per member."""
    feas = np.ones(len(df), dtype=bool)
    taus = []
    for c in c_tuple:
        cv = df[c].to_numpy(dtype=float)
        tau = A.feasibility_tau(cv, p, c)
        feas &= A.feasible_mask(cv, tau, c)
        taus.append(tau)
    return feas, taus


def _constraint_label(c_tuple):
    return '+'.join(c_tuple)


# ─────────────────────────── (e) constrained PF(tau) ─────────────────────────

def analyze_e(space, df, manifest, out_dir, force=False):
    """Constrained Pareto front vs. tau: for each objective pair and candidate
    constraint (single or joint AND-pair), recomputes the front under the
    feasibility mask at each tau and reports HV relative to the unconstrained
    front, IGD (exact spaces only), and the surviving-front fraction."""
    # 'complete' (whole true population, e.g. NB101's DB-exact enumeration) is
    # what matters for exactness here, not 'enumerable' (dense raw-grid, which
    # (d)'s neighbour-feasibility needs specifically); older manifests without
    # 'complete' fall back to 'enumerable', where the two happened to coincide.
    exact = manifest.get('complete', manifest['enumerable'])
    csv_path = os.path.join(out_dir, 'hv_curve.csv')
    if not force and os.path.exists(csv_path):
        tab = pd.read_csv(csv_path)
        if len(tab):
            _plot_e(space, tab, out_dir)
        return {'hv_curve': tab, 'exact': exact, 'cached': True}

    metrics = IO.metric_cols(manifest)
    fams = manifest['families']
    results = []
    ref_dir = os.path.join(out_dir, 'pf_tau')
    for objset in A.objective_sets(metrics):
        Mobj = df[list(objset)].to_numpy(dtype=float)
        Fn, ref = _normalize_min(Mobj, list(objset), fams)
        # unconstrained PF(inf)
        idx0 = _subsample(len(Fn), FRONT_CAP) if not exact else np.arange(len(Fn))
        ff0_local = FR.first_front(Fn[idx0])
        ff0 = idx0[ff0_local]
        keys0 = set(df.index[ff0])
        hv0 = FR.hypervolume(Fn[ff0], ref)
        ref_set = Fn[ff0]
        for c_tuple in A.constraint_candidates(metrics, objset):
            c = _constraint_label(c_tuple)
            for p in TAU:
                feas, taus = _joint_feasibility(df, c_tuple, p)
                fidx = np.where(feas)[0]
                if fidx.size == 0:
                    continue
                sub = _subsample(fidx.size, FRONT_CAP) if not exact else np.arange(fidx.size)
                fsel = fidx[sub]
                ffloc = FR.first_front(Fn[fsel])
                ff = fsel[ffloc]
                hv = FR.hypervolume(Fn[ff], ref)
                igd = FR.igd(Fn[ff], ref_set) if exact else float('nan')
                surv = len(keys0 & set(df.index[ff])) / max(len(keys0), 1)
                results.append({
                    'objset': '+'.join(objset), 'constraint': c, 'n_constraints': len(c_tuple),
                    'percentile': p,
                    'feas_ratio': float(feas.mean()), 'tau': taus[0] if len(taus) == 1 else taus,
                    'hv': hv, 'hv_frac_of_unconstrained': hv / hv0 if hv0 > 0 else np.nan,
                    'survival_frac': surv, 'igd': igd, 'n_front': int(len(ff)),
                    'exact': exact,
                })
                if exact:
                    os.makedirs(os.path.join(ref_dir), exist_ok=True)
                    pd.DataFrame(Fn[ff], columns=list(objset)).to_parquet(
                        os.path.join(ref_dir, f'front_{"+".join(objset)}_{c}_{p}.parquet'))
    tab = pd.DataFrame(results)
    IO.write_csv(tab, os.path.join(out_dir, 'hv_curve.csv'))
    if len(tab):
        IO.write_tex(tab, os.path.join(out_dir, 'hv_curve.tex'),
                     caption=f'Constrained-front HV(tau) and survival -- {space} '
                             f'({"exact" if exact else "approx"}).',
                     label=f'tab:hv-{space}', index=False)
        _plot_e(space, tab, out_dir)
    return {'hv_curve': tab, 'exact': exact}


def _plot_e(space, tab, out_dir):
    primary = tab['objset'].iloc[0]
    sub = tab[tab.objset == primary]
    for metric_y, fname, ylab in [
        ('hv_frac_of_unconstrained', 'hv_curve.png', 'HV(tau) / HV(inf)'),
        ('survival_frac', 'survival.png', 'surviving-front fraction')]:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for k, c in enumerate(sorted(sub['constraint'].unique())):
            s = sub[sub.constraint == c].sort_values('feas_ratio')
            ax.plot(s['feas_ratio'], s[metric_y], marker='o', label=c,
                    color=ST.OKABE_ITO[(k + 1) % len(ST.OKABE_ITO)])
        ax.set_xlabel('feasible ratio (tau)'); ax.set_ylabel(ylab)
        ax.legend(fontsize=6); ax.set_title(f'{space}: {primary}')
        ST.savefig(fig, os.path.join(out_dir, fname))


# ─────────────────────────── (f) region classification ──────────────────────

_REGION_CACHE_CATEGORIES = ('infeasible-dominated', 'feasible-dominated',
                            'infeasible-valuable', 'feasible-nondominated')


def _region_key(objset, c):
    # keyed by BOTH objset and constraint: the same constraint label can
    # recur under multiple objsets, and a key on `c` alone made every write
    # silently overwrite the previous objset's cache/figure for that c.
    return f'{"+".join(objset)}__{c}'


def _region_cache_path(out_dir, objset, c):
    return os.path.join(out_dir, f'_region_cache_{_region_key(objset, c)}.npz')


def analyze_f(space, df, manifest, out_dir, force=False):
    """Classifies every point into feasible/infeasible x dominated/nondominated
    of the constrained front, counts region masses per tau, and (single
    constraints only) distance-to-feasibility of infeasible-valuable points --
    shows how much Pareto-optimal material a constraint discards vs. how much
    infeasible-but-valuable territory remains nearby. Also renders scatters."""
    counts_path = os.path.join(out_dir, 'region_counts.csv')
    exact = manifest.get('complete', manifest['enumerable'])
    # A region-cache npz is required to redraw scatters without recomputing;
    # its absence means this space predates the cache (or force=True runs
    # never write it as a cache to trust), so a full recompute is required
    # rather than silently leaving stale/missing scatter plots.
    has_region_cache = os.path.isdir(out_dir) and any(
        fn.startswith('_region_cache_') and fn.endswith('.npz') for fn in os.listdir(out_dir))
    if not force and os.path.exists(counts_path) and has_region_cache:
        tab = pd.read_csv(counts_path)
        _redraw_region_scatters(space, df, manifest, out_dir)
        return {'regions': tab, 'exact': exact, 'cached': True}

    metrics = IO.metric_cols(manifest)
    fams = manifest['families']
    rows = []
    scatter_done = set()
    for objset in A.objective_sets(metrics):
        Mobj = df[list(objset)].to_numpy(dtype=float)
        Fn, _ = _normalize_min(Mobj, list(objset), fams)
        for c_tuple in A.constraint_candidates(metrics, objset):
            c = _constraint_label(c_tuple)
            for p in TAU:
                feas, taus = _joint_feasibility(df, c_tuple, p)
                fidx = np.where(feas)[0]
                if fidx.size == 0:
                    continue
                sub = _subsample(fidx.size, FRONT_CAP) if not exact else np.arange(fidx.size)
                fsel = fidx[sub]
                cfront_local = FR.first_front(Fn[fsel])
                cfront = fsel[cfront_local]
                cfront_F = Fn[cfront]
                dom = FR.dominated_by_front(Fn, cfront_F)
                on_front = np.zeros(len(df), bool); on_front[cfront] = True
                region = np.empty(len(df), dtype=object)
                region[feas & on_front] = 'feasible-nondominated'
                region[feas & ~on_front] = 'feasible-dominated'
                region[~feas & ~dom] = 'infeasible-valuable'
                region[~feas & dom] = 'infeasible-dominated'
                counts = pd.Series(region).value_counts().to_dict()
                iv = region == 'infeasible-valuable'
                row = {'objset': '+'.join(objset), 'constraint': c, 'n_constraints': len(c_tuple),
                       'percentile': p, 'feas_ratio': float(feas.mean())}
                for r in ST.REGION_COLORS:
                    row[r] = int(counts.get(r, 0))
                # distance-to-feasibility for infeasible-valuable (constraint units) --
                # only well-defined for a single constraint, not a joint AND-pair.
                if len(c_tuple) == 1 and iv.any():
                    cv = df[c_tuple[0]].to_numpy(dtype=float)
                    # signed so violation is always positive, either direction
                    sgn = 1.0 if A.metric_direction(c_tuple[0]) == 'min' else -1.0
                    dist = sgn * (cv[iv] - taus[0])
                    row['iv_dist_mean'] = float(np.mean(dist))
                    row['iv_dist_median'] = float(np.median(dist))
                    for k, o in enumerate(objset):
                        row[f'iv_centroid_{o}'] = float(np.mean(Fn[iv, k]))
                rows.append(row)
                # one representative scatter per (objset, constraint) at median tightness
                if p == 10 and (objset, c) not in scatter_done and len(objset) == 2:
                    key = _region_key(objset, c)
                    _region_scatter(space, Fn, region, objset, c,
                                    os.path.join(out_dir, f'region_{key}.png'))
                    IO.save_categorical(_region_cache_path(out_dir, objset, c), region,
                                        list(_REGION_CACHE_CATEGORIES),
                                        objset=np.asarray(objset, dtype=object))
                    scatter_done.add((objset, c))
    tab = pd.DataFrame(rows)
    IO.write_csv(tab, os.path.join(out_dir, 'region_counts.csv'))
    if len(tab):
        IO.write_tex(tab, os.path.join(out_dir, 'region_counts.tex'),
                     caption=f'Region masses per tau -- {space}.',
                     label=f'tab:region-{space}', index=False)
    return {'regions': tab, 'exact': exact}


def _redraw_region_scatters(space, df, manifest, out_dir):
    """Re-render region_{objset}__{c}.png from the per-point region cache
    written by a prior full run of analyze_f, without recomputing any
    front/dominance.
    """
    fams = manifest['families']
    for fn in os.listdir(out_dir):
        if not (fn.startswith('_region_cache_') and fn.endswith('.npz')):
            continue
        key = fn[len('_region_cache_'):-len('.npz')]
        region, extra = IO.load_categorical(os.path.join(out_dir, fn))
        # caches written against an older, smaller Stage-1 sample can't be
        # indexed against the current frame; skip rather than crash the redraw
        if len(region) != len(df):
            print(f'  (f) skipping stale region cache {fn} '
                  f'({len(region):,} rows vs {len(df):,})')
            continue
        objset = tuple(extra['objset'].tolist())
        prefix = '+'.join(objset) + '__'
        c = key[len(prefix):] if key.startswith(prefix) else key
        Mobj = df[list(objset)].to_numpy(dtype=float)
        Fn, _ = _normalize_min(Mobj, list(objset), fams)
        _region_scatter(space, Fn, region, objset, c, os.path.join(out_dir, f'region_{key}.png'))


def _region_scatter(space, Fn, region, objset, c, path=None, cap=30_000, ax=None, legend=True):
    # Both "nondominated" classes are typically a tiny fraction of the
    # population, so a flat random subsample of everything can miss either
    # entirely by chance (observed on DARTS). Keep every point in both, and
    # spend the remaining display budget on a random sample of the rest.
    feas_nd_idx = np.where(region == 'feasible-nondominated')[0]
    # 'infeasible-nondominated': the non-dominated front among
    # 'infeasible-valuable' points -- i.e. restricted to infeasible points
    # *not already dominated by the feasible front*. 'infeasible-dominated'
    # points are excluded from this comparison entirely: since the feasible
    # front already dominates them, calling one "non-dominated among
    # infeasible points" would be true but useless -- it's still strictly
    # worse than something already feasible. (Restricting the input set here
    # is also lossless: an infeasible-dominated point can never dominate an
    # infeasible-valuable one, by transitivity through whatever feasible-front
    # point dominates it, so this gives the same valuable-side front as
    # computing over all infeasible points and filtering afterward.)
    valuable_idx = np.where(region == 'infeasible-valuable')[0]
    infeas_nd_idx = (valuable_idx[FR.first_front(Fn[valuable_idx])]
                     if valuable_idx.size else np.empty(0, dtype=int))

    priority_idx = np.union1d(feas_nd_idx, infeas_nd_idx).astype(int)
    rest_idx = np.setdiff1d(np.arange(len(Fn)), priority_idx)
    rest_budget = max(cap - len(priority_idx), 0)
    rest_sel = rest_idx[_subsample(len(rest_idx), rest_budget, seed=1)] if rest_budget else np.empty(0, dtype=int)
    idx = np.concatenate([priority_idx, rest_sel]).astype(int)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
    # Background -> foreground: the three unchanged region colours/markers
    # first, then the two nondominated classes -- each its own colour AND
    # marker shape (never just a recolour), so they can't be confused with
    # each other or with the three regular regions, and feasible-nondominated
    # (the more important of the two) is drawn last, on top.
    for r, col in ST.REGION_COLORS.items():
        if r == 'feasible-nondominated':
            continue
        sel = idx[region[idx] == r]
        if sel.size:
            ax.scatter(Fn[sel, 0], Fn[sel, 1], s=3, c=col, label=r,
                       rasterized=True, alpha=0.5)
    if infeas_nd_idx.size:
        ax.scatter(Fn[infeas_nd_idx, 0], Fn[infeas_nd_idx, 1], s=10,
                   c=ST.INFEASIBLE_NONDOMINATED_COLOR, marker=ST.REGION_MARKERS['infeasible-nondominated'],
                   label='infeasible-nondominated', rasterized=True)
    if feas_nd_idx.size:
        sel = idx[region[idx] == 'feasible-nondominated']
        ax.scatter(Fn[sel, 0], Fn[sel, 1], s=10, c=ST.REGION_COLORS['feasible-nondominated'],
                   marker=ST.REGION_MARKERS['feasible-nondominated'],
                   label='feasible-nondominated', rasterized=True)
    # maximize-better objectives are flipped by _normalize_min, so say so
    def _axlab(m):
        return f'{m} (norm.{", flipped" if A.metric_direction(m) == "max" else ""})'
    ax.set_xlabel(_axlab(objset[0])); ax.set_ylabel(_axlab(objset[1]))
    if legend:
        ax.legend(fontsize=5, markerscale=2)
    ax.set_title(f'{space}:\nconstraint {c} @10%')
    if own_fig:
        ST.savefig(fig, path)


# ─────────────────────────── (g) joint feasibility ───────────────────────────

def analyze_g(space, df, manifest, out_dir, spearman=None):
    """For each metric pair, holds both at their own 20th-percentile marginal
    threshold and compares observed joint feasibility to the independence-
    assumed product of marginals. Ratio > 1 = aligned/redundant constraints,
    < 1 = antagonistic; optionally cross-checked against Spearman > 0.9."""
    metrics = [m for m in IO.metric_cols(manifest) if m != 'err']
    hold = 20  # marginal tightness held constant
    rows = []
    for a, b in itertools.combinations(metrics, 2):
        if spearman is not None and a in spearman.index and b in spearman.columns:
            if abs(spearman.loc[a, b]) > 0.9:
                redundant = True
            else:
                redundant = False
        else:
            redundant = None
        va = df[a].to_numpy(dtype=float); vb = df[b].to_numpy(dtype=float)
        ta = A.feasibility_tau(va, hold, a)
        tb = A.feasibility_tau(vb, hold, b)
        fa = A.feasible_mask(va, ta, a); fb = A.feasible_mask(vb, tb, b)
        marg_a = fa.mean(); marg_b = fb.mean()
        joint = (fa & fb).mean()
        product = marg_a * marg_b
        rows.append({'metric_a': a, 'metric_b': b, 'marg_a': float(marg_a),
                     'marg_b': float(marg_b), 'joint_obs': float(joint),
                     'product': float(product),
                     'ratio': float(joint / product) if product > 0 else np.nan,
                     'redundant_spearman_gt_0.9': redundant})
    tab = pd.DataFrame(rows).sort_values('ratio')
    IO.write_csv(tab, os.path.join(out_dir, 'joint_ratio.csv'))
    IO.write_tex(tab, os.path.join(out_dir, 'joint_ratio.tex'),
                 caption=f'Joint feasibility ratio at {hold}% marginals -- {space}. '
                         f'>1 aligned/redundant, <1 antagonistic.',
                 label=f'tab:joint-{space}', index=False)
    # heatmap of ratio
    mm = [m for m in metrics]
    R = pd.DataFrame(np.nan, index=mm, columns=mm)
    for _, r in tab.iterrows():
        R.loc[r.metric_a, r.metric_b] = r.ratio
        R.loc[r.metric_b, r.metric_a] = r.ratio
    fig, ax = plt.subplots(figsize=(0.6 * len(mm) + 2, 0.6 * len(mm) + 1.5))
    im = ax.imshow(np.log2(R.to_numpy().astype(float)), cmap=ST.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(mm))); ax.set_xticklabels(mm, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mm))); ax.set_yticklabels(mm, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label='log2(joint/product)')
    ax.set_title(f'{space} joint-feasibility ratio')
    ST.savefig(fig, os.path.join(out_dir, 'joint_ratio.png'))
    return {'joint': tab}
