"""Aggregate the sample-selection-bias campaign into tables and figures.

Reads {results_root}/{tightness}/{role}/{hw}/{arm}/seed_*.pkl and writes, to
--output_dir:

  runs.csv        one row per run: final feasible HV, HV as a fraction of the
                  cell's EXACT attainable optimum, evaluable rate, and the
                  surrogate's accuracy on each side of the evaluability gate.
  cells.csv       per (tightness, role, hw) means plus the paired bias gap
                  1 - HV_feasible / HV_oracle -- the headline number.
  hv.png          HV vs size budget, per hardware metric, arm as linestyle.
  surrogate.png   surrogate MAE on the GATED side (the region the realistic
                  arm never observes) -- the mechanism behind hv.png.
  bias_gap.png    the bias gap alone, by role, averaged over hardware metrics.

HV is normalised by the exact constrained front, recomputed here from the
NB201 enumeration: the three roles search three different objective spaces, so
raw HV is not comparable across them and only the fraction-of-optimum (and the
arm-to-arm gap, which is scale-free) is.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import config as CF

ROLE_COLOR = {'obj': '#0072B2', 'constr': '#D55E00', 'multi': '#009E73'}
ARM_STYLE = {'feasible': '-', 'oracle': '--'}
ARM_MARKER = {'feasible': 'o', 'oracle': '^'}   # linestyle alone is unreadable
                                                # in a small shared legend


def true_eval_matrix(cache_path):
    """The whole NB201 space (all 15 625 architectures) evaluated with
    ``true_eval=True``, cached as npz.

    true_eval, not the stage-1 sample: the callback scores its archive with
    ``true_eval=True`` (accuracy averaged over the benchmark's training runs),
    which is a different number from the single-run value used during search.
    Measuring the reference optimum in the other space makes runs score above
    their own ceiling -- observed at hv_frac = 1.02 before this was fixed.
    """
    if os.path.exists(cache_path):
        return np.load(cache_path)['F']
    from problem.evoxbench.utils import get_benchmark
    bench = get_benchmark(CF.SUITE, CF.PID)
    ss = bench.search_space
    lb, ub = np.asarray(ss.lb, int), np.asarray(ss.ub, int)
    X = np.stack(np.meshgrid(*[np.arange(lo, hi + 1) for lo, hi in zip(lb, ub)],
                             indexing='ij'), axis=-1).reshape(-1, len(lb))
    print(f'  evaluating the full {len(X)}-architecture NB201 grid (true_eval) ...')
    F = bench.evaluate(X, true_eval=True)
    if not bench.normalized_objectives:
        F = bench.normalize(F)
    F = np.where(np.isfinite(F), F, 1.0)
    np.savez_compressed(cache_path, F=F)
    return F


def optimal_hv(F_all, spec):
    """HV of the EXACT constrained Pareto front for one cell, over the full
    NB201 enumeration -- the ceiling any run in that cell could reach."""
    idx, thr, sns = CF.scoring_constraints(spec)
    mask = np.ones(len(F_all), dtype=bool)
    for i, t, s in zip(idx, thr, sns):
        mask &= (s * (F_all[:, i] - t) <= 0)
    if not mask.any():
        return float('nan'), 0
    F = F_all[mask][:, spec['obj_indices']]
    for pos in spec['flip_obj_pos']:
        F[:, pos] = 1.0 - F[:, pos]
    front = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    return float(HV(ref_point=np.asarray(spec['ref_point'], float))(front)), int(mask.sum())


def load_runs(results_root):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_root, '*', '*', '*', '*',
                                              'seed_*.pkl'))):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        m = data['meta']
        # Last generation that actually carries surrogate diagnostics (the DOE
        # generation has no fitted surrogate yet).
        probe = next((p for p in reversed(data.get('probe', []))
                      if 'mae_gated_o0' in p), {})
        row = {k: m[k] for k in ('tightness', 'role', 'hw', 'arm', 'seed',
                                 'size_feasible_fraction', 'n_gen',
                                 'n_hf_evaluated', 'n_hf_evaluable',
                                 'n_archive', 'n_rejected')}
        row['obj_metrics'] = '+'.join(m['obj_metrics'])
        row['constr_metrics'] = '+'.join(m['constr_metrics'])
        row['hv'] = data['indicators'][-1]['hv'] if data['indicators'] else np.nan
        row['evaluable_rate'] = m['n_hf_evaluable'] / max(1, m['n_hf_evaluated'])
        row.update({k: v for k, v in probe.items() if k.startswith(('mae_', 'rho_'))})
        rows.append(row)
    if not rows:
        raise SystemExit(f'No pkls found under {results_root}')
    return pd.DataFrame(rows)


def add_optimum(df, cache_path):
    """hv_max / hv_frac per row, from the exact enumeration."""
    F_all = true_eval_matrix(cache_path)
    cache = {}
    hv_max, n_feas = [], []
    for _, r in df.iterrows():
        key = (r['tightness'], r['role'], r['hw'])
        if key not in cache:
            cache[key] = optimal_hv(F_all, CF.cell(*key))
        hv_max.append(cache[key][0])
        n_feas.append(cache[key][1])
    df['hv_max'] = hv_max
    df['n_feasible_arch'] = n_feas
    df['hv_frac'] = df['hv'] / df['hv_max']
    return df


def summarise(df):
    """Per (tightness, role, hw): arm means and the paired bias gap."""
    keys = ['tightness', 'role', 'hw']
    agg = {'hv': 'mean', 'hv_frac': 'mean', 'evaluable_rate': 'mean',
           'mae_gated_o0': 'mean', 'mae_evaluable_o0': 'mean',
           'rho_gated_o0': 'mean', 'n_archive': 'mean'}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    wide = df.groupby(keys + ['arm']).agg(agg).unstack('arm')
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    # The headline: how much of the oracle's hypervolume the realistic,
    # feasible-only surrogate gives up. Scale-free, so it IS comparable across
    # the three roles even though their raw HVs are not.
    if 'hv_oracle' in wide.columns:
        wide['bias_gap'] = 1.0 - wide['hv_feasible'] / wide['hv_oracle']
    meta = df.groupby(keys)[['size_feasible_fraction', 'hv_max',
                             'n_feasible_arch']].first()
    return meta.join(wide).reset_index()


def _panels(hws, ylabel, title):
    fig, axes = plt.subplots(1, len(hws), figsize=(4.0 * len(hws), 3.6),
                             squeeze=False, sharex=True)
    axes = axes.ravel()
    for ax, hw in zip(axes, hws):
        ax.set_title(CF.HW[hw]['metric'], fontsize=10)
        ax.set_xlabel('#Params budget (share of space evaluable)')
        ax.set_xscale('log')
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13)
    return fig, axes


def _save(fig, path, bottom=0.0):
    fig.tight_layout(rect=(0, bottom, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  wrote {path}')


def plot_by_arm(cells, column, ylabel, title, path):
    """One panel per hardware metric; colour = role, linestyle = arm."""
    hws = [h for h in CF.HW if h in set(cells['hw'])]
    fig, axes = _panels(hws, ylabel, title)
    for ax, hw in zip(axes, hws):
        for role in CF.ROLES:
            sub = cells[(cells.hw == hw) & (cells.role == role)].sort_values(
                'size_feasible_fraction')
            for arm in CF.ARMS:
                col = f'{column}_{arm}'
                if not len(sub) or col not in sub:
                    continue
                ax.plot(sub['size_feasible_fraction'], sub[col],
                        marker=ARM_MARKER[arm], ms=5, color=ROLE_COLOR[role],
                        ls=ARM_STYLE[arm], label=f'{role} / {arm}')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   fontsize=8, bbox_to_anchor=(0.5, 0.0))
    _save(fig, path, bottom=0.10)


def plot_bias_gap(cells, path):
    """The bias gap by role, averaged over hardware metrics -- the ablation."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for role in CF.ROLES:
        sub = cells[cells.role == role].groupby('size_feasible_fraction')[
            'bias_gap'].agg(['mean', 'std']).reset_index()
        if not len(sub):
            continue
        ax.errorbar(sub['size_feasible_fraction'], sub['mean'], yerr=sub['std'],
                    marker='o', capsize=3, color=ROLE_COLOR[role],
                    label=f'#Params as {role}')
    ax.axhline(0.0, color='0.5', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('#Params budget (share of space evaluable)')
    ax.set_ylabel(r'bias gap  $1 - HV_{feasible} / HV_{oracle}$')
    ax.set_title('Cost of training the surrogate on the feasible population only',
                 fontsize=11)
    ax.legend(fontsize=9)
    _save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results_root',
                    default=os.path.join('results2', 'sample_selection_bias'))
    ap.add_argument('--output_dir', default=None,
                    help='Default: {results_root}/analysis')
    args = ap.parse_args()
    out = args.output_dir or os.path.join(args.results_root, 'analysis')
    os.makedirs(out, exist_ok=True)

    runs = add_optimum(load_runs(args.results_root),
                       os.path.join(out, 'nb201_true_eval.npz'))
    cells = summarise(runs)
    runs.to_csv(os.path.join(out, 'runs.csv'), index=False)
    cells.to_csv(os.path.join(out, 'cells.csv'), index=False)
    print(f'  wrote {out}/runs.csv ({len(runs)} runs), cells.csv ({len(cells)} cells)')

    plot_by_arm(cells, 'hv_frac', 'HV / exact optimum',
                'Feasible hypervolume reached, as a share of the attainable optimum',
                os.path.join(out, 'hv.png'))
    plot_by_arm(cells, 'mae_gated_o0', 'surrogate MAE (error objective)',
                'Surrogate accuracy ABOVE the size budget -- the unobserved region',
                os.path.join(out, 'surrogate.png'))
    if 'bias_gap' in cells:
        plot_bias_gap(cells, os.path.join(out, 'bias_gap.png'))

    show = ['tightness', 'role', 'hw', 'hv_frac_feasible', 'hv_frac_oracle',
            'bias_gap', 'mae_gated_o0_feasible', 'mae_gated_o0_oracle']
    show = [c for c in show if c in cells.columns]
    print()
    print(cells[show].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
