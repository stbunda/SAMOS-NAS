"""Aggregate the sample-selection-bias campaign into tables and figures.

Reads {results_root}/{tightness}/{role}/{hw}/{arm}/seed_*.pkl and writes to
--output_dir:

  runs.csv       one row per run: HV at each evaluation budget, evaluations to
                 reach 95% of the cell optimum, whether the run ever found a
                 feasible point, and the surrogate's accuracy on each side of
                 the evaluability gate.
  cells.csv      per (tightness, role, hw) arm means and the paired bias gap.
  budget.csv     bias gap and success rate as a function of the evaluation
                 budget -- the table the headline figure is drawn from.
  bias_gap.png   bias gap vs evaluation budget, per role. THE result.
  surrogate.png  surrogate MAE above the size budget -- the mechanism.
  success.png    share of seeds that ever find a feasible solution.
  hv.png         HV reached, at the unsaturated budget.

Three measurement decisions, each forced by what the data turned out to be:

1. HV IS READ AT SEVERAL BUDGETS, not just at the end. At the campaign's
   1200-evaluation budget 81% of runs sit above 99% of the exact optimum -- a
   1200-evaluation budget enumerates 7.7% of a 15 625-architecture space, so
   both arms simply solve the problem and final HV cannot separate them. The
   discrimination lives early.

2. SUCCESS RATE IS REPORTED SEPARATELY from HV. In the two roles that impose
   a joint (size AND hardware) constraint, some seeds never find a single
   feasible architecture, so their HV is exactly 0 and the outcome is bimodal
   -- ~0.997 or 0, nothing between. A mean over that mixture is not a central
   tendency, and a bias gap built from two such means has an essentially
   random sign. HV statistics are therefore computed over SUCCEEDING runs
   only, with the success rate carried alongside as its own outcome.

3. THE ARM COMPARISON IS PAIRED BY SEED and tested with Wilcoxon signed-rank.
   The surviving HV differences are small relative to seed-to-seed spread, so
   an unpaired eyeball on the means would read noise as effect.
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
from scipy.stats import wilcoxon

import config as CF

# The tie-correct O(n log n) 2-objective sweep from the constraint-analysis
# pipeline. pymoo's default non-dominated sort is O(MN^2) and does not finish
# on a 200k-row surrogate-space reference within minutes; this one is also the
# implementation that was verified against pymoo across 450+ randomized trials
# after the original sweep was found to silently drop tied points. Same
# cross-experiment import pattern scenario_run/derive_tau.py already uses.
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments2', 'constraint_analysis'))
import _fronts as FR

# Evaluation budgets HV is read at. The campaign runs to 1200; the cells
# saturate by roughly 80, so the interesting range is the low end.
BUDGETS = [40, 80, 160, 320, 640, 1200]
TARGET_FRAC = 0.95     # "solved" threshold for evals_to_target

ROLE_COLOR = {'obj': '#0072B2', 'constr': '#D55E00', 'multi': '#009E73'}
ARM_STYLE = {'feasible': '-', 'oracle': '--'}
ARM_MARKER = {'feasible': 'o', 'oracle': '^'}


# ── exact reference optimum ───────────────────────────────────────────────────

PARQUET = os.path.join(_REPO_ROOT, 'results2', 'constraint_analysis', 'full',
                       '{space}', 'samples.parquet')


def reference_matrix(space, cache_path):
    """Reference population in evaluate space, evaluated with
    ``true_eval=True`` and cached as npz.

    ``true_eval``, not the stage-1 values: the callback scores its archive with
    ``true_eval=True``, a different number from the single-run value used
    during search (they differ by up to 0.043 on MobileNetV3). Measuring the
    reference optimum in the other space makes runs score above their own
    ceiling -- seen at hv_frac = 1.02 on NB201 before this was fixed.

    Exact spaces are ENUMERATED, so their optimum is the true one. Sampled
    spaces reuse the WHOLE stage-1 cache (1M architectures for MobileNetV3, the
    same population the constraint-analysis pipeline was built on); their
    optimum is still an UNDER-estimate of the true front, so hv_frac there is
    approximate and can sit slightly above 1 -- but using the full cache rather
    than a subsample makes that gap as small as the cached evidence allows.
    """
    if os.path.exists(cache_path):
        return np.load(cache_path)['F']
    from problem.evoxbench.utils import get_benchmark
    cfg = CF.space(space)
    bench = get_benchmark(*CF.suite_pid(space))
    ss = bench.search_space
    if cfg['exact']:
        lb, ub = np.asarray(ss.lb, int), np.asarray(ss.ub, int)
        X = np.stack(np.meshgrid(*[np.arange(lo, hi + 1) for lo, hi in zip(lb, ub)],
                                 indexing='ij'), axis=-1).reshape(-1, len(lb))
        print(f'  evaluating the full {len(X)}-architecture {space} grid '
              f'(true_eval) ...')
    else:
        df = pd.read_parquet(PARQUET.format(space=space))
        X = df[[c for c in df.columns if c.startswith('x')]].to_numpy(int)
        print(f'  evaluating the full {len(X)}-architecture {space} stage-1 '
              f'cache (true_eval; the front is an approximation) ...')
    # Chunked: a single evaluate() call over 1M architectures runs the
    # benchmark's predictor on one 1M-row batch, which is a needless memory
    # spike. 50k keeps it flat and costs nothing.
    F = np.vstack([bench.evaluate(X[i:i + 50_000], true_eval=True)
                   for i in range(0, len(X), 50_000)])
    if not bench.normalized_objectives:
        F = bench.normalize(F)
    F = np.where(np.isfinite(F), F, 1.0)
    np.savez_compressed(cache_path, F=F)
    return F


def optimal_hv(F_all, spec):
    """(HV of the EXACT constrained Pareto front, size of the feasible set)
    for one cell -- the ceiling any run in that cell could reach."""
    idx, thr, sns = CF.scoring_constraints(spec)
    mask = np.ones(len(F_all), dtype=bool)
    for i, t, s in zip(idx, thr, sns):
        mask &= (s * (F_all[:, i] - t) <= 0)
    if not mask.any():
        return float('nan'), 0
    F = F_all[mask][:, spec['obj_indices']]
    for pos in spec['flip_obj_pos']:
        F[:, pos] = 1.0 - F[:, pos]
    front = F[FR.first_front(F)]
    return float(HV(ref_point=np.asarray(spec['ref_point'], float))(front)), int(mask.sum())


# ── per-run extraction ────────────────────────────────────────────────────────

def _hv_at(hv, n_eval, budget):
    """HV of the archive after at most ``budget`` real evaluations. The series
    is a step function of the cumulative evaluation count, which the gated DOE
    can advance by more than pop_size in one generation, so this indexes on
    n_evaluated rather than on the generation number."""
    ok = n_eval <= budget
    return float(hv[ok][-1]) if ok.any() else 0.0


def load_runs(results_root, hv_max_of):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_root, '*', '*', '*', '*',
                                              'seed_*.pkl'))):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        m = data['meta']
        hv = np.array([i['hv'] for i in data['indicators']], dtype=float)
        n_eval = np.array(data['n_evaluated'], dtype=float)
        hv_max, n_feas = hv_max_of(m['tightness'], m['role'], m['hw'])

        row = {k: m[k] for k in ('tightness', 'role', 'hw', 'arm', 'seed',
                                 'size_feasible_fraction', 'n_gen',
                                 'n_hf_evaluated', 'n_hf_evaluable',
                                 'n_archive', 'n_rejected')}
        row['obj_metrics'] = '+'.join(m['obj_metrics'])
        row['hv_max'] = hv_max
        row['n_feasible_arch'] = n_feas
        row['evaluable_rate'] = m['n_hf_evaluable'] / max(1, m['n_hf_evaluated'])
        for b in BUDGETS:
            row[f'hv_frac_{b}'] = _hv_at(hv, n_eval, b) / hv_max
        row['hv_frac'] = row[f'hv_frac_{BUDGETS[-1]}']
        row['success'] = bool(hv[-1] > 0) if len(hv) else False
        frac = hv / hv_max
        hit = np.flatnonzero(frac >= TARGET_FRAC)
        row['evals_to_target'] = float(n_eval[hit[0]]) if len(hit) else np.nan

        # Last generation carrying surrogate diagnostics (the DOE generation
        # has no fitted surrogate yet).
        probe = next((p for p in reversed(data.get('probe', []))
                      if 'mae_gated_o0' in p), {})
        row.update({k: v for k, v in probe.items() if k.startswith(('mae_', 'rho_'))})
        rows.append(row)
    if not rows:
        raise SystemExit(f'No pkls found under {results_root}')
    return pd.DataFrame(rows)


# ── aggregation ───────────────────────────────────────────────────────────────

def _paired_gap(df, col):
    """(gap, p) for one group: mean relative shortfall of the feasible arm
    against the oracle, paired by seed, with a Wilcoxon signed-rank p-value.
    Pairs where both arms are 0 carry no information and are dropped."""
    piv = df.pivot_table(index=['tightness', 'role', 'hw', 'seed'],
                         columns='arm', values=col)
    if 'feasible' not in piv or 'oracle' not in piv:
        return np.nan, np.nan, 0
    piv = piv.dropna()
    piv = piv[(piv['feasible'] > 0) | (piv['oracle'] > 0)]
    if len(piv) < 3:
        return np.nan, np.nan, len(piv)
    gap = float(1.0 - piv['feasible'].mean() / piv['oracle'].mean())
    d = piv['oracle'] - piv['feasible']
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return gap, p, len(piv)


def summarise_cells(df):
    """Per (tightness, role, hw): arm means and the paired bias gap. HV means
    are over SUCCEEDING runs only (see the module docstring); the success rate
    is reported next to them as its own outcome."""
    keys = ['tightness', 'role', 'hw']
    ok = df[df.success]
    hv_cols = [f'hv_frac_{b}' for b in BUDGETS]
    agg = {c: 'mean' for c in hv_cols}
    agg.update({'evals_to_target': 'median', 'evaluable_rate': 'mean',
                'mae_gated_o0': 'mean', 'mae_evaluable_o0': 'mean',
                'rho_gated_o0': 'mean'})
    agg = {k: v for k, v in agg.items() if k in df.columns}

    wide = ok.groupby(keys + ['arm']).agg(agg).unstack('arm')
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    succ = df.groupby(keys + ['arm'])['success'].mean().unstack('arm')
    succ.columns = [f'success_{a}' for a in succ.columns]

    gaps = []
    for key, grp in ok.groupby(keys):
        row = dict(zip(keys, key))
        for b in BUDGETS:
            g, p, n = _paired_gap(grp, f'hv_frac_{b}')
            row[f'bias_gap_{b}'] = g
            row[f'p_{b}'] = p
        row['n_pairs'] = n
        gaps.append(row)
    gaps = pd.DataFrame(gaps).set_index(keys)

    meta = df.groupby(keys)[['size_feasible_fraction', 'hv_max',
                             'n_feasible_arch']].first()
    out = meta.join(wide).join(succ).join(gaps).reset_index()
    out['bias_gap'] = out[f'bias_gap_{BUDGETS[-1]}']
    return out


def summarise_budget(df):
    """Bias gap and success rate per (role, tightness, budget) -- pooled over
    hardware metrics, which is the level the headline figure reads at."""
    ok = df[df.success]
    rows = []
    for (role, tight), grp in df.groupby(['role', 'tightness']):
        grp_ok = ok[(ok.role == role) & (ok.tightness == tight)]
        succ = grp.groupby('arm')['success'].mean()
        for b in BUDGETS:
            gap, p, n = _paired_gap(grp_ok, f'hv_frac_{b}')
            rows.append(dict(
                role=role, tightness=tight,
                size_feasible_fraction=grp['size_feasible_fraction'].iloc[0],
                budget=b, bias_gap=gap, p=p, n_pairs=n,
                hv_frac_feasible=grp_ok[grp_ok.arm == 'feasible'][f'hv_frac_{b}'].mean(),
                hv_frac_oracle=grp_ok[grp_ok.arm == 'oracle'][f'hv_frac_{b}'].mean(),
                success_feasible=succ.get('feasible', np.nan),
                success_oracle=succ.get('oracle', np.nan)))
    return pd.DataFrame(rows)


# ── figures ───────────────────────────────────────────────────────────────────

def _save(fig, path, bottom=0.0):
    fig.tight_layout(rect=(0, bottom, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  wrote {path}')


def _tight_panels(space, ylabel, title, width=4.0, present=None):
    # Only panels for tightness levels actually present: a partial campaign
    # would otherwise leave empty log-scaled axes, which matplotlib refuses to
    # lay out ("Data has no positive values").
    tights = [t for t in CF.tightness(space)
              if present is None or t in set(present)]
    fig, axes = plt.subplots(1, len(tights), figsize=(width * len(tights), 3.6),
                             squeeze=False, sharey=True)
    axes = axes.ravel()
    for ax, t in zip(axes, tights):
        ax.set_title(f'{t}: '
                     f'{CF.tightness(space)[t]["feasible_fraction"]:.1%} evaluable',
                     fontsize=10)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13)
    return fig, axes, tights


def plot_bias_gap(space, budget, path):
    """The headline: the bias gap as a function of the evaluation budget, one
    panel per size budget, one line per role."""
    fig, axes, tights = _tight_panels(
        space, r'bias gap  $1 - HV_{feasible}/HV_{oracle}$',
        f'{space}: cost of training the surrogate on the feasible population only',
        present=set(budget['tightness']))
    for ax, t in zip(axes, tights):
        for role in CF.ROLES:
            sub = budget[(budget.tightness == t) & (budget.role == role)
                         ].sort_values('budget')
            if not len(sub):
                continue
            ax.plot(sub['budget'], sub['bias_gap'], marker='o', ms=5,
                    color=ROLE_COLOR[role], label=f'#Params as {role}')
            sig = sub[sub.p < 0.05]
            ax.plot(sig['budget'], sig['bias_gap'], ls='', marker='o', ms=10,
                    mfc='none', mec=ROLE_COLOR[role], mew=1.6)
        ax.axhline(0.0, color='0.5', lw=0.8)
        ax.set_xscale('log')
        # Explicit limits, not autoscale: with too few seed-pairs to test, every
        # gap is NaN and a log axis with no positive data cannot be laid out.
        ax.set_xlim(BUDGETS[0] * 0.8, BUDGETS[-1] * 1.25)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels(BUDGETS, fontsize=8)
        ax.set_xlabel('evaluation budget')
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], ls='', marker='o', ms=9, mfc='none',
                              mec='0.3', label='p < 0.05 (Wilcoxon, paired)'))
    labels.append('p < 0.05 (Wilcoxon, paired)')
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    _save(fig, path, bottom=0.10)


def plot_surrogate(space, cells, path):
    """The mechanism: surrogate error above the size budget, the region the
    realistic arm never observes."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for role in CF.ROLES:
        sub = cells[cells.role == role].groupby('size_feasible_fraction')[
            ['mae_gated_o0_feasible', 'mae_gated_o0_oracle']].mean().reset_index()
        ax.plot(sub['size_feasible_fraction'], sub['mae_gated_o0_feasible'],
                marker=ARM_MARKER['feasible'], ls=ARM_STYLE['feasible'],
                color=ROLE_COLOR[role], label=f'{role} / feasible')
        ax.plot(sub['size_feasible_fraction'], sub['mae_gated_o0_oracle'],
                marker=ARM_MARKER['oracle'], ls=ARM_STYLE['oracle'],
                color=ROLE_COLOR[role], label=f'{role} / oracle', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('#Params budget (share of space evaluable)')
    ax.set_ylabel('surrogate MAE on the error objective, ABOVE the budget')
    ax.set_title(f'{space}: surrogate accuracy in the unobserved region', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_success(space, cells, path):
    """The outcome that actually separates the arms in the joint-constraint
    roles: whether a seed finds any feasible architecture at all."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for role in CF.ROLES:
        sub = cells[cells.role == role].groupby('size_feasible_fraction')[
            ['success_feasible', 'success_oracle']].mean().reset_index()
        ax.plot(sub['size_feasible_fraction'], sub['success_feasible'],
                marker=ARM_MARKER['feasible'], ls=ARM_STYLE['feasible'],
                color=ROLE_COLOR[role], label=f'{role} / feasible')
        ax.plot(sub['size_feasible_fraction'], sub['success_oracle'],
                marker=ARM_MARKER['oracle'], ls=ARM_STYLE['oracle'],
                color=ROLE_COLOR[role], label=f'{role} / oracle', alpha=0.7)
    ax.set_xscale('log')
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('#Params budget (share of space evaluable)')
    ax.set_ylabel('share of seeds finding any feasible solution')
    ax.set_title(f'{space}: success rate over 20 seeds', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_hv(space, cells, path, budget=80):
    """HV reached at an UNSATURATED budget, per hardware metric."""
    col = f'hv_frac_{budget}'
    hws = [h for h in CF.hw(space) if h in set(cells['hw'])]
    fig, axes = plt.subplots(1, len(hws), figsize=(4.0 * len(hws), 3.6),
                             squeeze=False, sharey=True)
    axes = axes.ravel()
    for ax, hw in zip(axes, hws):
        ax.set_title(CF.hw(space)[hw]['metric'], fontsize=10)
        ax.set_xlabel('#Params budget (share evaluable)')
        ax.set_xscale('log')
        for role in CF.ROLES:
            sub = cells[(cells.hw == hw) & (cells.role == role)].sort_values(
                'size_feasible_fraction')
            for arm in CF.ARMS:
                c = f'{col}_{arm}'
                if not len(sub) or c not in sub:
                    continue
                ax.plot(sub['size_feasible_fraction'], sub[c],
                        marker=ARM_MARKER[arm], ms=5, color=ROLE_COLOR[role],
                        ls=ARM_STYLE[arm], label=f'{role} / {arm}')
    axes[0].set_ylabel(f'HV / exact optimum, at {budget} evaluations')
    fig.suptitle(f'{space}: hypervolume reached after {budget} evaluations '
                 f'(succeeding seeds only)', fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   fontsize=8, bbox_to_anchor=(0.5, 0.0))
    _save(fig, path, bottom=0.10)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--space', default=CF.DEFAULT_SPACE, choices=list(CF.SPACES))
    ap.add_argument('--results_root', default=None,
                    help="Default: the space's own root (config.results_root).")
    ap.add_argument('--output_dir', default=None,
                    help='Default: {results_root}/analysis')
    args = ap.parse_args()
    space = args.space
    results_root = args.results_root or CF.results_root(space)
    out = args.output_dir or os.path.join(results_root, 'analysis')
    os.makedirs(out, exist_ok=True)

    F_all = reference_matrix(space, os.path.join(out, f'{space}_reference.npz'))
    opt_cache = {}

    def hv_max_of(tightness, role, hw):
        key = (tightness, role, hw)
        if key not in opt_cache:
            opt_cache[key] = optimal_hv(F_all, CF.cell(*key, space))
        return opt_cache[key]

    runs = load_runs(results_root, hv_max_of)
    cells = summarise_cells(runs)
    budget = summarise_budget(runs)
    runs.to_csv(os.path.join(out, 'runs.csv'), index=False)
    cells.to_csv(os.path.join(out, 'cells.csv'), index=False)
    budget.to_csv(os.path.join(out, 'budget.csv'), index=False)
    print(f'  wrote runs.csv ({len(runs)}), cells.csv ({len(cells)}), '
          f'budget.csv ({len(budget)}) to {out}')

    plot_bias_gap(space, budget, os.path.join(out, 'bias_gap.png'))
    plot_surrogate(space, cells, os.path.join(out, 'surrogate.png'))
    plot_success(space, cells, os.path.join(out, 'success.png'))
    plot_hv(space, cells, os.path.join(out, 'hv.png'))

    print('\n=== surrogate error above the size budget (mean over roles/hw) ===')
    surr = cells.groupby('tightness')[['mae_gated_o0_feasible', 'mae_gated_o0_oracle',
                                       'mae_evaluable_o0_feasible']].mean()
    surr['ratio_feasible_over_oracle'] = (surr['mae_gated_o0_feasible']
                                          / surr['mae_gated_o0_oracle'])
    print(surr.round(4).to_string())

    print('\n=== success rate (any feasible solution found, 20 seeds) ===')
    print(cells.groupby(['role', 'tightness'])[
        ['success_feasible', 'success_oracle']].mean().unstack('tightness').round(3).to_string())

    print(f'\n=== paired bias gap by budget (pooled over hw); * = p < 0.05 ===')
    piv = budget.pivot_table(index=['role', 'tightness'], columns='budget',
                             values='bias_gap')
    pv = budget.pivot_table(index=['role', 'tightness'], columns='budget', values='p')
    show = piv.round(4).astype(str) + np.where(pv < 0.05, '*', '')
    print(show.to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
