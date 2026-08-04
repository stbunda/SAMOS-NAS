"""experiments2/scenario_run/_metrics.py -- feasibility-region metric
helpers used by analyse_scenarios.py: attainable-cloud construction,
true-eval re-evaluation, feasible-front extraction, hypervolume, empirical
attainment surfaces, and paired significance testing.

Self-contained port of the single-constraint subset of
experiments/constraint2/analyse_constraint.py -- no multi-constraint
branching, no coupling to run_constraint's SCENARIOS/THRESHOLDS/OBJ_METRICS.
Re-evaluation conventions (round X to int, benchmark.evaluate(X,
true_eval=True), then benchmark.normalize() only when not
benchmark.normalized_objectives) match callbacks.py / constrained_problem.py
exactly, same as the original.
"""

import itertools
import math
import random

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import wilcoxon


# ─── re-evaluation helpers ─────────────────────────────────────────────────

# MoSegNAS (S6/S7) adds simulated measurement noise through Python's stdlib
# random -- +/-2% on latency, +/-5% on energy, plus RankNet dropout on err --
# so repeated true-evals of one architecture differ. Re-seeding before every
# batch makes an analysis run reproducible; every other benchmark here is
# already deterministic and is unaffected. Values still depend on how rows are
# batched, so two analyses over DIFFERENT run sets can differ slightly on
# S6/S7; re-running the same command reproduces exactly.
REEVAL_SEED = 0


def _evaluate_true(benchmark, X):
    """true_eval batch with the stdlib RNG pinned (see REEVAL_SEED)."""
    random.seed(REEVAL_SEED)
    return benchmark.evaluate(X, true_eval=True)


def _norm_once(benchmark, F):
    """benchmark.normalize() only when the benchmark does not already return
    normalized objectives -- never stacked twice."""
    if not benchmark.normalized_objectives:
        F = benchmark.normalize(F)
    return np.where(np.isfinite(F), F, np.nan)


def _full_reeval(benchmark, X, cache: dict):
    """True-eval every row of *X* (n, n_var), via *cache* keyed by rounded-int
    row (shared across generations/runs of one instance). Returns (F_full,
    finite_mask), both length len(X)."""
    if len(X) == 0:
        return np.empty((0, benchmark.evaluator.n_objs)), np.array([], dtype=bool)
    X_int = np.round(np.asarray(X, dtype=float)).astype(int)
    keys = [tuple(row) for row in X_int]
    missing = sorted(set(i for i, k in enumerate(keys) if k not in cache))
    if missing:
        F_new = _norm_once(benchmark, _evaluate_true(benchmark, X_int[missing]))
        for i, row in zip(missing, F_new):
            cache[keys[i]] = np.asarray(row, dtype=float)
    F_full = np.array([cache[k] for k in keys], dtype=float)
    finite_mask = np.isfinite(F_full).all(axis=1)
    return F_full, finite_mask


def space_size(benchmark) -> int:
    """Cardinality of the integer search space, from the benchmark's own
    bounds. Python ints -- no overflow for huge spaces."""
    lb = [int(v) for v in benchmark.search_space.lb]
    ub = [int(v) for v in benchmark.search_space.ub]
    return math.prod(u - l + 1 for l, u in zip(lb, ub))


def enumerate_full_space(benchmark):
    """Exhaustive true-eval of the full space (only call when enumerable)."""
    lb = np.asarray(benchmark.search_space.lb, dtype=int)
    ub = np.asarray(benchmark.search_space.ub, dtype=int)
    ranges = [range(int(l), int(u) + 1) for l, u in zip(lb, ub)]
    X_all = np.array(list(itertools.product(*ranges)), dtype=int)
    F_all = _norm_once(benchmark, _evaluate_true(benchmark, X_all))
    finite_mask = np.isfinite(F_all).all(axis=1)
    return X_all[finite_mask], F_all[finite_mask]


def sample_cloud(benchmark, n=10_000, seed=0):
    """Fixed-seed uniform random sample of the space, true-evaled -- an
    APPROXIMATE attainable cloud for non-enumerable instances (plot
    background only; never used as a true front). np.random.seed(seed) then
    per-column randint(lb, ub + 1)."""
    lb = [int(v) for v in benchmark.search_space.lb]
    ub = [int(v) for v in benchmark.search_space.ub]
    np.random.seed(seed)
    X = np.column_stack([np.random.randint(lo, hi + 1, size=n)
                         for lo, hi in zip(lb, ub)]).astype(int)
    F = _norm_once(benchmark, _evaluate_true(benchmark, X))
    finite_mask = np.isfinite(F).all(axis=1)
    return X[finite_mask], F[finite_mask]


def true_feasible_front(F_all, obj_indices, constr_idx, tau):
    """(feasible_fraction, feasible ND front in {obj_indices} space, same
    front with the raw constrained metric appended -- (n, 3), for boundary
    diagnostics). Single constraint: feasible <=> F[:, constr_idx] <= tau."""
    v = (F_all[:, constr_idx] - tau) / tau
    feasible_mask = v <= 0
    frac = float(feasible_mask.mean())
    F_feas_full = F_all[feasible_mask]
    F_feas = F_feas_full[:, obj_indices]
    if len(F_feas) == 0:
        return frac, np.empty((0, len(obj_indices))), np.empty((0, len(obj_indices) + 1))
    nd_idx = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)
    front_c = np.column_stack([F_feas[nd_idx], F_feas_full[nd_idx, constr_idx]])
    return frac, F_feas[nd_idx], front_c


def _hv(F, ref_point):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or len(F) == 0:
        return 0.0
    return float(HV(ref_point=ref_point)(F))


# ─── empirical attainment surfaces (2-D minimization) ──────────────────────

def attainment_surfaces(fronts, ks):
    """k-out-of-n empirical attainment surfaces for 2-D minimization fronts.

    Parameters
    ----------
    fronts : list of (m_i, 2) arrays (one per seed/run).
    ks : iterable of int in [1, n] -- k=1 is the best (attained by >= 1 run),
         k=n the worst (attained by all), k=ceil(n/2) the median surface.

    Returns
    -------
    (xs, {k: ys}) where xs is the sorted union of all fronts' x-coordinates
    and ys[j] is the y-level attained (weakly dominated) by at least k of the
    n fronts at x = xs[j]: the k-th smallest of the per-front minimal y over
    points with x' <= xs[j]. np.inf where fewer than k fronts attain any
    point at that x. Drawn as a staircase with step(..., where='post').
    """
    fronts = [np.asarray(f, dtype=float) for f in fronts]
    assert fronts and all(f.ndim == 2 and f.shape[1] == 2 and len(f) > 0 for f in fronts), \
        'attainment_surfaces needs >= 1 non-empty (m, 2) front'
    n = len(fronts)
    xs = np.unique(np.concatenate([f[:, 0] for f in fronts]))
    Y = np.full((n, xs.size), np.inf)
    for i, f in enumerate(fronts):
        order = np.argsort(f[:, 0], kind='stable')
        fx, fy = f[order, 0], f[order, 1]
        cummin = np.minimum.accumulate(fy)
        idx = np.searchsorted(fx, xs, side='right') - 1
        valid = idx >= 0
        Y[i, valid] = cummin[idx[valid]]
    Ys = np.sort(Y, axis=0)
    return xs, {int(k): Ys[int(k) - 1, :] for k in ks}


def _pad_mean(rows):
    """Elementwise nanmean over ragged lists of floats (right-padded with
    NaN) -- averages per-generation trajectories over seeds whose run lengths
    might differ slightly."""
    if not rows:
        return []
    max_len = max(len(r) for r in rows)
    arr = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    with np.errstate(invalid='ignore'):
        return np.nanmean(arr, axis=0).tolist()


# ─── significance testing ───────────────────────────────────────────────────

def _paired_wilcoxon(a_vals, b_vals):
    """Two-sided Wilcoxon signed-rank on seed-paired samples. An all-zero
    difference vector (byte-identical runs) is a p = 1.0 tie by definition --
    scipy's wilcoxon raises on it instead. zero_method='wilcox' drops the
    remaining zero differences; method='auto' picks the exact null
    distribution when the (nonzero) sample is small and tie-free."""
    d = np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float)
    if np.all(d == 0.0):
        return 0.0, 1.0
    stat, p = wilcoxon(a_vals, b_vals, alternative='two-sided',
                       zero_method='wilcox', method='auto')
    return float(stat), float(p)


def _holm(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the same order as *pvals* (each clipped to [running max, 1])."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted
