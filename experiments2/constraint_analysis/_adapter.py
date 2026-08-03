"""EvoXBench access layer for the constraint-analysis pipeline.

Every EvoXBench call lives here so an upstream API change touches one file.
Everything downstream reads cached Parquet and never re-enters this module.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from problem.evoxbench.utils import EVOX_DB_PATH, EVOX_DATA_PATH, get_benchmark

# ── the richest PID per search space (the one exposing the most metrics) ──────
# Stage 1 samples one Parquet per space through this PID so every available
# metric column is present; the objective *subsets* used by narrower PIDs are a
# projection of these columns.
RICHEST_PID = {
    'NB101':       ('c10mop', 2),
    'NATS':        ('c10mop', 4),
    'NB201':       ('c10mop', 7),
    'DARTS':       ('c10mop', 9),
    'ResNet-50D':  ('in1kmop', 3),
    'Transformer': ('in1kmop', 6),
    'MobileNetV3': ('in1kmop', 9),
    'MoSegNAS':    ('citysegmop', 15),
}

# Exhaustively enumerate when the naive product of per-variable cardinalities is
# below this; otherwise draw random samples.
ENUM_CAP = 200_000

# Fixed grids shared across analyzers.
QUANTILE_GRID = (1, 5, 10, 20, 50, 90)      # percentiles for (a) and (c)
TAU_RATIOS = (50, 20, 10, 5, 1)             # feasible-fraction % -> tau = p-th pctile

_ERROR_TOKENS = ('err', 'error', 'acc', 'accuracy', 'miou', 'iou')

# Tabular (real-lookup accuracy) vs surrogate (predicted accuracy). Governs the
# (b) attenuation annotation and whether (e) fronts are exact reference sets.
TABULAR = {'NB101', 'NATS', 'NB201'}

# Metrics where larger is better. Arithmetic intensity is OPs/Byte: high means
# compute-bound (the accelerator's MACs stay busy), so a real budget on it is a
# floor, not a ceiling. EvoXBench stores the raw value under an all-minimized
# objective convention, so the direction has to be reimposed here.
# Substring match (device prefixes like `eyeriss_`), plus bare aliases matched
# whole so a short token can never hit an unrelated metric name.
MAXIMIZE_SUBSTRINGS = ('arithmetic_intensity', 'arith_intensity')
MAXIMIZE_EXACT = ('ai',)


def is_tabular(space: str) -> bool:
    return space in TABULAR


def metric_direction(name: str) -> str:
    low = name.lower()
    if low in MAXIMIZE_EXACT or any(t in low for t in MAXIMIZE_SUBSTRINGS):
        return 'max'
    return 'min'


def feasibility_tau(values, p: float, name: str) -> float:
    """tau leaving p% of `values` feasible, in the metric's own direction:
    the p-th percentile for minimize, the (100-p)-th for maximize."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    q = p if metric_direction(name) == 'min' else 100.0 - p
    return float(np.percentile(v, q))


def feasible_mask(values, tau: float, name: str):
    """`metric <= tau` for minimize metrics, `metric >= tau` for maximize."""
    v = np.asarray(values, dtype=float)
    return v <= tau if metric_direction(name) == 'min' else v >= tau


MAX_OBJSET_OTHERS = 3  # cap objective sets at err + up to this many other metrics


def objective_sets(metrics) -> list[tuple]:
    """Every combination of err + 1..MAX_OBJSET_OTHERS other metrics (2-4 total
    objectives) -- exhaustive over every objective set a paper might plausibly
    use, bounded so richer spaces (NB201 has 8 metrics, MoSegNAS 7) don't blow
    up into a many-objective combinatorial explosion (an uncapped power-set
    would be 127 objsets for NB201 alone). Metrics with only 3 total (most
    surrogate spaces: err+params+flops) are unaffected -- every combination of
    up to 3 others among 2 is already exhaustive."""
    metrics = list(metrics)
    if 'err' not in metrics:
        return []
    others = [m for m in metrics if m != 'err']
    sets: list[tuple] = []
    for k in range(1, min(MAX_OBJSET_OTHERS, len(others)) + 1):
        for combo in itertools.combinations(others, k):
            sets.append(('err',) + combo)
    return sets


def constraint_candidates(metrics, objset, max_k: int = 2) -> list[tuple]:
    """Single- and paired-constraint candidates outside the objective set (and
    never err): 1-tuples are single constraints, 2-tuples are a joint
    (AND-feasibility) pair -- the 'do two constraints together carve out a
    different part of the front than either alone' question from the plan's
    (g) design, folded into (e)/(f) rather than left only as a marginal
    joint-ratio screen."""
    cands = [m for m in metrics if m not in objset and m != 'err']
    out: list[tuple] = []
    for k in range(1, max_k + 1):
        out.extend(itertools.combinations(cands, k))
    return out


def metric_family(name: str) -> str:
    """'error' (bounded, linear axis) vs 'resource' (positive, log axis)."""
    low = name.lower()
    if any(tok in low for tok in _ERROR_TOKENS):
        return 'error'
    return 'resource'


def get_space(space: str):
    """Return the EvoXBench benchmark handle for a search-space abbreviation."""
    suite, pid = RICHEST_PID[space]
    return get_benchmark(suite, pid)


def metric_names(bench) -> list[str]:
    """Ordered metric-column names, e.g. ['err', 'params', 'flops', ...]."""
    return list(bench.evaluator.objs.split('&'))


def cardinality(bench) -> int:
    """Naive product of per-variable category counts (python int, no overflow)."""
    cats = getattr(bench.search_space, 'categories', None)
    if not cats:
        ss = bench.search_space
        lb = np.asarray(ss.lb).astype(int)
        ub = np.asarray(ss.ub).astype(int)
        card = 1
        for lo, hi in zip(lb, ub):
            card *= int(hi - lo + 1)
        return card
    card = 1
    for c in cats:
        card *= len(c)
    return card


def is_enumerable(bench) -> bool:
    return cardinality(bench) <= ENUM_CAP


def _arch_keys(X: np.ndarray) -> np.ndarray:
    """Deterministic per-row architecture key = the integer genotype joined by '-'."""
    return np.array(['-'.join(map(str, row)) for row in X], dtype=object)


def evaluate(bench, X: np.ndarray) -> np.ndarray:
    """Evaluate integer genotypes -> raw metric matrix (no normalize, no guard).

    Returns exactly what the benchmark reports (already-normalized for the
    normalized benchmarks, raw units otherwise); non-finite rows are left as-is
    for the caller to drop and count.
    """
    X_int = np.round(np.asarray(X)).astype(int)
    return np.asarray(bench.evaluate(X_int, true_eval=False), dtype=float)


def enumerate_all(bench) -> np.ndarray:
    """Cartesian product of all category values (enumerable spaces only)."""
    cats = bench.search_space.categories
    grids = np.meshgrid(*[np.asarray(c, dtype=int) for c in cats], indexing='ij')
    return np.stack([g.ravel() for g in grids], axis=1)


def _pad_nb101_phenotype(matrix, ops, n_vertices: int = 7):
    """Pad a pruned NB101 graph (2-7 vertices) up to the fixed 7-vertex shape
    ``_encode`` expects, with isolated (edge-free) dummy interior vertices
    inserted just before the output node.

    Only used to produce a storable decision vector for this space -- row
    uniqueness comes from the database's own canonical fingerprint (see
    ``collect_nb101_exact``), not from this encoding, so the padding need only
    be deterministic, not injective or round-trippable back through
    ``_decode``.
    """
    k = len(ops)
    if k == n_vertices:
        return matrix, ops
    m_old = np.asarray(matrix)
    m_new = np.zeros((n_vertices, n_vertices), dtype=int)
    m_new[:k - 1, :k - 1] = m_old[:k - 1, :k - 1]      # edges among input + interior ops
    m_new[:k - 1, -1] = m_old[:k - 1, -1]              # their edges into output, shifted to col -1
    pad = n_vertices - k
    new_ops = [ops[0]] + list(ops[1:-1]) + (['conv3x3-bn-relu'] * pad) + [ops[-1]]
    return m_new, new_ops


def collect_nb101_exact(bench, log=print):
    """Enumerate NASBench-101 exactly from its own tabular database, one row
    per canonical architecture, instead of rejection-sampling the raw
    26-variable genotype encoding (cardinality ~5.1e8).

    That raw encoding is not in bijection with actual architectures: NB101's
    evaluator canonicalizes every genotype to an isomorphism/pruning-invariant
    graph fingerprint and looks up one of the paper's ~423k unique trained
    architectures by it, so most of that 5.1e8 is either invalid or a
    duplicate of one of the 423k in metric-space. Querying the database
    directly (via Django, initialized as a side effect of ``get_space``) gives
    the true, complete, deduplicated population -- every row here already
    valid by construction, so unlike ``sample_unique`` there is nothing to
    reject. The "true" (mean-over-repeats) accuracy is used for a
    deterministic reference value rather than the evaluator's single-draw
    default, since this is meant to be an exact cache, not a re-sampled query.
    """
    from nasbench101.models import NASBench101Result  # only importable post-init
    ss = bench.search_space
    fidelity = bench.evaluator.fidelity
    names = metric_names(bench)
    raw = {'err': None, 'params': None, 'flops': None}

    X_list: list = []
    F_rows: list = []
    keys_list: list = []
    qs = NASBench101Result.objects.all().values(
        'index', 'phenotype', 'flops', 'params', 'final_test_accuracy')
    for r in qs.iterator():
        ph = r['phenotype']
        matrix, ops = _pad_nb101_phenotype(ph['module_adjacency'], ph['module_operations'])
        x = ss._encode({'matrix': matrix, 'ops': ops})
        top1 = float(np.mean(r['final_test_accuracy'][f'epoch{fidelity}']))
        raw['err'], raw['params'], raw['flops'] = 1.0 - top1, r['params'], r['flops']
        X_list.append(x)
        F_rows.append([raw[n] for n in names])
        keys_list.append(r['index'])  # the DB's own isomorphism-invariant fingerprint

    X = np.asarray(X_list, dtype=int)
    F = np.asarray(F_rows, dtype=float)
    if bench.normalized_objectives:
        F = bench.normalize(F)
    keys = np.asarray(keys_list, dtype=object)
    stats = {
        'mode': 'enumerate_db',
        'n_total': int(len(X)),
        'n_invalid': 0,
        'n_kept': int(len(X)),
        'hit_target': True,
        'complete': True,
    }
    log(f'  NB101 exact DB enumeration: {len(X):,} canonical architectures')
    return X, F, keys, stats


def sample_unique(bench, n_target: int, seed: int, time_budget_s: float,
                  batch: int = 20_000, log=print, seen_keys=None,
                  target_batch_seconds: float = 10.0, min_batch: int = 500):
    """Draw up to ``n_target`` unique, finite-metric genotypes.

    Reject-and-resample in batches, deduping on the architecture key and
    dropping rows whose metrics are non-finite (invalid architectures), until
    ``n_target`` uniques are collected or ``time_budget_s`` elapses. Returns
    ``(X, F, keys, stats)`` with stats recording the real sampling effort so
    tail trustworthiness stays explicit downstream.

    ``seen_keys``, if given, pre-seeds the dedup set with architectures
    already cached from a prior run (same ``seed`` -> the RNG replays that
    prior run's exact draw sequence first, which this pre-seeding lets it
    skip without re-evaluating, before it starts drawing genuinely new
    architectures). Only newly-collected rows are returned -- the caller
    concatenates them onto the prior cache.

    ``batch`` is a ceiling, not a fixed size: the first batch is capped at
    ``min_batch`` to get a quick throughput reading, then every later batch is
    sized to take about ``target_batch_seconds`` of evaluate() time (clamped
    to [min_batch, batch]). A slow space (e.g. ~17 samples/sec) evaluating a
    flat 20,000-sized batch would overrun the time budget by up to ~20
    minutes and lose all of that batch's progress if interrupted mid-way;
    adapting the batch size keeps both the overrun and the at-risk work small
    regardless of throughput.
    """
    ss = bench.search_space
    lb = np.asarray(ss.lb).astype(int)
    ub = np.asarray(ss.ub).astype(int)
    rng = np.random.default_rng(seed)

    seen: set = set(seen_keys) if seen_keys else set()  # dedup only: valid + invalid alike
    n_prev_valid = len(seen)  # seen_keys carries only valid rows from a prior cache
    X_keep: list = []
    F_keep: list = []
    keys_keep: list = []
    n_drawn = 0
    n_invalid = 0
    n_valid_new = 0
    cur_batch = min(batch, min_batch)
    t0 = time.time()
    # Stop on VALID count reaching n_target, not raw unique-attempted count --
    # a high invalid rate (e.g. NB101's ~27%) would otherwise exhaust n_target
    # on failed draws well before n_target valid genotypes are actually found.
    while (n_prev_valid + n_valid_new) < n_target and (time.time() - t0) < time_budget_s:
        X = rng.integers(lb, ub + 1, size=(cur_batch, ss.n_var))
        n_drawn += cur_batch
        keys = _arch_keys(X)
        fresh = np.array([k not in seen for k in keys])
        if not fresh.any():
            continue
        Xf, kf = X[fresh], keys[fresh]
        t_eval0 = time.time()
        Ff = evaluate(bench, Xf)
        eval_dt = time.time() - t_eval0
        if len(Xf) and eval_dt > 0:
            rate = len(Xf) / eval_dt
            cur_batch = int(np.clip(rate * target_batch_seconds, min_batch, batch))
        finite = np.isfinite(Ff).all(axis=1)
        n_invalid += int((~finite).sum())
        for k in kf:
            seen.add(k)
        Xf, kf, Ff = Xf[finite], kf[finite], Ff[finite]
        n_valid_new += len(Xf)
        if len(Xf):
            X_keep.append(Xf)
            F_keep.append(Ff)
            keys_keep.append(kf)
        log(f'    drawn={n_drawn:,} new_valid={n_valid_new:,} '
            f'total_valid={n_prev_valid + n_valid_new:,} elapsed={time.time()-t0:.0f}s '
            f'next_batch={cur_batch:,}')
    X_all = np.concatenate(X_keep) if X_keep else np.empty((0, ss.n_var), int)
    F_all = np.concatenate(F_keep) if F_keep else np.empty((0, len(metric_names(bench))))
    keys_all = np.concatenate(keys_keep) if keys_keep else np.empty((0,), object)
    n_new_target = n_target - n_prev_valid
    if len(X_all) > n_new_target > 0:
        X_all, F_all, keys_all = X_all[:n_new_target], F_all[:n_new_target], keys_all[:n_new_target]
    stats = {
        'n_drawn': int(n_drawn),
        'n_unique_seen': int(len(seen)),
        'n_new_kept': int(len(X_all)),
        'n_invalid': int(n_invalid),
        'seconds': round(time.time() - t0, 1),
        'hit_target': bool((n_prev_valid + n_valid_new) >= n_target),
    }
    return X_all, F_all, keys_all, stats


def collect_space(bench, n_target: int, seed: int, time_budget_s: float, log=print,
                   seen_keys=None):
    """Enumerate (small spaces), enumerate-via-DB (NB101), or sample (large)
    -> (X, F, keys, stats).

    ``seen_keys`` (sample mode only) resumes a prior sampling run -- see
    :func:`sample_unique`. Enumerable/DB-exact spaces are always exhaustive
    already, so ``seen_keys`` is ignored for them.
    """
    if bench.search_space.name == 'NASBench101SearchSpace':
        return collect_nb101_exact(bench, log=log)
    if is_enumerable(bench):
        X = enumerate_all(bench)
        F = evaluate(bench, X)
        finite = np.isfinite(F).all(axis=1)
        keys = _arch_keys(X)
        stats = {
            'mode': 'enumerate',
            'n_total': int(len(X)),
            'n_invalid': int((~finite).sum()),
            'n_kept': int(finite.sum()),
            'hit_target': True,
            'complete': True,
        }
        return X[finite], F[finite], keys[finite], stats
    X, F, keys, stats = sample_unique(bench, n_target, seed, time_budget_s, log=log,
                                       seen_keys=seen_keys)
    stats['mode'] = 'sample'
    stats['complete'] = False
    return X, F, keys, stats


def evoxbench_version() -> str:
    try:
        import evoxbench
        return getattr(evoxbench, '__version__', 'unknown')
    except Exception:
        return 'unknown'


def database_fingerprint() -> str:
    """Cheap provenance hash over (relpath, size) of the EvoXBench db + data dirs."""
    h = hashlib.sha256()
    for root in (EVOX_DB_PATH, EVOX_DATA_PATH):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                fp = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(fp)
                except OSError:
                    size = -1
                rel = os.path.relpath(fp, root)
                h.update(rel.encode('utf-8', 'ignore'))
                h.update(str(size).encode())
    return h.hexdigest()[:16]


def probe_spaces() -> dict:
    """Capability probe over every eligible space (>=3 metrics available)."""
    out = {}
    for space in RICHEST_PID:
        bench = get_space(space)
        names = metric_names(bench)
        if len(names) < 3:
            continue
        out[space] = {
            'suite_pid': RICHEST_PID[space],
            'n_var': int(bench.search_space.n_var),
            'metrics': names,
            'families': {m: metric_family(m) for m in names},
            'normalized': bool(bench.normalized_objectives),
            'cardinality': cardinality(bench),
            'enumerable': is_enumerable(bench),
        }
    return out
