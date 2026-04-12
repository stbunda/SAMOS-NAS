"""
analysis/convergence.py — Combined Pareto approximation for fair cross-method comparison.

For each (suite, pid, budget), pools the final cumulative test-objective archives
from *all* methods and seeds, computes the non-dominated front of that union, and
derives a shared nadir / reference point.  Indicator trajectories and final-gen
statistics can then be recomputed on-the-fly against this single shared reference,
making HV and IGD+ values directly comparable across methods.

Cache
-----
The combined Pareto approximation is stored next to the per-method result folders:

    results/evoxbench/<suite>/pid<pid>/B<budget>_P<pop_size>/pareto_approx.pkl

The cache records, for every contributing file, its absolute path and mtime at
build time.  On the next call the cache is validated entry-by-entry; any missing
or modified file causes a full rebuild.

Public API
----------
build_pareto_approximation(suite, pid, methods, pop_size, n_gen,
                            force_rebuild=False) -> dict | None
recompute_indicator_trajectories(method, n_gen, results_root,
                                  ref_point, pareto_approx) -> tuple | None
recompute_final_indicators(method, results_root,
                            ref_point, pareto_approx) -> dict | None
"""

import os
import pickle
import tempfile

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# ─── fast non-dominated sort ──────────────────────────────────────────────────

def _nd_filter_2d(F: np.ndarray) -> np.ndarray:
    """O(n log n) non-dominated filter for 2-objective minimization.

    Sort by first objective ascending (ties broken by second ascending),
    then scan forward tracking the running minimum of the second objective.
    A point is on the Pareto front iff its second-objective value is strictly
    below the minimum seen in all preceding points.
    """
    sort_idx = np.lexsort((F[:, 1], F[:, 0]))   # f0 asc, then f1 asc for ties
    nd_list  = []
    min_f1   = np.inf
    for global_i in sort_idx:
        f1_i = F[global_i, 1]
        if f1_i < min_f1:
            nd_list.append(global_i)
            min_f1 = f1_i
    return np.array(nd_list, dtype=int)


def _nd_filter(F: np.ndarray) -> np.ndarray:
    """Non-dominated filter for arbitrary number of objectives.

    Uses the fast 2D sort-scan for d=2 (O(n log n)) and vectorised
    numpy broadcasting with chunking for d>2 (O(n²d) but cache-friendly).
    Falls back to pymoo NonDominatedSorting for very large inputs.
    """
    n, d = F.shape
    if n <= 1:
        return np.arange(n)
    if d == 2:
        return _nd_filter_2d(F)

    # Vectorised chunk approach — bound peak memory to ~CHUNK×n×d×8 bytes
    CHUNK       = 512
    is_dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, CHUNK):
        end  = min(start + CHUNK, n)
        # diff[local_i, j, k] = F[start+local_i, k] - F[j, k]
        diff = F[start:end, None, :] - F[None, :, :]   # (chunk, n, d)
        # j dominates local_i: diff≥0 all dims AND diff>0 some dim
        dom  = (diff >= 0).all(axis=2) & (diff > 0).any(axis=2)  # (chunk, n)
        # Remove self-dominance on the block diagonal
        dom[np.arange(end - start), np.arange(start, end)] = False
        is_dominated[start:end] = dom.any(axis=1)
    return np.where(~is_dominated)[0]


# ─── internal path helper (mirrors analyze_evoxbench._results_root) ───────────

def _results_root(suite: str, pid: int, pop_size: int, n_gen: int) -> str:
    return os.path.join(
        'results', 'evoxbench', suite,
        f'pid{pid}', f'B{n_gen * pop_size}_P{pop_size}',
    )


# ─── per-seed archive loading ─────────────────────────────────────────────────

def _load_final_test_archive(seed_dir: str):
    """Yield ``(F_array, abs_path)`` pairs from every ``seed_*.pkl`` in *seed_dir*.

    Loads ``data['test_obj_archive'][-1]`` — the final-generation cumulative
    non-dominated archive re-evaluated with ``true_eval=True``, already
    normalized by the benchmark.  Entries that are missing, empty, or
    non-finite are skipped with a warning.
    """
    if not os.path.isdir(seed_dir):
        return
    for fname in sorted(os.listdir(seed_dir)):
        if not fname.endswith('.pkl'):
            continue
        abs_path = os.path.abspath(os.path.join(seed_dir, fname))
        try:
            with open(abs_path, 'rb') as fh:
                data = pickle.load(fh)
        except Exception as exc:
            print(f'  [convergence] WARN: failed to load {abs_path}: {exc}')
            continue

        archives = data.get('test_obj_archive', [])
        if not archives:
            print(f'  [convergence] WARN: no test_obj_archive in {abs_path}, skipping.')
            continue
        F = archives[-1]
        if F is None or (hasattr(F, '__len__') and len(F) == 0):
            print(f'  [convergence] WARN: empty final archive in {abs_path}, skipping.')
            continue
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or F.shape[0] == 0:
            print(f'  [convergence] WARN: unexpected archive shape {F.shape} in {abs_path}, skipping.')
            continue
        yield F, abs_path


# ─── cache helpers ────────────────────────────────────────────────────────────

def _cache_path(results_root: str) -> str:
    return os.path.join(results_root, 'pareto_approx.pkl')


def _load_cache(results_root: str) -> dict | None:
    """Return the cached approximation dict if it exists and is still valid.

    Validity: every source entry ``{seed_file, mtime}`` must still exist on
    disk and must *not* have been modified since the cache was written
    (i.e. current mtime <= stored mtime, with a 1 ms tolerance).
    """
    path = _cache_path(results_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as fh:
            cache = pickle.load(fh)
    except Exception as exc:
        print(f'  [convergence] WARN: could not read cache {path}: {exc}')
        return None

    for entry in cache.get('sources', []):
        seed_file = entry['seed_file']
        if not os.path.isfile(seed_file):
            print(f'  [convergence] Cache stale: {seed_file} no longer exists.')
            return None
        current_mtime = os.path.getmtime(seed_file)
        if current_mtime > entry['mtime'] + 1e-3:
            print(f'  [convergence] Cache stale: {seed_file} has been modified.')
            return None

    n_methods = len({e['method'] for e in cache.get('sources', [])})
    n_seeds   = len(cache.get('sources', []))
    print(
        f'  [convergence] Cache hit: {path}\n'
        f'                {len(cache["pareto_approx"])} PF points '
        f'from {n_methods} methods, {n_seeds} seed files.'
    )
    return cache


def _save_cache(results_root: str, approx_info: dict) -> None:
    """Atomically write *approx_info* to the cache file."""
    path    = _cache_path(results_root)
    dir_    = os.path.dirname(path) or '.'
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            pickle.dump(approx_info, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)           # atomic on POSIX; best-effort on Windows
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(f'  [convergence] Cache saved -> {path}')


# ─── public API ───────────────────────────────────────────────────────────────

def compute_empirical_norm_bounds(
    methods: list,
    results_root: str,
) -> dict | None:
    """Derive per-objective normalization bounds from the raw data.

    Pools every finite value in ``test_obj_archive[-1]`` (final cumulative
    non-dominated archive, ``true_eval=True``) across all methods and seeds
    under *results_root*, and returns the per-objective min/max.

    Parameters
    ----------
    methods :
        List of method directory names inside *results_root*.
    results_root :
        The ``B{budget}_P{pop_size}`` directory containing per-method folders.

    Returns
    -------
    ``{'obj_min': (n_obj,) ndarray, 'obj_max': (n_obj,) ndarray}``
    or ``None`` if no data could be loaded.
    """
    all_points: list[np.ndarray] = []
    for method in methods:
        seed_dir = os.path.join(results_root, method)
        for F, _ in _load_final_test_archive(seed_dir):
            # F may contain non-finite values (raw mode may have NaN for
            # architectures that could not be evaluated); keep finite rows only.
            finite = np.isfinite(F).all(axis=1)
            F = F[finite]
            if len(F) > 0:
                all_points.append(F)

    if not all_points:
        print(
            f'  [convergence] WARN: no data found in {results_root}; '
            'cannot compute empirical norm bounds.'
        )
        return None

    combined = np.vstack(all_points)
    obj_min  = combined.min(axis=0)
    obj_max  = combined.max(axis=0)
    # Guard against degenerate case where all values are identical
    flat = (obj_max - obj_min) < 1e-12
    obj_max[flat] = obj_min[flat] + 1.0
    print(
        f'  [convergence] Empirical norm bounds from {results_root}:\n'
        f'                obj_min = {np.round(obj_min, 4)}\n'
        f'                obj_max = {np.round(obj_max, 4)}'
    )
    return {'obj_min': obj_min, 'obj_max': obj_max}


def _apply_norm_bounds(F: np.ndarray, norm_bounds: dict) -> np.ndarray:
    """Linearly scale *F* into [0, 1] using pre-computed empirical bounds."""
    obj_min = norm_bounds['obj_min']
    obj_max = norm_bounds['obj_max']
    return (F - obj_min) / (obj_max - obj_min)


def build_pareto_approximation(
    suite: str,
    pid: int,
    methods: list,
    pop_size: int,
    n_gen: int,
    force_rebuild: bool = False,
    norm_bounds: dict | None = None,
    results_root: str | None = None,
) -> dict | None:
    """Build (or load from cache) the combined Pareto approximation for one PID.

    Pools ``test_obj_archive[-1]`` (final cumulative ND archive, true-eval)
    from every seed file of every method, optionally normalizes with
    *norm_bounds*, computes the non-dominated front of the union, and derives
    nadir and reference point.

    Parameters
    ----------
    suite, pid, pop_size, n_gen
        Identify which results folder to use (ignored when *results_root* is
        given explicitly).
    methods
        List of method directory names inside the results root.
    force_rebuild
        If ``True``, ignore any on-disk cache and rebuild from scratch.
    norm_bounds
        When provided (``{'obj_min': array, 'obj_max': array}`` from
        :func:`compute_empirical_norm_bounds`), each loaded archive is
        linearly rescaled into ``[0, 1]`` before the Pareto front is computed.
        The bounds are stored inside the returned dict and the on-disk cache.
    results_root
        Override the folder derived from *suite/pid/pop_size/n_gen*.  Useful
        when results live under a non-standard root (e.g. ``evoxbench_no_norm``).

    Returns
    -------
    dict with keys:
        ``pareto_approx`` — ``(n_pts, n_obj)`` ndarray, the combined ND front.
        ``nadir``         — ``(n_obj,)`` per-objective maximum of the PF.
        ``ref_point``     — ``(n_obj,)`` reference point.
        ``sources``       — list of ``{method, seed_file, mtime}`` dicts.
        ``norm_bounds``   — the *norm_bounds* dict used (or ``None``).
    Returns ``None`` if no data could be loaded from any method/seed.
    """
    root = results_root if results_root is not None else _results_root(suite, pid, pop_size, n_gen)

    if not force_rebuild:
        cached = _load_cache(root)
        if cached is not None:
            # Invalidate if norm_bounds mode has changed (None ↔ provided)
            cached_nb = cached.get('norm_bounds')
            nb_match = (
                (norm_bounds is None and cached_nb is None)
                or (
                    norm_bounds is not None
                    and cached_nb is not None
                    and np.allclose(cached_nb['obj_min'], norm_bounds['obj_min'])
                    and np.allclose(cached_nb['obj_max'], norm_bounds['obj_max'])
                )
            )
            if nb_match:
                return cached
            print(
                '  [convergence] Cache stale: norm_bounds have changed; rebuilding.'
            )

    all_points: list[np.ndarray] = []
    sources: list[dict] = []

    for method in methods:
        seed_dir = os.path.join(root, method)
        for F, abs_path in _load_final_test_archive(seed_dir):
            if norm_bounds is not None:
                finite = np.isfinite(F).all(axis=1)
                F = F[finite]
                if len(F) == 0:
                    continue
                F = _apply_norm_bounds(F, norm_bounds)
            all_points.append(F)
            sources.append({
                'method':    method,
                'seed_file': abs_path,
                'mtime':     os.path.getmtime(abs_path),
            })

    if not all_points:
        print(f'  [convergence] WARN: no data found for {suite}/pid{pid}; '
              f'cannot build Pareto approximation.')
        return None

    combined = np.vstack(all_points)

    # Drop rows containing any non-finite value, then deduplicate
    finite_mask = np.isfinite(combined).all(axis=1)
    combined    = combined[finite_mask]
    combined    = np.unique(combined, axis=0)   # remove duplicate points first

    if len(combined) == 0:
        print(f'  [convergence] WARN: all points non-finite for {suite}/pid{pid}.')
        return None

    nd_idx        = _nd_filter(combined)
    pareto_approx = combined[nd_idx]

    nadir     = np.max(pareto_approx, axis=0)
    ref_point = np.maximum(nadir * 1.05, nadir + 1e-6)

    n_methods_found = len({s['method'] for s in sources})
    print(
        f'  [convergence] Built Pareto approx for {suite}/pid{pid}: '
        f'{len(pareto_approx)} PF points from '
        f'{n_methods_found} methods, {len(sources)} seed files.\n'
        f'                nadir     = {np.round(nadir, 4)}\n'
        f'                ref_point = {np.round(ref_point, 4)}'
    )

    result = {
        'pareto_approx': pareto_approx,
        'nadir':         nadir,
        'ref_point':     ref_point,
        'sources':       sources,
        'norm_bounds':   norm_bounds,
    }
    _save_cache(root, result)
    return result


def recompute_indicator_trajectories(
    method: str,
    n_gen: int,
    results_root: str,
    ref_point: np.ndarray,
    pareto_approx: np.ndarray,
    norm_bounds: dict | None = None,
) -> tuple | None:
    """Recompute HV / IGD+ trajectories using the shared reference / PF.

    Reads ``test_obj_archive`` (one array per generation) from every
    ``seed_*.pkl`` under ``results_root/method``, recomputes HV and IGD+
    at each generation using *ref_point* and *pareto_approx*, then
    aggregates over seeds.

    When *norm_bounds* is provided, each generation's archive is linearly
    rescaled into ``[0, 1]`` before computing indicators — use this for
    runs that stored raw (un-normalized) objectives.

    Uses the same downsample / pad logic as
    ``plotter.load_indicator_trajectories`` so the result is directly
    substitutable in all plotting calls.

    Returns
    -------
    ``(hv_mean, hv_std, igd_mean, igd_std)`` — each a ``(n_gen,)`` ndarray.
    ``None`` if no seed files are found for *method*.
    """
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None

    hv_ind  = HV(ref_point=ref_point)
    igd_ind = IGDPlus(pareto_approx)

    hv_runs, igd_runs = [], []

    for fname in sorted(os.listdir(seed_dir)):
        if not fname.endswith('.pkl'):
            continue
        abs_path = os.path.join(seed_dir, fname)
        try:
            with open(abs_path, 'rb') as fh:
                data = pickle.load(fh)
        except Exception as exc:
            print(f'  [convergence] WARN: failed to load {abs_path}: {exc}')
            continue

        archives = data.get('test_obj_archive', [])
        if not archives:
            continue

        hv_series, igd_series = [], []
        # Fast duplicate-skip: cache the last (shape, checksum) → (hv, igd+).
        # Consecutive identical archives produce the same indicators.
        prev_key = None
        prev_hv  = None
        prev_igd = None
        for F_gen in archives:
            if F_gen is None or len(F_gen) == 0:
                hv_series.append(0.0)
                igd_series.append(np.nan)
                prev_key = None
                continue
            F_gen = np.asarray(F_gen, dtype=float)
            if norm_bounds is not None:
                finite = np.isfinite(F_gen).all(axis=1)
                F_gen = F_gen[finite]
                if len(F_gen) == 0:
                    hv_series.append(0.0)
                    igd_series.append(np.nan)
                    prev_key = None
                    continue
                F_gen = _apply_norm_bounds(F_gen, norm_bounds)
            else:
                F_gen = np.where(np.isfinite(F_gen), F_gen, 1.0)
            if len(F_gen) == 0:
                hv_series.append(0.0)
                igd_series.append(np.nan)
                prev_key = None
                continue
            # Cheap structural fingerprint — shape + element sum
            key = (F_gen.shape, float(F_gen.sum()))
            if key == prev_key:
                hv_series.append(prev_hv)
                igd_series.append(prev_igd)
                continue
            prev_hv  = float(hv_ind(F_gen))
            prev_igd = float(igd_ind(F_gen))
            prev_key = key
            hv_series.append(prev_hv)
            igd_series.append(prev_igd)

        # Downsample / pad — identical logic to load_indicator_trajectories
        if len(hv_series) == n_gen:
            hv_runs.append(hv_series)
            igd_runs.append(igd_series)
        elif len(hv_series) >= n_gen:
            step = len(hv_series) // n_gen
            hv_runs.append([
                hv_series[min((g + 1) * step - 1, len(hv_series) - 1)]
                for g in range(n_gen)
            ])
            igd_runs.append([
                igd_series[min((g + 1) * step - 1, len(igd_series) - 1)]
                for g in range(n_gen)
            ])
        else:
            hv_runs.append(hv_series)
            igd_runs.append(igd_series)

    if not hv_runs:
        return None

    max_len = max(len(r) for r in hv_runs)
    for r in hv_runs:
        while len(r) < max_len:
            r.append(r[-1])
    for r in igd_runs:
        while len(r) < max_len:
            r.append(r[-1])

    hv_arr  = np.array(hv_runs,  dtype=float)
    igd_arr = np.array(igd_runs, dtype=float)
    return (
        hv_arr.mean(axis=0),  hv_arr.std(axis=0),
        igd_arr.mean(axis=0), igd_arr.std(axis=0),
    )


def recompute_final_indicators(
    method: str,
    results_root: str,
    ref_point: np.ndarray,
    pareto_approx: np.ndarray,
    norm_bounds: dict | None = None,
) -> dict | None:
    """Recompute final-generation HV / IGD+ using the shared reference / PF.

    When *norm_bounds* is provided, each loaded archive is rescaled before
    computing indicators (use for raw / un-normalized runs).
    """
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None

    hv_ind  = HV(ref_point=ref_point)
    igd_ind = IGDPlus(pareto_approx)

    hvs, igds = [], []
    for F, _ in _load_final_test_archive(seed_dir):
        if norm_bounds is not None:
            finite = np.isfinite(F).all(axis=1)
            F = F[finite]
            if len(F) == 0:
                continue
            F = _apply_norm_bounds(F, norm_bounds)
        else:
            F = np.where(np.isfinite(F), F, 1.0)
        if len(F) == 0:
            continue
        hvs.append(float(hv_ind(F)))
        igds.append(float(igd_ind(F)))

    if not hvs:
        return None
    return {
        'hv':       (float(np.mean(hvs)),  float(np.std(hvs))),
        'igd_plus': (float(np.mean(igds)), float(np.std(igds))),
    }


def recompute_final_indicators_seeds(
    method: str,
    results_root: str,
    ref_point: np.ndarray,
    pareto_approx: np.ndarray,
    norm_bounds: dict | None = None,
) -> dict | None:
    """Like recompute_final_indicators but returns per-seed raw arrays.

    When *norm_bounds* is provided, each loaded archive is rescaled before
    computing indicators (use for raw / un-normalized runs).

    Returns
    -------
    ``{'hv': np.ndarray, 'igd_plus': np.ndarray}`` or ``None`` if no data.
    """
    seed_dir = os.path.join(results_root, method)
    if not os.path.isdir(seed_dir):
        return None

    hv_ind  = HV(ref_point=ref_point)
    igd_ind = IGDPlus(pareto_approx)

    hvs, igds = [], []
    for F, _ in _load_final_test_archive(seed_dir):
        if norm_bounds is not None:
            finite = np.isfinite(F).all(axis=1)
            F = F[finite]
            if len(F) == 0:
                continue
            F = _apply_norm_bounds(F, norm_bounds)
        else:
            F = np.where(np.isfinite(F), F, 1.0)
        if len(F) == 0:
            continue
        hvs.append(float(hv_ind(F)))
        igds.append(float(igd_ind(F)))

    if not hvs:
        return None
    return {
        'hv':       np.array(hvs),
        'igd_plus': np.array(igds),
    }


# ─── trajectory cache helpers ─────────────────────────────────────────────────

def get_or_recompute_trajectories(
    method: str,
    n_gen: int,
    results_root: str,
    approx_info: dict,
    norm_bounds: dict | None = None,
) -> tuple | None:
    """Return trajectory for *method*, serving from / updating the trajectory
    cache embedded in *approx_info*.

    The first call computes the trajectory (expensive) and stores the result
    in ``approx_info['traj_cache'][method]``.  Subsequent calls for the same
    method return the cached value instantly.  Call :func:`save_approx_cache`
    after processing all methods to persist the updated cache to disk.

    Parameters
    ----------
    method : str
        Method directory name inside *results_root*.
    n_gen : int
        Number of generations (for the downsample / pad logic).
    results_root : str
        The ``B{budget}_P{pop_size}`` directory that contains per-method
        sub-directories and the ``pareto_approx.pkl`` cache file.
    approx_info : dict
        The dict returned by :func:`build_pareto_approximation`.  Must
        contain ``ref_point`` and ``pareto_approx``.
    norm_bounds : dict | None
        When provided, raw objectives are rescaled before computing indicators.
        See :func:`compute_empirical_norm_bounds`.

    Returns
    -------
    ``(hv_mean, hv_std, igd_mean, igd_std)`` or ``None``.
    """
    cache = approx_info.setdefault('traj_cache', {})
    if method in cache:
        return cache[method]

    traj = recompute_indicator_trajectories(
        method, n_gen, results_root,
        approx_info['ref_point'],
        approx_info['pareto_approx'],
        norm_bounds=norm_bounds,
    )
    cache[method] = traj
    return traj


def save_approx_cache(results_root: str, approx_info: dict) -> None:
    """Persist *approx_info* (including any accumulated trajectory cache) to
    the ``pareto_approx.pkl`` file in *results_root*."""
    _save_cache(results_root, approx_info)
