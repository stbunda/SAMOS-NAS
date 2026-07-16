"""
Utilities for the evoxbench integration.
"""

import os

# ─── data paths ───────────────────────────────────────────────────────────────

# When running as a SLURM array job, multiple tasks on the same node share
# ~/.config/evoxbench/config.json.  Each task writes its scratch-local data
# path, but concurrent writes create a race: task A's config gets overwritten
# by task B before task A constructs its benchmark, causing FileNotFoundError.
#
# Fix: always resolve data paths relative to SLURM_SUBMIT_DIR (the stable
# project root on the shared filesystem).  All concurrent tasks then write
# the *same* value to the config file, so writes are idempotent and races
# are harmless.  When not running under SLURM (local dev), fall back to the
# path derived from this file's location.
_SUBMIT_DIR   = os.environ.get('SLURM_SUBMIT_DIR')
_PROJECT_ROOT = _SUBMIT_DIR if _SUBMIT_DIR else \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVOX_DB_PATH   = os.path.join(_PROJECT_ROOT, 'problem', 'data', 'evoxbench', 'database')
EVOX_DATA_PATH = os.path.join(_PROJECT_ROOT, 'problem', 'data', 'evoxbench', 'data')

_initialized = False


def init_evoxbench() -> None:
    """Initialise the evoxbench database. Idempotent — safe to call multiple times."""
    global _initialized
    if _initialized:
        return
    from evoxbench.database.init import config as _evox_config
    _evox_config(EVOX_DB_PATH, EVOX_DATA_PATH)
    _initialized = True


# ─── benchmark factory ────────────────────────────────────────────────────────

_SUITE_FACTORIES = {
    'c10mop':     lambda pid: __import__('evoxbench.test_suites', fromlist=['c10mop']).c10mop(pid),
    'in1kmop':    lambda pid: __import__('evoxbench.test_suites', fromlist=['in1kmop']).in1kmop(pid),
    'citysegmop': lambda pid: __import__('evoxbench.test_suites', fromlist=['citysegmop']).citysegmop(pid),
}


def _patch_mnv3_decode(benchmark) -> None:
    """Make the MobileNetV3 genotype->architecture mapping deterministic.

    evoxbench's ``MobileNetV3SearchSpace.var2str`` substitutes a RANDOM
    operation whenever a layer gene is 0, and ``_decode``'s depth semantics
    ("use the first d layers of each stage") treat a skipped slot that
    precedes an active one as ACTIVE. A genotype that skips a layer before
    an active one in the same stage (~36% of the uniform space; routinely
    produced by integer crossover/mutation) therefore decodes to a
    different architecture on every call -- the random substitution is
    materialized and the explicitly specified trailing gene is dropped --
    so ``benchmark.evaluate`` is not a function of X: even #Params changes
    between two evaluations of the same genotype.

    The library's own ``_sample`` repairs exactly this pattern before
    decoding (the active layer is shifted up into the skipped slot); this
    wrapper applies that identical repair to EVERY decode, so ill-formed
    genotypes map deterministically to the same architecture the repaired
    sample would. The random filler that ``var2str`` puts in INACTIVE
    (beyond-depth) ks/e slots is pinned to a fixed value as well -- every
    predictor masks those slots by stage depth, so objective values are
    unaffected, but the decoded phenotype becomes a pure function of the
    genotype. No-op for non-MNV3 search spaces.
    """
    import numpy as np

    ss = getattr(benchmark, 'search_space', None)
    if type(ss).__name__ != 'MobileNetV3SearchSpace' or getattr(ss, '_decode_repaired', False):
        return
    orig_decode = ss._decode

    def _det_var2str(v, ub):
        if v > 0:
            return ss.var2str_mapping[v]
        return ss.var2str_mapping[1]     # inactive-slot filler: fixed, depth-masked everywhere

    def _repaired_decode(x):
        x = np.asarray(x, dtype=int).copy()
        for indices in ss.stage_layer_indices:
            if x[indices[-2]] == 0 and x[indices[-1]] > 0:
                x[indices[-2]] = x[indices[-1]]
                x[indices[-1]] = 0
        return orig_decode(x)

    ss.var2str = _det_var2str
    ss._decode = _repaired_decode
    ss._decode_repaired = True


def get_benchmark(suite: str, pid: int):
    """Return the evoxbench benchmark for the given suite and problem id.

    n_var, n_obj, lb, ub, pareto_front, and objectives are all provided by
    the returned benchmark object — no extra configuration is required.

    Parameters
    ----------
    suite : str
        One of ``'c10mop'``, ``'in1kmop'``, ``'citysegmop'``.
    pid : int
        Problem id within the suite (e.g. 1–9 for c10mop / in1kmop,
        1–15 for citysegmop).
    """
    init_evoxbench()
    if suite not in _SUITE_FACTORIES:
        raise ValueError(
            f"Unknown suite {suite!r}. Choose from: {list(_SUITE_FACTORIES)}"
        )
    benchmark = _SUITE_FACTORIES[suite](pid)
    _patch_mnv3_decode(benchmark)
    return benchmark
