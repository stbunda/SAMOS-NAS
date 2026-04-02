"""
Utilities for the evoxbench integration.
"""

import os

# ─── data paths ───────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    return _SUITE_FACTORIES[suite](pid)
