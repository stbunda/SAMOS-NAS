"""
problem/evoxbench/benchmark_meta.py — Static metadata for every EvoXBench problem.

Fields per entry
----------------
label      : Short display name, e.g. "C-10/MOP1"
search_space: Human-readable search space abbreviation
n_var      : Number of decision variables
n_obj      : Number of objectives
"""

# ─── search-space class name → abbreviation ───────────────────────────────────

_SS_ABBREV = {
    'NASBench101SearchSpace': 'NB101',
    'NATSBenchSearchSpace':   'NATS',
    'NASBench201SearchSpace': 'NB201',
    'DARTSSearchSpace':       'DARTS',
    'ResNet50DSearchSpace':   'ResNet-50D',
    'TransformerSearchSpace': 'Transformer',
    'MobileNetV3SearchSpace': 'MobileNetV3',
}

# ─── per-suite, per-pid metadata ─────────────────────────────────────────────
#
# Keyed as  BENCHMARK_META[suite][pid]
# Values are (label, search_space_abbrev, n_var, n_obj)

BENCHMARK_META: dict[str, dict[int, dict]] = {
    'c10mop': {
        1: {'label': 'C-10/MOP1',  'search_space': 'NB101',  'n_var': 26, 'n_obj': 2},
        2: {'label': 'C-10/MOP2',  'search_space': 'NB101',  'n_var': 26, 'n_obj': 3},
        3: {'label': 'C-10/MOP3',  'search_space': 'NATS',   'n_var':  5, 'n_obj': 3},
        4: {'label': 'C-10/MOP4',  'search_space': 'NATS',   'n_var':  5, 'n_obj': 4},
        5: {'label': 'C-10/MOP5',  'search_space': 'NB201',  'n_var':  6, 'n_obj': 5},
        6: {'label': 'C-10/MOP6',  'search_space': 'NB201',  'n_var':  6, 'n_obj': 6},
        7: {'label': 'C-10/MOP7',  'search_space': 'NB201',  'n_var':  6, 'n_obj': 8},
        8: {'label': 'C-10/MOP8',  'search_space': 'DARTS',  'n_var': 32, 'n_obj': 2},
        9: {'label': 'C-10/MOP9',  'search_space': 'DARTS',  'n_var': 32, 'n_obj': 3},
    },
    'in1kmop': {
        1: {'label': 'IN-1k/MOP1', 'search_space': 'ResNet-50D',   'n_var': 25, 'n_obj': 2},
        2: {'label': 'IN-1k/MOP2', 'search_space': 'ResNet-50D',   'n_var': 25, 'n_obj': 2},
        3: {'label': 'IN-1k/MOP3', 'search_space': 'ResNet-50D',   'n_var': 25, 'n_obj': 3},
        4: {'label': 'IN-1k/MOP4', 'search_space': 'Transformer',  'n_var': 34, 'n_obj': 2},
        5: {'label': 'IN-1k/MOP5', 'search_space': 'Transformer',  'n_var': 34, 'n_obj': 2},
        6: {'label': 'IN-1k/MOP6', 'search_space': 'Transformer',  'n_var': 34, 'n_obj': 3},
        7: {'label': 'IN-1k/MOP7', 'search_space': 'MobileNetV3',  'n_var': 21, 'n_obj': 2},
        8: {'label': 'IN-1k/MOP8', 'search_space': 'MobileNetV3',  'n_var': 21, 'n_obj': 3},
        9: {'label': 'IN-1k/MOP9', 'search_space': 'MobileNetV3',  'n_var': 21, 'n_obj': 4},
    },
}


def pid_header(suite: str, pid: int) -> str:
    """Return the full block header string for a LaTeX table cell.

    Example: ``C-10/MOP1 \\\\ (NB101, 26 vars, 2 obj)``
    Falls back to ``PID {pid}`` for unknown suite / pid.
    """
    meta = BENCHMARK_META.get(suite, {}).get(pid)
    if meta is None:
        return f'PID {pid}'
    return (
        f"\\makecell[c]{{\\textbf{{{meta['label']}}} \\\\ "
        f"({meta['search_space']}, "
        f"{meta['n_var']}\\,vars, "
        f"{meta['n_obj']}\\,obj)}}"
    )
