"""
Shared constants, vector-conversion helpers, and archive utilities for the
integer-vector NASBench-101 search space (26 genes: 5 ops + 21 edges).
"""

import numpy as np
from pymoo.core.duplicate import DuplicateElimination

from strategy.genetics.nasbench101_lib.model_spec import ModelSpec as _ModelSpec101

# Canonical op list for NASBench-101 hash_spec (must NOT include 'input'/'output')
_CANONICAL_OPS_101 = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']

# ─── search-space constants ───────────────────────────────────────────────────

N_VAR      = 26          # 5 op genes + 21 edge genes
N_OPS      = 5
N_EDGES    = 21
MIN_PARAMS = 227_274
MAX_PARAMS = 49_979_274

XL = np.array([0] * N_OPS + [0] * N_EDGES, dtype=int)
XU = np.array([2] * N_OPS + [1] * N_EDGES, dtype=int)


# ─── vector conversion helpers ────────────────────────────────────────────────

def _prog_to_vec(prog) -> np.ndarray:
    """NASBENCH101 instance -> 26-element int vector (ops ‖ edges)."""
    return np.concatenate([prog.nb_ops, prog.nb_edges]).astype(int)


def _vec_to_arch_str(vec: np.ndarray, bench_db: dict) -> 'str | None':
    """Integer vector -> arch hash (MD5 of canonical ModelSpec)."""
    ops_idx = vec[:N_OPS]
    edges   = vec[N_OPS:]
    ops = ['input'] + [_CANONICAL_OPS_101[int(o)] for o in ops_idx] + ['output']
    mat = np.zeros((7, 7), dtype=int)
    k = 0
    for i in range(7):
        for j in range(i + 1, 7):
            mat[i, j] = int(edges[k])
            k += 1
    spec = _ModelSpec101(matrix=mat, ops=ops)
    if not spec.valid_spec:
        return None
    return spec.hash_spec(_CANONICAL_OPS_101)


def _is_valid_vec(vec: np.ndarray) -> bool:
    ops_idx = vec[:N_OPS]
    edges   = vec[N_OPS:]
    ops = ['input'] + [_CANONICAL_OPS_101[int(o)] for o in ops_idx] + ['output']
    mat = np.zeros((7, 7), dtype=int)
    k = 0
    for i in range(7):
        for j in range(i + 1, 7):
            mat[i, j] = int(edges[k])
            k += 1
    spec = _ModelSpec101(matrix=mat, ops=ops)
    return spec.valid_spec and int(np.sum(edges)) <= 9

# ─── archive helpers ──────────────────────────────────────────────────────────

def _update_archive(var_arch, obj_arch, var_new, obj_new):
    """Non-dominated archive update (identical to PDNS update_elitist_archive)."""
    var_arch = list(var_arch)
    obj_arch = list(obj_arch)

    dominated_idx = []
    new_dominated  = False

    for idx, obj_existing in enumerate(obj_arch):
        if obj_new[0] >= obj_existing[0] and obj_new[1] >= obj_existing[1]:
            new_dominated = True
            break
        if obj_existing[0] >= obj_new[0] and obj_existing[1] >= obj_new[1]:
            dominated_idx.append(idx)

    if not new_dominated:
        for idx in sorted(dominated_idx, reverse=True):
            var_arch.pop(idx)
            obj_arch.pop(idx)
        var_arch.append(var_new)
        obj_arch.append(obj_new)

    return var_arch, obj_arch


def _test_archive(var_arch, bench_db):
    """Evaluate elitist archive on test_acc_108 and return non-dominated front."""
    test_objs = []
    for vec in var_arch:
        arch_str = _vec_to_arch_str(np.array(vec), bench_db)
        if arch_str is None or arch_str not in bench_db:
            test_objs.append((1.0, 1.0))
            continue
        entry = bench_db[arch_str]
        test_err = 1.0 - entry.get('test_acc_108', 0.0)
        n_params_norm = (entry['n_params'] - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)
        test_objs.append((test_err, n_params_norm))

    # keep only non-dominated
    nd_var, nd_obj = [], []
    for i, (v, o) in enumerate(zip(var_arch, test_objs)):
        dominated = any(
            other[0] <= o[0] and other[1] <= o[1] and other != o
            for other in test_objs
        )
        if not dominated:
            nd_var.append(v)
            nd_obj.append(o)
    return np.array(nd_var) if nd_var else np.empty((0, N_VAR)), \
           np.array(nd_obj)  if nd_obj  else np.empty((0, 2))
