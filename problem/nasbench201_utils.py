"""
Shared constants, vector-conversion helpers, and archive utilities for the
NASBench-201 search space (6 categorical operation genes, 5 options each).

Each architecture is a 6-element integer vector encoding the operation
assigned to each directed edge in the 4-node cell DAG:

  edge 0: node1 ← node0        (gene 0)
  edge 1: node2 ← node0        (gene 1)
  edge 2: node2 ← node1        (gene 2)
  edge 3: node3 ← node0        (gene 3)
  edge 4: node3 ← node1        (gene 4)
  edge 5: node3 ← node2        (gene 5)

Resulting arch_str format:
  |op0~0|+|op1~0|op2~1|+|op3~0|op4~1|op5~2|

All 5^6 = 15,625 vectors are valid — no validity check is needed.
"""

import numpy as np

# ─── search-space constants ───────────────────────────────────────────────────

OP_LIST        = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
N_OPS_PER_GENE = len(OP_LIST)   # 5
N_VAR          = 6              # 6 categorical genes

XL = np.zeros(N_VAR, dtype=int)
XU = np.full(N_VAR, N_OPS_PER_GENE - 1, dtype=int)   # [4, 4, 4, 4, 4, 4]

# ─── Dataset info ─────────────────────────────────────────────────────────────
# Maps the user-facing dataset name to:
#   val_key  — sub-dict key inside bench_db[arch_str] used for validation fitness
#   test_key — sub-dict key used for the test-accuracy Pareto reference front
DATASET_INFO = {
    'cifar10-valid':  {'val_key': 'cifar10-valid', 'test_key': 'cifar10'},
    'cifar100':       {'val_key': 'cifar100',       'test_key': 'cifar100'},
    'ImageNet16-120': {'val_key': 'ImageNet16-120', 'test_key': 'ImageNet16-120'},
}

VALID_DATASETS = list(DATASET_INFO.keys())


# ─── Architecture string helpers ─────────────────────────────────────────────

def _vec_to_arch_str(vec) -> str:
    """Convert a 6-gene integer vector to the NASBench-201 arch_str.

    Format:  |op0~0|+|op1~0|op2~1|+|op3~0|op4~1|op5~2|
    """
    ops = [OP_LIST[int(round(float(v)))] for v in vec]
    return (
        f"|{ops[0]}~0|+"
        f"|{ops[1]}~0|{ops[2]}~1|+"
        f"|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
    )


# ─── FLOPs range helper ────────────────────────────────────────────────────────

def compute_flops_range(bench_db: dict, dataset_key: str):
    """Return ``(min_flops, max_flops)`` for *dataset_key*, computed from bench_db."""
    all_flops = np.array([
        v[dataset_key]['flops']
        for v in bench_db.values()
        if dataset_key in v and 'flops' in v[dataset_key]
    ])
    return float(all_flops.min()), float(all_flops.max())


def build_test_pareto_ref(
    bench_db: dict,
    test_key: str,
    test_min_flops: float,
    test_max_flops: float,
) -> np.ndarray:
    """Build the test-acc Pareto front (test_err × flops_norm) from all architectures.

    Fast O(n log n) sweep — works for all three NASBench-201 datasets.
    Returns an ``(M, 2)`` array of non-dominated points.
    """
    F_all = np.array([
        [
            1.0 - v[test_key]['epoch-200'] / 100.0,
            (v[test_key]['flops'] - test_min_flops) / (test_max_flops - test_min_flops),
        ]
        for v in bench_db.values()
        if test_key in v
    ])
    order     = np.argsort(F_all[:, 0], kind='stable')
    F_sorted  = F_all[order]
    nd_mask   = np.zeros(len(F_sorted), dtype=bool)
    best_obj1 = np.inf
    for i in range(len(F_sorted)):
        if F_sorted[i, 1] < best_obj1:
            nd_mask[i] = True
            best_obj1  = F_sorted[i, 1]
    return F_sorted[nd_mask]


# ─── archive helpers ──────────────────────────────────────────────────────────

def _update_archive(var_arch, obj_arch, var_new, obj_new):
    """Non-dominated archive update — identical to PDNS update_elitist_archive.

    Minimisation assumed for both objectives.
    """
    var_arch = list(var_arch)
    obj_arch = list(obj_arch)

    dominated_idx = []
    new_dominated  = False

    for idx, obj_existing in enumerate(obj_arch):
        # existing weakly dominates new → new should not be added
        if obj_new[0] >= obj_existing[0] and obj_new[1] >= obj_existing[1]:
            new_dominated = True
            break
        # new weakly dominates existing → existing should be removed
        if obj_existing[0] >= obj_new[0] and obj_existing[1] >= obj_new[1]:
            dominated_idx.append(idx)

    if not new_dominated:
        for idx in sorted(dominated_idx, reverse=True):
            var_arch.pop(idx)
            obj_arch.pop(idx)
        var_arch.append(var_new)
        obj_arch.append(obj_new)

    return var_arch, obj_arch


def _test_archive(var_arch, bench_db: dict, test_key: str, min_flops: float, max_flops: float):
    """Re-evaluate a validation archive on test objectives (test_err × flops_norm).

    Returns a pair ``(test_var_arch, test_obj_array)`` where ``test_obj_array``
    is an ``(M, 2)`` ndarray of non-dominated test objectives.
    """
    if not var_arch:
        return [], np.empty((0, 2))

    test_objs = []
    for vec in var_arch:
        arch_str = _vec_to_arch_str(np.array(vec))
        if arch_str not in bench_db or test_key not in bench_db[arch_str]:
            test_objs.append((1.0, 1.0))
            continue
        entry      = bench_db[arch_str][test_key]
        test_err   = 1.0 - entry['epoch-200'] / 100.0
        flops_norm = (entry['flops'] - min_flops) / (max_flops - min_flops)
        test_objs.append((test_err, flops_norm))

    # Keep only non-dominated points
    nd_var, nd_obj = [], []
    for i, (v, o) in enumerate(zip(var_arch, test_objs)):
        dominated = any(
            other[0] <= o[0] and other[1] <= o[1] and other != o
            for other in test_objs
        )
        if not dominated:
            nd_var.append(v)
            nd_obj.append(o)

    return (
        nd_var,
        np.array(nd_obj) if nd_obj else np.empty((0, 2)),
    )
