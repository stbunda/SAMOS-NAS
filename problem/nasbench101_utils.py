"""
Shared constants, vector-conversion helpers, and archive utilities for the
integer-vector NASBench-101 search space (26 genes: 5 ops + 21 edges).
"""

import os
import pickle

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

# Precomputed upper-triangular indices for the 7×7 adjacency matrix (21 edges).
_TRIU_I, _TRIU_J = np.triu_indices(7, k=1)

LUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'nasbench101_lut.pkl')


# ─── fast canonical pruner (no ModelSpec / no MD5) ───────────────────────────

def _fast_canonical_bytes(vec_int: np.ndarray) -> 'bytes | None':
    """Prune a raw vector to its canonical form using numpy BFS — no ModelSpec,
    no MD5.  Returns the pruned adjacency matrix + ops concatenated as bytes,
    which uniquely identifies the canonical architecture.  Returns None for
    invalid (disconnected) graphs.

    ~10–20x faster than _vec_to_arch_str on a cache miss.
    """
    ops   = vec_int[:N_OPS]
    edges = vec_int[N_OPS:]

    mat = np.zeros((7, 7), dtype=bool)
    mat[_TRIU_I, _TRIU_J] = edges.astype(bool)

    # Forward reachability from input (node 0)
    fwd = np.zeros(7, dtype=bool); fwd[0] = True
    for _ in range(6):
        fwd |= (fwd[:, None] & mat).any(axis=0)
    if not fwd[6]:
        return None  # output not reachable → invalid

    # Backward reachability to output (node 6)
    bwd = np.zeros(7, dtype=bool); bwd[6] = True
    for _ in range(6):
        bwd |= (mat & bwd[None, :]).any(axis=1)

    valid = fwd & bwd
    pruned_mat = mat[np.ix_(valid, valid)]
    pruned_ops = ops[valid[1:-1]].astype(np.int8)   # int8 matches LUT build dtype
    return pruned_mat.tobytes() + pruned_ops.tobytes()


# ─── lookup-table build / load ────────────────────────────────────────────────

def build_nasbench101_lut(lut_path: str = LUT_PATH) -> dict:
    """Precompute canonical_bytes → arch_str for every valid NASBench-101
    architecture and save to disk.

    Enumerates all raw vectors with ≤ 9 edges (~169 M) in vectorised numpy
    batches.  The expensive hash_module (50 MD5 calls) is invoked only once
    per unique canonical form (~423 k times).  Typical runtime: 3–5 minutes.
    """
    from itertools import combinations
    from itertools import product as iproduct

    # All edge vectors with ≤ 9 edges set (695 860 configs)
    edge_vecs = []
    for n in range(10):
        for combo in combinations(range(N_EDGES), n):
            ev = np.zeros(N_EDGES, dtype=np.int8)
            ev[list(combo)] = 1
            edge_vecs.append(ev)
    edge_vecs = np.array(edge_vecs, dtype=np.int8)          # (695860, 21)

    # All op vectors: 3^5 = 243 configs
    op_vecs = np.array(list(iproduct(range(3), repeat=N_OPS)), dtype=np.int8)

    lut   = {}
    BATCH = 10_000
    n_ev  = len(edge_vecs)

    print(f'Building LUT: {len(op_vecs)} op × {n_ev:,} edge = {len(op_vecs)*n_ev:,} vectors')

    for oi, op_vec in enumerate(op_vecs):
        if oi % 50 == 0:
            print(f'  op {oi:>3}/{len(op_vecs)}  LUT size: {len(lut):,}')

        for start in range(0, n_ev, BATCH):
            batch_edges = edge_vecs[start:start + BATCH]    # (B, 21)
            B = len(batch_edges)

            # Build B adjacency matrices in one shot
            mats = np.zeros((B, 7, 7), dtype=bool)
            mats[:, _TRIU_I, _TRIU_J] = batch_edges.astype(bool)

            # Forward BFS from node 0
            fwd = np.zeros((B, 7), dtype=bool); fwd[:, 0] = True
            for _ in range(6):
                fwd |= (fwd[:, :, None] & mats).any(axis=1)

            vi = np.where(fwd[:, 6])[0]   # output reachable
            if not vi.size:
                continue

            sub_mats = mats[vi]
            sub_fwd  = fwd[vi]

            # Backward BFS to node 6
            bwd = np.zeros((len(vi), 7), dtype=bool); bwd[:, 6] = True
            for _ in range(6):
                bwd |= (sub_mats & bwd[:, None, :]).any(axis=2)

            valid_nodes = sub_fwd & bwd   # (n_valid, 7)

            for k, idx in enumerate(vi):
                vn = valid_nodes[k]
                if not (vn[0] and vn[6]):
                    continue

                pruned_mat = sub_mats[k][np.ix_(vn, vn)]
                pruned_ops = op_vec[vn[1:-1]]
                cb = pruned_mat.tobytes() + pruned_ops.tobytes()

                if cb not in lut:
                    full_vec = np.concatenate([op_vec, batch_edges[idx]]).astype(int)
                    lut[cb] = _vec_to_arch_str(full_vec, {})

    print(f'LUT complete: {len(lut):,} unique architectures')
    os.makedirs(os.path.dirname(lut_path), exist_ok=True)
    with open(lut_path, 'wb') as f:
        pickle.dump(lut, f)
    print(f'Saved → {lut_path}')
    return lut


def load_nasbench101_lut(lut_path: str = LUT_PATH) -> 'dict | None':
    """Load the precomputed LUT from disk.  Returns None if not found or incomplete."""
    if not os.path.exists(lut_path):
        return None
    with open(lut_path, 'rb') as f:
        lut = pickle.load(f)
    if len(lut) < 400_000:
        print(f'[WARNING] LUT at {lut_path} has only {len(lut):,} entries '
              f'(expected ~423k). Falling back to ModelSpec.')
        return None
    return lut


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
