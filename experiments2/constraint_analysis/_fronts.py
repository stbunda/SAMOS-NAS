"""First-front extraction and HV/IGD, minimization convention.

Never materializes an n x n domination matrix: 2-objective fronts use a
sort-sweep, >=2 use an incremental skyline that compares each candidate only
against the running non-dominated set. HV/IGD delegate to pymoo indicators.
"""

from __future__ import annotations

import numpy as np


def first_front(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated (first) front of ``F`` (minimization)."""
    F = np.asarray(F, dtype=float)
    n, m = F.shape if F.ndim == 2 else (0, 0)
    if n == 0:
        return np.empty(0, dtype=int)
    if m == 2:
        # Sweep ascending f0, grouping exact ties in f0 together (tie-broken by
        # f1 ascending). A group is on the front only if its minimum f1 beats
        # every smaller-f0 point seen so far -- and when it does, EVERY point
        # in the group tied at that minimum belongs on the front too (a tie is
        # not dominance: dominance needs a strict improvement in some
        # objective). Keeping only the first-encountered point of a tie (a
        # plain strict '<' sweep) silently drops every genuinely non-dominated
        # duplicate, which is common here since params/flops are discrete and
        # widely shared across architectures.
        order = np.lexsort((F[:, 1], F[:, 0]))
        keep = []
        best = np.inf
        i, n_ord = 0, len(order)
        while i < n_ord:
            f0 = F[order[i], 0]
            j = i
            group_min = np.inf
            while j < n_ord and F[order[j], 0] == f0:
                group_min = min(group_min, F[order[j], 1])
                j += 1
            if group_min < best:
                keep.extend(order[k] for k in range(i, j) if F[order[k], 1] == group_min)
                best = group_min
            i = j
        return np.sort(np.asarray(keep, dtype=int))
    # general m >= 3: pymoo's efficient non-dominated sort (no n x n matrix).
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    idx = NonDominatedSorting(method='efficient_non_dominated_sort').do(
        F, only_non_dominated_front=True)
    return np.sort(np.asarray(idx, dtype=int))


def dominated_by_front(F: np.ndarray, front_F: np.ndarray, block: int = 50_000) -> np.ndarray:
    """Boolean mask: is each row of ``F`` dominated by ANY row of ``front_F``.

    Blocked so a large candidate set is tested against the (small) front without
    an all-vs-all matrix.
    """
    F = np.asarray(F, float)
    front_F = np.asarray(front_F, float)
    n = len(F)
    out = np.zeros(n, dtype=bool)
    if len(front_F) == 0 or n == 0:
        return out
    for s in range(0, n, block):
        chunk = F[s:s + block][:, None, :]           # (b, 1, m)
        fr = front_F[None, :, :]                       # (1, k, m)
        le = np.all(fr <= chunk, axis=2)               # (b, k)
        lt = np.any(fr < chunk, axis=2)
        out[s:s + block] = np.any(le & lt, axis=1)
    return out


def _hv_igd():
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    return HV, IGD


def hypervolume(front_F: np.ndarray, ref_point: np.ndarray) -> float:
    """HV of a front against a fixed reference (nadir) point (minimization)."""
    front_F = np.asarray(front_F, float)
    if len(front_F) == 0:
        return 0.0
    HV, _ = _hv_igd()
    return float(HV(ref_point=np.asarray(ref_point, float))(front_F))


def igd(front_F: np.ndarray, reference_set: np.ndarray) -> float:
    """IGD of a front against a reference Pareto set (minimization)."""
    front_F = np.asarray(front_F, float)
    reference_set = np.asarray(reference_set, float)
    if len(front_F) == 0 or len(reference_set) == 0:
        return float('nan')
    _, IGD = _hv_igd()
    return float(IGD(reference_set)(front_F))
