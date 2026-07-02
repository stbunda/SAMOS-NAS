"""strategy/surrogate/canonical.py --- Canonical-phenotype dedup (SAMOS2 C2).

Some evoxbench search spaces map many distinct integer genotypes onto the
same architecture (NASBench-101's op+edge encoding, DARTS's genotype), which
is invisible to raw-vector dedup (``IntegerVectorDuplicateElimination`` /
SAMOSMinimal's default ``dedup_key_fn``). ``Canonicalizer`` decodes a
genotype to its phenotype (via ``benchmark.search_space.decode``) and uses
that as the dedup / archive key instead.

This is *not* full graph-isomorphism canonicalisation (e.g. it will not
collapse two NASBench-101 graphs that are isomorphic under node relabelling
but decode to different ``matrix``/``ops`` dicts) -- it only collapses
genotypes that decode to an identical phenotype object. That already
removes a meaningful share of the redundancy (e.g. dangling/unreachable
edges in NB101 that decode to the same effective graph).
"""

from __future__ import annotations

import numpy as np
from pymoo.core.duplicate import DuplicateElimination


def _canonical_str(decoded) -> str:
    """Deterministic string key for one decoded phenotype."""
    if isinstance(decoded, dict):
        items = []
        for k in sorted(decoded.keys()):
            v = decoded[k]
            if isinstance(v, np.ndarray):
                v = v.tolist()
            items.append((k, v))
        return str(items)
    return str(decoded)


class Canonicalizer:
    """key(x) -> hashable canonical-phenotype key for one genotype vector."""

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def key(self, x: np.ndarray):
        x_int = np.round(np.asarray(x)).astype(int)[None, :]
        decoded = self.benchmark.search_space.decode(x_int)[0]
        return _canonical_str(decoded)

    def keys(self, X: np.ndarray) -> list:
        """Batched form of ``key`` -- use for whole-archive operations
        (e.g. training-set collapse) to avoid one decode() call per row."""
        X_int = np.round(np.asarray(X)).astype(int)
        decoded = self.benchmark.search_space.decode(X_int)
        return [_canonical_str(d) for d in decoded]


class PhenotypeDuplicateElimination(DuplicateElimination):
    """pymoo DuplicateElimination using Canonicalizer phenotype keys instead
    of raw-vector equality (drop-in swap for
    ``IntegerVectorDuplicateElimination`` in the inner GA)."""

    def __init__(self, canonicalizer: Canonicalizer):
        super().__init__()
        self.canonicalizer = canonicalizer

    def _do(self, pop, other, is_duplicate):
        seen = {}
        if other is not None:
            for k in self.canonicalizer.keys(self.func(other)):
                seen[k] = True
        keys = self.canonicalizer.keys(self.func(pop))
        for i, k in enumerate(keys):
            if k in seen:
                is_duplicate[i] = True
            else:
                seen[k] = True
        return is_duplicate
