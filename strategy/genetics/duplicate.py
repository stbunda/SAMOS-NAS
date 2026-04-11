from pprint import pprint

import numpy as np
from pymoo.core.duplicate import DuplicateElimination


class NoDuplicateElimination(DuplicateElimination):

    def do(self, pop, *args, **kwargs):
        return pop

    def to_config(self):
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": {},
        }


class GPDuplicateElimination(DuplicateElimination):

    def __init__(self, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _do(self, pop, other, is_duplicate):
        X_pop = self.func(pop)

        if other is not None:
            X_other = self.func(other)

            seen = {
                prog[0].tuple_program: i
                for i, prog in enumerate(X_other)
            }
        else:
            seen = {}

        for i, (prog,) in enumerate(X_pop):
            prog_key = prog.tuple_program

            if prog_key in seen:
                is_duplicate[i] = True
            else:
                seen[prog_key] = i

        return is_duplicate

    def to_config(self):
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": {
                'epsilon': self.epsilon,
            },
        }

class NASBench101DuplicateElimination(DuplicateElimination):
    """
    Duplicate elimination for the integer-vector NASBench-101 baseline.

    Compares individuals by their canonical ``arch_str`` (MD5 of the pruned
    ModelSpec) rather than the raw 26-gene vector.  Two vectors that differ
    only in edges connected to unreachable nodes decode to the same
    ``arch_str`` and are correctly treated as duplicates, preventing redundant
    evaluations.
    """

    def __init__(self, bench_db: dict, **kwargs):
        from problem.nasbench101.utils import (
            _vec_to_arch_str, _fast_canonical_bytes, load_nasbench101_lut,
        )
        super().__init__(**kwargs)
        self.bench_db = bench_db
        # Per-run cache: raw_vec_bytes -> arch_str | None
        self._cache: dict = {}
        # Precomputed LUT: canonical_bytes -> arch_str (loaded from disk).
        # When present, replaces ModelSpec + hash_module on every cache miss.
        self._lut: 'dict | None' = load_nasbench101_lut()
        self._vec_to_arch_str = _vec_to_arch_str
        self._fast_canonical_bytes = _fast_canonical_bytes

    def _resolve(self, vec_int8: np.ndarray) -> 'str | None':
        """Map a rounded int8 vector to its arch_str.

        Uses the precomputed LUT (fast numpy pruner + dict lookup) when
        available, falling back to ModelSpec + hash_module otherwise.
        """
        if self._lut is not None:
            cb = _fast_canonical_bytes(vec_int8.astype(int))
            return self._lut.get(cb) if cb is not None else None
        return _vec_to_arch_str(vec_int8.astype(int), self.bench_db)

    def key(self, x: np.ndarray) -> 'str | None':
        """Return the canonical arch_str for vector x, with caching.

        Accepts float or integer arrays; rounding is applied internally.
        """
        k = np.round(x).astype(np.int8).tobytes()
        if k not in self._cache:
            self._cache[k] = self._resolve(np.frombuffer(k, dtype=np.int8))
        return self._cache[k]

    def _do(self, pop, other, is_duplicate):
        # Batch-round once per population to avoid repeated numpy overhead.
        seen = {}

        if other is not None:
            for row in np.round(self.func(other)).astype(np.int8):
                k = row.tobytes()
                if k not in self._cache:
                    self._cache[k] = self._resolve(row)
                arch = self._cache[k]
                if arch is not None:
                    seen[arch] = True

        for i, row in enumerate(np.round(self.func(pop)).astype(np.int8)):
            k = row.tobytes()
            if k not in self._cache:
                self._cache[k] = self._resolve(row)
            arch = self._cache[k]
            if arch is None or arch in seen:
                is_duplicate[i] = True
            else:
                seen[arch] = True

        return is_duplicate


# ─── NASBench-201 duplicate elimination ──────────────────────────────────────

def _vec_to_arch_str_201(x):
    from problem.nasbench201.utils import _vec_to_arch_str as _fn
    return _fn(x)


class NASBench201DuplicateElimination(DuplicateElimination):
    """
    Duplicate elimination for the categorical NASBench-201 baseline.

    Compares individuals by their canonical arch_str which is trivially
    computed from the 6-gene vector — no LUT or ModelSpec required.
    """

    def key(self, x: np.ndarray) -> str:
        """Return the arch_str for vector x (float or int; rounding applied)."""
        return _vec_to_arch_str_201(np.round(x).astype(int))

    def _do(self, pop, other, is_duplicate):
        seen = {}

        if other is not None:
            for row in np.round(self.func(other)).astype(int):
                seen[_vec_to_arch_str_201(row)] = True

        for i, row in enumerate(np.round(self.func(pop)).astype(int)):
            arch = _vec_to_arch_str_201(row)
            if arch in seen:
                is_duplicate[i] = True
            else:
                seen[arch] = True


# ─── EvoXBench duplicate elimination ─────────────────────────────────────────


class IntegerVectorDuplicateElimination(DuplicateElimination):
    """Duplicate elimination for integer-vector evoxbench search spaces.

    Compares individuals by their rounded integer tuple — pure vector-level
    dedup with no arch-string semantics or structural equivalence checks.
    Suitable for all evoxbench benchmarks (NB-101, NB-201, NATS, DARTS, …).
    """

    def key(self, x: np.ndarray) -> tuple:
        return tuple(np.round(x).astype(int).tolist())

    def _do(self, pop, other, is_duplicate):
        seen = {}

        if other is not None:
            for row in self.func(other):
                k = tuple(np.round(row).astype(int).tolist())
                seen[k] = True

        for i, row in enumerate(self.func(pop)):
            k = tuple(np.round(row).astype(int).tolist())
            if k in seen:
                is_duplicate[i] = True
            else:
                seen[k] = True

        return is_duplicate


# ─── EvoXBench NASBench-101 arch-str duplicate elimination ───────────────────


class EvoxNASBench101DuplicateElimination(DuplicateElimination):
    """Arch-str duplicate elimination for the evoxbench NASBench-101 search space.

    EvoXBench encodes NASBench-101 architectures as a 26-dimensional integer
    vector with **edges first, then ops**:
        x[:21]  — 21 binary edge values (upper-triangular adjacency)
        x[21:]  — 5 op indices (0=conv3x3, 1=conv1x1, 2=maxpool)

    This is the reverse of our internal ``NASBench101DuplicateElimination``
    which uses ``[ops(5), edges(21)]``.  Before hashing, vectors are reordered
    to ops-first so that the existing ``_fast_canonical_bytes`` / LUT logic can
    be reused without modification.

    Two vectors that differ only in edges connected to pruned-away nodes will
    hash to the same arch_str and are treated as duplicates, preventing
    redundant real evaluations.

    Parameters
    ----------
    bench_db : dict or None
        NASBench-101 lookup table.  Only used as a fallback when the
        precomputed LUT is unavailable.  Pass ``None`` to skip (LUT path
        is tried first).
    """

    def __init__(self, bench_db: 'dict | None' = None, **kwargs):
        from problem.nasbench101.utils import (
            _vec_to_arch_str, _fast_canonical_bytes, load_nasbench101_lut,
        )
        super().__init__(**kwargs)
        self.bench_db = bench_db or {}
        self._cache: dict = {}
        self._lut: 'dict | None' = load_nasbench101_lut()
        self._vec_to_arch_str = _vec_to_arch_str
        self._fast_canonical_bytes = _fast_canonical_bytes

    @staticmethod
    def _reorder(vec_int: np.ndarray) -> np.ndarray:
        """Convert evoxbench edges-first vector to the ops-first internal format."""
        edges = vec_int[:21]
        ops   = vec_int[21:]
        return np.concatenate([ops, edges])

    def _resolve(self, vec_int: np.ndarray) -> 'str | None':
        """Map a rounded edges-first integer vector to its canonical arch_str."""
        internal = self._reorder(vec_int.astype(int))
        if self._lut is not None:
            cb = self._fast_canonical_bytes(internal)
            return self._lut.get(cb) if cb is not None else None
        return self._vec_to_arch_str(internal, self.bench_db)

    def key(self, x: np.ndarray) -> 'str | None':
        """Return the canonical arch_str for an evoxbench-encoded vector x."""
        k = np.round(x).astype(np.int8).tobytes()
        if k not in self._cache:
            self._cache[k] = self._resolve(np.frombuffer(k, dtype=np.int8))
        return self._cache[k]

    def _do(self, pop, other, is_duplicate):
        seen = {}

        if other is not None:
            for row in np.round(self.func(other)).astype(np.int8):
                k = row.tobytes()
                if k not in self._cache:
                    self._cache[k] = self._resolve(row)
                arch = self._cache[k]
                if arch is not None:
                    seen[arch] = True

        for i, row in enumerate(np.round(self.func(pop)).astype(np.int8)):
            k = row.tobytes()
            if k not in self._cache:
                self._cache[k] = self._resolve(row)
            arch = self._cache[k]
            if arch is None or arch in seen:
                is_duplicate[i] = True
            else:
                seen[arch] = True

        return is_duplicate