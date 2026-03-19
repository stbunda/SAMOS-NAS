from pprint import pprint

import numpy as np
from pymoo.core.duplicate import DuplicateElimination
from problem.nasbench101_utils import _vec_to_arch_str


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
        super().__init__(**kwargs)
        self.bench_db = bench_db
        # Cache: bytes(vec_int8) -> arch_str | None.  Avoids rebuilding ModelSpec
        # for vectors that appear in multiple generations.  tobytes() on an int8
        # array is ~5x faster to hash than tuple(int_list).
        self._cache: dict = {}

    def key(self, x: np.ndarray) -> 'str | None':
        """Return the canonical arch_str for vector x, with caching.

        Accepts float or integer arrays; rounding is applied internally.
        """
        k = np.round(x).astype(np.int8).tobytes()
        if k not in self._cache:
            self._cache[k] = _vec_to_arch_str(
                np.frombuffer(k, dtype=np.int8).astype(int), self.bench_db
            )
        return self._cache[k]

    def _do(self, pop, other, is_duplicate):
        # Batch-round once per population to avoid repeated numpy overhead
        # inside key() for each individual.
        seen = {}

        if other is not None:
            for row in np.round(self.func(other)).astype(np.int8):
                k = row.tobytes()
                if k not in self._cache:
                    self._cache[k] = _vec_to_arch_str(row.astype(int), self.bench_db)
                arch = self._cache[k]
                if arch is not None:
                    seen[arch] = True

        for i, row in enumerate(np.round(self.func(pop)).astype(np.int8)):
            k = row.tobytes()
            if k not in self._cache:
                self._cache[k] = _vec_to_arch_str(row.astype(int), self.bench_db)
            arch = self._cache[k]
            if arch is None or arch in seen:
                is_duplicate[i] = True
            else:
                seen[arch] = True

        return is_duplicate