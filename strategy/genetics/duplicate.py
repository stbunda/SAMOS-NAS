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
        # Cache: tuple(vec_int) -> arch_str | None.  Avoids rebuilding ModelSpec
        # for vectors that appear in multiple generations.
        self._cache: dict = {}

    def key(self, x: np.ndarray) -> 'str | None':
        """Return the canonical arch_str for vector x, with caching."""
        k = tuple(np.round(x).astype(int).tolist())
        if k not in self._cache:
            self._cache[k] = _vec_to_arch_str(np.asarray(k, dtype=int), self.bench_db)
        return self._cache[k]

    def _do(self, pop, other, is_duplicate):
        seen = {}

        if other is not None:
            for x in self.func(other):
                arch = self.key(x)
                if arch is not None:
                    seen[arch] = True

        for i, x in enumerate(self.func(pop)):
            arch = self.key(x)
            if arch is None or arch in seen:
                is_duplicate[i] = True
            else:
                seen[arch] = True

        return is_duplicate