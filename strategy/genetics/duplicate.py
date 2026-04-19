import numpy as np
from pymoo.core.duplicate import DuplicateElimination


class IntegerVectorDuplicateElimination(DuplicateElimination):
    """Duplicate elimination for integer-vector evoxbench search spaces.

    Compares individuals by their rounded integer tuple -- pure vector-level
    dedup with no arch-string semantics or structural equivalence checks.
    Suitable for all evoxbench benchmarks (NB-101, NB-201, NATS, DARTS, ...).
    """

    def key(self, x: "np.ndarray") -> tuple:
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
