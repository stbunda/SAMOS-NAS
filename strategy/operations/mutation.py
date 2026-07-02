import numpy as np
from pymoo.core.mutation import Mutation


class EncodingAwareMutation(Mutation):
    """Per-gene mutation that respects categorical vs. ordinal gene semantics
    (SAMOS2 C1). Categorical genes (unordered choices, e.g. NAS op selection)
    are uniform-reset to a different value, same as ``IntegerPointMutation``.
    Ordinal genes (e.g. channel-width/depth genes) take a local +/-1 step
    instead, respecting the locality assumption the surrogate itself relies
    on (neighbours in an ordinal gene are neighbours in performance).

    Parameters
    ----------
    xl, xu : np.ndarray
        Per-gene integer bounds.
    categorical_cols : sequence[int]
        Column indices treated as unordered categories; every other column is
        ordinal. Empty sequence -> identical to ``IntegerPointMutation``.
    """

    def __init__(self, xl: "np.ndarray", xu: "np.ndarray", categorical_cols=(), **kwargs):
        super().__init__(**kwargs)
        self.xl = np.asarray(xl, dtype=int)
        self.xu = np.asarray(xu, dtype=int)
        self.categorical_cols = set(int(c) for c in categorical_cols)

    def _do(self, problem, X, **kwargs):
        n_var = X.shape[1]
        X     = X.astype(float)
        Xp    = np.copy(X)
        p_mut = 1.0 / n_var

        for i in range(len(X)):
            for gene in range(n_var):
                if np.random.rand() >= p_mut:
                    continue
                lo, hi  = int(self.xl[gene]), int(self.xu[gene])
                current = int(round(X[i, gene]))
                if gene in self.categorical_cols:
                    choices = [v for v in range(lo, hi + 1) if v != current]
                    if choices:
                        Xp[i, gene] = float(np.random.choice(choices))
                else:
                    steps = [s for s in (-1, 1) if lo <= current + s <= hi]
                    if steps:
                        Xp[i, gene] = float(current + np.random.choice(steps))

        return Xp


class IntegerPointMutation(Mutation):
    """Per-gene point mutation for integer-valued evoxbench search spaces.

    Each gene is independently replaced with probability 1/n_var by a
    uniformly-chosen different integer within [xl[i], xu[i]]. Works
    for any n_var and any integer bounds -- no validity constraints to check.

    Parameters
    ----------
    xl : np.ndarray
        Per-gene lower bounds (integer).
    xu : np.ndarray
        Per-gene upper bounds (integer).
    """

    def __init__(self, xl: "np.ndarray", xu: "np.ndarray", **kwargs):
        super().__init__(**kwargs)
        self.xl = np.asarray(xl, dtype=int)
        self.xu = np.asarray(xu, dtype=int)

    def _do(self, problem, X, **kwargs):
        n_var = X.shape[1]
        X     = X.astype(float)
        Xp    = np.copy(X)
        p_mut = 1.0 / n_var

        for i in range(len(X)):
            for gene in range(n_var):
                if np.random.rand() < p_mut:
                    lo      = int(self.xl[gene])
                    hi      = int(self.xu[gene])
                    current = int(round(X[i, gene]))
                    choices = [v for v in range(lo, hi + 1) if v != current]
                    if choices:
                        Xp[i, gene] = float(np.random.choice(choices))

        return Xp
