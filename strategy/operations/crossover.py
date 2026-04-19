import numpy as np
from pymoo.core.crossover import Crossover


class IntegerUniformCrossover(Crossover):
    """Uniform crossover for integer-valued evoxbench search spaces.

    For each gene independently, offspring 0 inherits from parent 0 with
    probability 0.5 (offspring 1 gets the opposite). Works for any n_var
    and any integer bounds -- no validity constraints to check.
    prob (default 0.9) is the per-mating crossover probability.
    """

    def __init__(self, prob: float = 0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Xp = np.empty_like(X)

        for i in range(n_matings):
            mask = np.random.rand(n_var) < 0.5
            c0 = X[0, i].copy()
            c1 = X[1, i].copy()
            c0[~mask] = X[1, i, ~mask].copy()
            c1[~mask] = X[0, i, ~mask].copy()
            Xp[0, i] = c0
            Xp[1, i] = c1

        return Xp
