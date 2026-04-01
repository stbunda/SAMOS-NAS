import copy

import numpy as np
from numpy.random import RandomState
from pymoo.core.crossover import Crossover

# from search_space.cgp.genotype import Genotype
# from strategy.genetics.genetic_base import GeneticProgram, CartesianGeneticProgram

class NoCrossoverProgram(Crossover):
    def __init__(self, random_state: np.random.RandomState = None, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=1.0)
        self.random_state = random_state

    def _do(self, problem, X, **kwargs):

        # X.shape = (n_parents, n_matings, 1)
        n_parents, n_matings, _ = X.shape

        X_off = np.empty_like(X, dtype=object)

        for p in range(n_parents):
            for m in range(n_matings):
                child = copy.deepcopy(X[p, m, 0])

                # TODO: fix Beun oplossing advancing the random_state
                seed = p * 100000 + m
                child.advance_random_state(seed)
                X_off[p, m, 0] = child

        return X_off

class CrossoverProgram(Crossover):
    def __init__(self, n_offsprings=2, prob=0.9, do_crossover=True, random_state: RandomState = None, eps=1e-4,
                 **kwargs):
        n_parents = 2
        super().__init__(n_parents, n_offsprings, prob, **kwargs)
        self.do_crossover = do_crossover
        self.random_state = random_state
        self.eps = eps

    def to_config(self):
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": {
                'do_crossover': self.do_crossover,
                'random_state': self.random_state,
                'eps': self.eps,
            },
        }

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        assert n_parents == 2
        assert n_var == 1

        X_off = np.empty_like(X, dtype=object)

        for c in range(n_matings):
            p1 = copy.deepcopy(X[0, c, 0])
            p2 = copy.deepcopy(X[1, c, 0])

            if self.do_crossover:
                # IMPORTANT: crossover must return NEW objects
                c1 = p1.crossover(copy.deepcopy(p2))
                c2 = p2.crossover(copy.deepcopy(p1))
            else:
                c1 = p1
                c2 = p2

            X_off[0, c, 0] = c1
            X_off[1, c, 0] = c2

            assert X_off[0, c, 0] is not X[0, c, 0]
            assert X_off[1, c, 0] is not X[1, c, 0]

        return X_off

    def calc_betaq(self, beta, rand):
        """
        Calculate the beta distribution used in SBX.

        Args:
            beta (ndarray): Current beta values for crossover.

        Returns:
            ndarray: Adjusted beta values for offspring generation.
        """
        alpha = 2.0 - np.power(beta, -(self.sbx_eta + 1.0))
        mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))
        betaq = np.zeros(mask.shape)
        betaq[mask] = np.power((rand * alpha), (1.0 / (self.sbx_eta + 1.0)))[mask]
        betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)),
                                   (1.0 / (self.sbx_eta + 1.0)))[mask_not]
        return betaq

    def crossover_base(self, X, problem):
        n_children, n_matings, n_var = X.shape
        parents = copy.deepcopy(X)
        xl, xu = problem.xl, problem.xu

        # Create crossover mask
        do_crossover = np.full(parents[0].shape, True)
        do_crossover[self.random_state.random((n_matings, n_var))] = False
        do_crossover[np.abs(parents[0] - parents[1]) <= self.eps] = False

        # Determine the range
        y1 = np.min(parents, axis=0)
        y2 = np.max(parents, axis=0)

        # Generate random values for the Simulated Binary Crossover
        rand = self.random_state.random((n_matings, n_var))

        # Calculate offspring genetics
        delta = y2 - y1
        delta[delta < 1.0e-10] = 1.0e-10

        beta1 = 1.0 + (2.0 * (y1 - xl) / delta)
        c1 = 0.5 * ((y1 + y2) + self.calc_betaq(beta1, rand) * delta)

        beta2 = 1.0 + (2.0 * (xu - y2) / delta)
        c2 = 0.5 * ((y1 + y2) + self.calc_betaq(beta2, rand) * delta)

        # Perform random swap
        swap_idx = self.random_state.random((n_matings, n_var)) <= 0.5
        c1[swap_idx], c2[swap_idx] = c2[swap_idx], c1[swap_idx]

        # Create the children
        c = np.copy(parents)
        c[0, do_crossover] = c1[do_crossover]
        c[1, do_crossover] = c2[do_crossover]

        for child in range(n_children):
            for mating in range(n_matings):
                pass
                # I need a way to force a mutation in the genome


# ─── NASBench-101 crossover operators ────────────────────────────────────────


class NoCrossover(Crossover):
    """
    No crossover: each parent is copied directly into its offspring slot.
    Mirrors NoCrossoverProgram but operates on integer-vector individuals.
    """

    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=1.0, **kwargs)

    def _do(self, problem, X, **kwargs):
        # X.shape = (n_parents, n_matings, n_var)
        Xp = np.empty_like(X)
        for p in range(X.shape[0]):
            for m in range(X.shape[1]):
                Xp[p, m] = X[p, m].copy()
        return Xp


class TwoPointCrossover101(Crossover):
    """
    2-point crossover over the 26-gene integer vector.
    Retries up to n_var times to obtain a valid pair; falls back to parents.
    prob=0.9 (per-mating crossover probability).
    """

    def __init__(self, prob: float = 0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)

    def to_config(self):
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": {"prob": self.prob},
        }

    def _do(self, problem, X, **kwargs):
        from problem.nasbench101.utils import _is_valid_vec as _nb101_is_valid_vec
        _, n_matings, n_var = X.shape
        Xp = np.empty_like(X)

        for i in range(n_matings):
            for _ in range(n_var):           # up to n_var retries
                # choose 2 cut points
                cuts = np.sort(
                    np.random.choice(n_var - 1, 2, replace=False) + 1
                )
                a, b = int(cuts[0]), int(cuts[1])

                c0 = X[0, i].copy()
                c1 = X[1, i].copy()
                c0[a:b], c1[a:b] = X[1, i, a:b].copy(), X[0, i, a:b].copy()

                if _nb101_is_valid_vec(c0.astype(int)) and _nb101_is_valid_vec(c1.astype(int)):
                    Xp[0, i] = c0
                    Xp[1, i] = c1
                    break
            else:
                # fallback: pass parents through unchanged
                Xp[0, i] = X[0, i].copy()
                Xp[1, i] = X[1, i].copy()

        return Xp


# ─── NASBench-201 crossover operators ────────────────────────────────────────
# Note: NoCrossover101 is fully problem-agnostic and works for NASBench-201 too
# — no separate NoCrossover201 is needed.


class UniformCrossover201(Crossover):
    """
    Uniform crossover over the 6-gene NASBench-201 vector.

    For each gene independently, offspring 0 takes from parent 0 with
    probability 0.5 (and parent 1 otherwise); offspring 1 gets the opposite.
    All resulting architectures are valid (no validity check needed).
    prob=0.9 controls the per-mating crossover probability.
    """

    def __init__(self, prob: float = 0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Xp = np.empty_like(X)

        for i in range(n_matings):
            # mask[j] == True  → offspring0 inherits gene j from parent0
            mask = np.random.rand(n_var) < 0.5
            c0 = X[0, i].copy()
            c1 = X[1, i].copy()
            c0[~mask] = X[1, i, ~mask].copy()
            c1[~mask] = X[0, i, ~mask].copy()
            Xp[0, i] = c0
            Xp[1, i] = c1

        return Xp
