import copy

import numpy as np
from numpy.random import RandomState
from typing import List
from pymoo.core.sampling import Sampling

import importlib

def load_class(class_path):
    module = importlib.import_module(class_path["module"])
    return getattr(module, class_path["class"])


class SamplingGP(Sampling):

    def __init__(self,
                 blueprint: dict):
        """
        This abstract class represents any sampling strategy that can be used to create an initial population or
        an initial search point.
        """
        super().__init__()
        blueprint_program = copy.deepcopy(blueprint)
        self.blueprint_program = blueprint_program.pop('type')
        self.blueprint_config = blueprint_program

    def _do(self, problem, n_samples, **kwargs):
        # Create an array to store sampled individuals
        X = np.empty((n_samples, 1), dtype=object)

        # Populate each individual in the array
        for n in range(n_samples):
            X[n, 0] = self.blueprint_program(**self.blueprint_config)
        return X


# ─── NASBench-101 sampling ────────────────────────────────────────────────────


class ValidRandomSampling101(Sampling):
    """Generates a matrix of valid 26-gene NASBench-101 integer vectors."""

    def _do(self, problem, n_samples, **kwargs):
        from problem.nasbench101.utils import (
            N_VAR as _NB101_N_VAR,
            N_OPS as _NB101_N_OPS,
            N_EDGES as _NB101_N_EDGES,
            _is_valid_vec as _nb101_is_valid_vec,
        )
        X = np.zeros((n_samples, _NB101_N_VAR), dtype=float)
        for i in range(n_samples):
            for _ in range(200):
                ops   = np.random.randint(0, 3, size=_NB101_N_OPS)
                n_e   = np.random.randint(1, 10)
                edges = np.zeros(_NB101_N_EDGES, dtype=int)
                edges[np.random.choice(_NB101_N_EDGES, n_e, replace=False)] = 1
                vec = np.concatenate([ops, edges])
                if _nb101_is_valid_vec(vec):
                    X[i] = vec.astype(float)
                    break
        return X


# ─── NASBench-201 sampling ────────────────────────────────────────────────────


class ValidRandomSampling201(Sampling):
    """Generates a matrix of valid 6-gene NASBench-201 integer vectors.

    All 5^6 = 15,625 architectures are valid — no retry is needed.
    """

    def _do(self, problem, n_samples, **kwargs):
        from problem.nasbench201.utils import (
            N_VAR as _NB201_N_VAR,
            N_OPS_PER_GENE as _NB201_N_OPS,
        )
        return np.random.randint(
            0, _NB201_N_OPS, size=(n_samples, _NB201_N_VAR)
        ).astype(float)


# ─── EvoXBench sampling ───────────────────────────────────────────────────────


class EvoxBenchSampler(Sampling):
    """Uniform random integer sampling over any evoxbench search space.

    Bounds are passed explicitly so that this sampler works for every
    evoxbench benchmark (NB-101, NB-201, NATS, DARTS, MNv3, …) without any
    benchmark-specific logic.  All evoxbench spaces are unconstrained within
    their integer bounds, so no validity retry is needed.

    Parameters
    ----------
    xl : np.ndarray
        Per-gene lower bounds (integer).
    xu : np.ndarray
        Per-gene upper bounds (integer).
    """

    def __init__(self, xl: np.ndarray, xu: np.ndarray):
        super().__init__()
        self.xl = np.asarray(xl, dtype=int)
        self.xu = np.asarray(xu, dtype=int)

    def _do(self, problem, n_samples, **kwargs):
        return np.column_stack([
            np.random.randint(lo, hi + 1, size=n_samples)
            for lo, hi in zip(self.xl, self.xu)
        ]).astype(float)
