import numpy as np
from numpy.random import RandomState
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


def _resolve_rng(random_state, seed):
    if random_state is not None:
        return random_state
    if seed is not None:
        return RandomState(seed)
    raise EnvironmentError("random_state is None and no seed provided")


class GPR:
    """GPR with fixed RBF kernel — fast, stable, no MLE optimisation."""

    def __init__(self, length_scale=1.0, random_state: RandomState = None, seed: int = None):
        self.length_scale = length_scale
        self.random_state = _resolve_rng(random_state, seed)

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            length_scale, length_scale_bounds="fixed"
        )
        self.name = f'Gaussian Process Regressor (length_scale={length_scale})'
        self.str = f'gpr_{self.length_scale}'
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self.random_state,
            normalize_y=True,
            alpha=1e-10,
        )
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_std(self, x):
        _, std = self.model.predict(x, return_std=True)
        return std

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "length_scale": self.length_scale,
        }
        return self.metadata


class GPR_MLE:
    """GPR with free RBF kernel — sklearn tunes length-scale via marginal likelihood."""

    def __init__(self, length_scale=1.0, random_state: RandomState = None, seed: int = None):
        self.length_scale = length_scale
        self.random_state = _resolve_rng(random_state, seed)

        kernel = ConstantKernel(1.0) * RBF(length_scale)
        self.name = f'GPR MLE (length_scale_init={length_scale})'
        self.str = f'gpr_mle_{length_scale}'
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self.random_state,
            normalize_y=True,
            alpha=1e-10,
            n_restarts_optimizer=3,
        )
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_std(self, x):
        _, std = self.model.predict(x, return_std=True)
        return std

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "length_scale": self.length_scale,
        }
        return self.metadata


class GPR_Matern15:
    """GPR with Matérn ν=1.5 kernel (once-differentiable, rougher than RBF)."""

    def __init__(self, length_scale=1.0, random_state: RandomState = None, seed: int = None):
        self.length_scale = length_scale
        self.random_state = _resolve_rng(random_state, seed)

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=length_scale, length_scale_bounds="fixed", nu=1.5
        )
        self.name = f'GPR Matérn ν=1.5 (length_scale={length_scale})'
        self.str = f'gpr_matern15_{length_scale}'
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self.random_state,
            normalize_y=True,
            alpha=1e-10,
        )
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_std(self, x):
        _, std = self.model.predict(x, return_std=True)
        return std

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "length_scale": self.length_scale,
            "nu": 1.5,
        }
        return self.metadata


class GPR_Matern25:
    """GPR with Matérn ν=2.5 kernel (twice-differentiable, between RBF and ν=1.5)."""

    def __init__(self, length_scale=1.0, random_state: RandomState = None, seed: int = None):
        self.length_scale = length_scale
        self.random_state = _resolve_rng(random_state, seed)

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=length_scale, length_scale_bounds="fixed", nu=2.5
        )
        self.name = f'GPR Matérn ν=2.5 (length_scale={length_scale})'
        self.str = f'gpr_matern25_{length_scale}'
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self.random_state,
            normalize_y=True,
            alpha=1e-10,
        )
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_std(self, x):
        _, std = self.model.predict(x, return_std=True)
        return std

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "length_scale": self.length_scale,
            "nu": 2.5,
        }
        return self.metadata


class GPR_White:
    """GPR with RBF + WhiteKernel — explicitly models observation noise."""

    def __init__(self, length_scale=1.0, noise_level=0.1,
                 random_state: RandomState = None, seed: int = None):
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.random_state = _resolve_rng(random_state, seed)

        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(length_scale, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=noise_level, noise_level_bounds="fixed")
        )
        self.name = f'GPR White (length_scale={length_scale}, noise={noise_level})'
        self.str = f'gpr_white_{length_scale}'
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self.random_state,
            normalize_y=True,
            alpha=1e-10,
        )
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_std(self, x):
        _, std = self.model.predict(x, return_std=True)
        return std

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "length_scale": self.length_scale,
            "noise_level": self.noise_level,
        }
        return self.metadata