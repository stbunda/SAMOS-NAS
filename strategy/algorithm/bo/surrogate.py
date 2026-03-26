"""Gaussian Process surrogate — one GP per objective.

Kernel: ConstantKernel × Matern-5/2 with ARD length scales, fitted via
sklearn's ``GaussianProcessRegressor`` (normalize_y=True, alpha=1e-5).
Inputs are assumed to be *normalised* to [0, 1]^d.

References
----------
[1] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for
    Machine Learning*, MIT Press, 2006.
    http://www.gaussianprocess.org/gpml/
    (Matern covariance functions: Chapter 4, Section 4.2.)

[2] M. Lukovic, Y. Tian, and W. Ma, "Diversity-guided multi-objective
    Bayesian optimization with batch evaluations," in Proc. NeurIPS, 2020.
    Source code: https://github.com/yunshengtian/DGEMO
    (Kernel configuration and GP hyperparameter choices adapted from
    ``mobo/surrogate_model/gaussian_process.py``.)
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .base import SurrogateModel


def _make_gp(n_var: int, n_train: int) -> GaussianProcessRegressor:
    """Build a single-output GP.

    The number of restarts is reduced for large training sets to keep
    wall-clock time manageable.
    """
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e3))
        * Matern(
            length_scale=np.ones(n_var),
            length_scale_bounds=(1e-10, 1e5),
            nu=2.5,
        )
    )
    n_restarts = 2 if n_train > 100 else min(5 * n_var, 25)
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-5,
        n_restarts_optimizer=n_restarts,
    )


class GaussianProcessSurrogate(SurrogateModel):
    """One sklearn GP per objective, fitted on normalised data.

    Parameters
    ----------
    n_var : int
        Number of decision variables (determines kernel length-scale shape).

    Notes
    -----
    ``fit`` rebuilds the GPs every time so that ``n_train`` is accurate for
    choosing the number of hyperparameter restarts.
    """

    def __init__(self, n_var: int) -> None:
        self.n_var = n_var
        self._gps: list[GaussianProcessRegressor] = []

    def fit(self, X_norm: np.ndarray, Y_norm: np.ndarray) -> None:
        n_train, n_obj = Y_norm.shape
        self._gps = []
        for i in range(n_obj):
            gp = _make_gp(self.n_var, n_train)
            gp.fit(X_norm, Y_norm[:, i])
            self._gps.append(gp)

    def predict(
        self, X_norm: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return predicted means (N, m) and optionally stds (N, m)."""
        means, stds = [], []
        for gp in self._gps:
            if return_std:
                mu, sigma = gp.predict(X_norm, return_std=True)
                stds.append(sigma)
            else:
                mu = gp.predict(X_norm)
            means.append(mu)

        F = np.stack(means, axis=1)
        if return_std:
            S = np.stack(stds, axis=1)
            return F, S
        return F
