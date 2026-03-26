"""Acquisition-function helpers.

Each helper operates on *normalised* surrogate predictions and is designed to
be *maximised* (returns positive values for promising candidates).

Available
---------
expected_improvement(mu, std, y_min) → float array (N, m)
    Per-objective Expected Improvement.

References
----------
[1] D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient global
    optimization of expensive black-box functions," J. Glob. Optim.,
    vol. 13, no. 4, pp. 455–492, 1998.
    https://doi.org/10.1023/A:1008306431147
    (EI formula: eq. (15); per-objective variant used here.)

[2] M. Lukovic, Y. Tian, and W. Ma, "Diversity-guided multi-objective
    Bayesian optimization with batch evaluations," in Proc. NeurIPS, 2020.
    Source code: https://github.com/yunshengtian/DGEMO
    (EI acquisition adapted from ``mobo/acquisition.py::EI.evaluate``.)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mu: np.ndarray,
    std: np.ndarray,
    y_min: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Expected Improvement for minimisation, per objective.

    Parameters
    ----------
    mu    : (N, m) predicted means.
    std   : (N, m) predicted standard deviations.
    y_min : (m,)   current best per objective (in normalised space).
    eps   : floor for std to avoid division by zero.

    Returns
    -------
    ei : (N, m) — non-negative EI values (to be maximised).
    """
    std   = np.maximum(std, eps)
    delta = y_min[None, :] - mu          # (N, m)
    z     = delta / std
    ei    = delta * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0.0)
