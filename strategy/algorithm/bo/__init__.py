"""Bayesian Optimisation backbone.

Sub-package layout
------------------
base.py        — Abstract interfaces (SurrogateModel, InnerSolver,
                 BayesianOptimizer) and the shared benchmark run-loop.
surrogate.py   — GaussianProcessSurrogate implementation.
acquisition.py — Acquisition-function helpers (Expected Improvement, …).
solver.py      — InnerSolver implementations (L-BFGS-B, …).

Typical usage
-------------
from strategy.algorithm.bo import (
    BayesianOptimizer,
    SurrogateModel,
    InnerSolver,
    GaussianProcessSurrogate,
    LBFGSBSolver,
    expected_improvement,
)

Acknowledgments
---------------
This implementation was developed with the assistance of GitHub Copilot.
"""

from .base import BayesianOptimizer, SurrogateModel, InnerSolver
from .surrogate import GaussianProcessSurrogate
from .acquisition import expected_improvement
from .solver import LBFGSBSolver

__all__ = [
    "BayesianOptimizer",
    "SurrogateModel",
    "InnerSolver",
    "GaussianProcessSurrogate",
    "expected_improvement",
    "LBFGSBSolver",
]
