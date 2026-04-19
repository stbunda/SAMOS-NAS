"""
Surrogate Models for SAMOS Framework
=====================================

This module provides various surrogate model implementations for
surrogate-assisted optimization.

Available Models:
-----------------
- RFR: Random Forest Regressor (ensemble std via predict_std)
- ETR: Extra Trees Regressor (ensemble std via predict_std)
- KNN: K-Nearest Neighbors
- GP: Gaussian Process (simple alias for GPR)
- GPR: Gaussian Process Regressor — fixed RBF kernel
- GPR_MLE: GPR with MLE kernel optimisation (free length-scale)
- GPR_Matern15: GPR with Matérn ν=1.5 kernel
- GPR_Matern25: GPR with Matérn ν=2.5 kernel
- GPR_White: GPR with RBF + WhiteKernel (noise-aware)
- XGBoost: XGBoost Regressor
- CART: Classification and Regression Trees

GPR variants all expose predict_std(x) returning posterior std.
RFR and ETR expose predict_std(x) via ensemble tree disagreement.
XGBoost and KNN raise NotImplementedError for predict_std().

Usage:
------
    from strategy.surrogate.models import RFR, GPR, GPR_MLE, ETR

    model = RFR(n_estimators=100, seed=42)
    model = GPR_MLE(seed=42)
"""

# Core surrogate models (always available)
from .rf import RFR
from .knn import KNN
from .gp import GP
from .kriging import GPR, GPR_MLE, GPR_Matern15, GPR_Matern25, GPR_White
from .carts import CART
from .extra_trees import ETR

# XGBoost (optional dependency)
try:
    from .xgboost import XGBoost
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    XGBoost = None

# RBF interpolation surrogates (scipy required)
try:
    from .rbf import (
        RBFSurrogate,
        RBF_Cubic,
        RBF_ThinPlateSpline,
        RBF_Gaussian,
        RBF_Multiquadric,
        RBF_InverseQuadratic,
        RBF_InverseMultiquadric,
    )
    _RBF_AVAILABLE = True
except (ImportError, OSError):
    _RBF_AVAILABLE = False
    RBFSurrogate = RBF_Cubic = RBF_ThinPlateSpline = RBF_Gaussian = None
    RBF_Multiquadric = RBF_InverseQuadratic = RBF_InverseMultiquadric = None

# Ensemble surrogate
from .ensemble import EnsembleSurrogate

# Build __all__ dynamically based on what's available
__all__ = [
    # Core models (always available)
    'RFR',
    'ETR',
    'KNN',
    'GP',
    'GPR',
    'GPR_MLE',
    'GPR_Matern15',
    'GPR_Matern25',
    'GPR_White',
    'XGBoost',
    'CART',
    'EnsembleSurrogate',
]

if _RBF_AVAILABLE:
    __all__.extend([
        'RBFSurrogate', 'RBF_Cubic', 'RBF_ThinPlateSpline', 'RBF_Gaussian',
        'RBF_Multiquadric', 'RBF_InverseQuadratic', 'RBF_InverseMultiquadric',
    ])

# Model registry for dynamic instantiation
SURROGATE_MODELS = {
    'RFR': RFR,
    'ETR': ETR,
    'KNN': KNN,
    'GP': GP,
    'GPR': GPR,
    'GPR_MLE': GPR_MLE,
    'GPR_Matern15': GPR_Matern15,
    'GPR_Matern25': GPR_Matern25,
    'GPR_White': GPR_White,
    'XGBoost': XGBoost,
    'CART': CART,
    'EnsembleSurrogate': EnsembleSurrogate,
}

if _RBF_AVAILABLE:
    SURROGATE_MODELS.update({
        'RBF_Cubic':              RBF_Cubic,
        'RBF_ThinPlateSpline':    RBF_ThinPlateSpline,
        'RBF_Gaussian':           RBF_Gaussian,
        'RBF_Multiquadric':       RBF_Multiquadric,
        'RBF_InverseQuadratic':   RBF_InverseQuadratic,
        'RBF_InverseMultiquadric': RBF_InverseMultiquadric,
    })


def get_surrogate_model(model_name: str):
    """
    Get a surrogate model class by name.

    Parameters:
    -----------
    model_name : str
        Name of the surrogate model

    Returns:
    --------
    class
        The surrogate model class

    Raises:
    -------
    ValueError
        If model_name is not recognized

    Example:
    --------
        >>> model_class = get_surrogate_model('XGBoost')
        >>> model = model_class(n_estimators=100, seed=42)
    """
    if model_name not in SURROGATE_MODELS:
        available = ', '.join(SURROGATE_MODELS.keys())
        raise ValueError(
            f"Unknown surrogate model: {model_name}. "
            f"Available models: {available}"
        )
    return SURROGATE_MODELS[model_name]


def list_surrogate_models():
    """
    List all available surrogate models.

    Returns:
    --------
    list
        List of available model names
    """
    return list(SURROGATE_MODELS.keys())



