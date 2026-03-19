"""
Surrogate Models for SAMOS Framework
=====================================

This module provides various surrogate model implementations for
surrogate-assisted optimization.

Available Models:
-----------------
- RFR: Random Forest Regressor
- KNN: K-Nearest Neighbors
- GP: Gaussian Process (simple)
- XGBoost: XGBoost Regressor
- GPR: Gaussian Process Regressor (Kriging)
- CART: Classification and Regression Trees
- MLP: Multi-Layer Perceptron
- RNN: Recurrent Neural Network

Usage:
------
    from strategy.surrogate.models import RFR, KNN, GP, XGBoost

    # Create a Random Forest surrogate
    model = RFR(n_estimators=100, seed=42)

    # Create an XGBoost surrogate
    model = XGBoost(n_estimators=100, seed=42)
"""

# Core surrogate models (always available)
from .rf import RFR
from .knn import KNN
from .gp import GP
from .kriging import GPR
from .carts import CART

# XGBoost (optional dependency)
try:
    from .xgboost import XGBoost
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    XGBoost = None

# Neural network-based models (may require additional dependencies)
try:
    from .rnn import RNN, MLP
    _RNN_AVAILABLE = True
except ImportError:
    _RNN_AVAILABLE = False
    RNN = None
    MLP = None

# Enhanced models
try:
    from .gpr_enhanced import GPR_Enhanced
    _GPR_ENHANCED_AVAILABLE = True
except ImportError:
    _GPR_ENHANCED_AVAILABLE = False
    GPR_Enhanced = None

# Build __all__ dynamically based on what's available
__all__ = [
    # Core models (always available)
    'RFR',
    'KNN',
    'GP',
    'XGBoost',
    'GPR',
    'CART',
]

if _RNN_AVAILABLE:
    __all__.extend(['MLP', 'RNN'])

if _GPR_ENHANCED_AVAILABLE:
    __all__.append('GPR_Enhanced')

# Model registry for dynamic instantiation
SURROGATE_MODELS = {
    'RFR': RFR,
    'KNN': KNN,
    'GP': GP,
    'XGBoost': XGBoost,
    'GPR': GPR,
    'CART': CART,
}

if _RNN_AVAILABLE:
    SURROGATE_MODELS['MLP'] = MLP
    SURROGATE_MODELS['RNN'] = RNN

if _GPR_ENHANCED_AVAILABLE:
    SURROGATE_MODELS['GPR_Enhanced'] = GPR_Enhanced


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



