"""
Gaussian Process wrapper for SAMOS.
Alias for GPR from kriging.py for consistency with naming convention.
"""
from strategy.surrogate.models.kriging import GPR as GP

__all__ = ['GP']

