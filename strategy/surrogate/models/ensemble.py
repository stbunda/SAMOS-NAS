"""Ensemble surrogate model that aggregates predictions from multiple members.

EnsembleSurrogate wraps a list of pre-configured surrogate objects and exposes
the standard fit / predict / predict_std interface:

  predict      — mean of member predictions (reduces bias)
  predict_std  — standard deviation of member predictions (calibrated uncertainty)

This design means any combination of surrogates (GPR + XGBoost, RFR + ETR, …)
can be composed without modifying the individual surrogate classes.

Usage
-----
    from strategy.surrogate.models.ensemble import EnsembleSurrogate
    from strategy.surrogate.models.kriging import GPR
    from strategy.surrogate.models.xgboost import XGBoost

    members = [GPR(seed=0), XGBoost(100, seed=1)]
    ens = EnsembleSurrogate(members)
    ens.fit(X_train, y_train)
    y_pred = ens.predict(X_test)
    y_std  = ens.predict_std(X_test)  # member disagreement
"""

import numpy as np


class EnsembleSurrogate:
    """Ensemble wrapper: mean prediction, member-disagreement uncertainty.

    Parameters
    ----------
    members : list
        List of surrogate objects, each implementing fit(X, y) and predict(X).
        At least two members are required for meaningful predict_std.
    """

    def __init__(self, members: list):
        if not members:
            raise ValueError('EnsembleSurrogate requires at least one member.')
        self.members = members
        names        = '+'.join(getattr(m, 'str', type(m).__name__) for m in members)
        self.name    = f'Ensemble({names})'
        self.str     = f'ens_{names}'

    def __str__(self):
        return self.name

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit every member on the same (x, y) data."""
        for member in self.members:
            member.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the mean prediction across all members."""
        preds = np.array([m.predict(x).ravel() for m in self.members])
        return preds.mean(axis=0)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Return the standard deviation of member predictions (disagreement proxy)."""
        preds = np.array([m.predict(x).ravel() for m in self.members])
        return preds.std(axis=0)

    def to_config(self):
        return {
            'class':   self.__class__.__name__,
            'module':  self.__class__.__module__,
            'members': [
                getattr(m, 'to_config', lambda: {'class': type(m).__name__})()
                for m in self.members
            ],
        }
