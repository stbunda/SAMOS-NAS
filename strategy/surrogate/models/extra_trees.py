import numpy as np
from numpy.random import RandomState
from sklearn.ensemble import ExtraTreesRegressor


class ETR:
    """Extra Trees Regressor surrogate model.

    Typically trains faster than Random Forest and provides a similar
    ensemble-based uncertainty proxy via predict_std().
    """

    def __init__(self, n_estimators=100, random_state: RandomState = None, seed: int = None):
        self.n_estimators = n_estimators
        if random_state is None and seed is not None:
            self.random_state = RandomState(seed)
        elif random_state is None:
            raise EnvironmentError("random_state is None and no seed provided")
        else:
            self.random_state = random_state

        self.model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.name = f'Extra Trees Regressor with {self.n_estimators} estimators'
        self.str = f'etr_{self.n_estimators}'
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_std(self, x):
        """Ensemble std across individual trees as an uncertainty proxy."""
        tree_preds = np.array([t.predict(x) for t in self.model.estimators_])
        return tree_preds.std(axis=0)

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "n_estimators": self.n_estimators,
        }
        return self.metadata
