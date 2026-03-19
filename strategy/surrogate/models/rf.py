import numpy as np
from numpy.random import RandomState

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


class RFR:
    def __init__(self, n_estimators, random_state: RandomState = None, seed: int = None):
        self.n_estimators = n_estimators
        if random_state is None and seed is not None:
            self.random_state = RandomState(seed)
        elif random_state is None:
            EnvironmentError("random_state is None")
        else:
            self.random_state = random_state
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        self.name = f'Random Forest Regressor with {self.n_estimators} estimators'
        self.str = f'rfr_{self.n_estimators}'
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "n_estimators": self.n_estimators,
        }
        return self.metadata