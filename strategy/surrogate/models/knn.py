import numpy as np
from numpy.random import RandomState

from sklearn.neighbors import KNeighborsRegressor

class KNN:
    def __init__(self, n_neighbors, random_state: RandomState):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.name = f'{n_neighbors}-Nearest Neighbors'
        self.str = f'KNN-{self.n_neighbors}'
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_std(self, x):
        raise NotImplementedError(
            "KNN does not support predict_std(). "
            "Use a GPR or ensemble-based surrogate for uncertainty-aware ablations."
        )

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "n_neighbors": self.n_neighbors,
        }
        return self.metadata