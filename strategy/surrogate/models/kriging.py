from numpy.random import RandomState
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor


class GPR:
    def __init__(self, length_scale=1.0, random_state: RandomState = None, seed: int = None):
        """
        Gaussian Process Regressor wrapper.

        Parameters
        ----------
        length_scale : float
            Length scale for the RBF kernel (default: 1.0)
        random_state : RandomState, optional
            Numpy random state object
        seed : int, optional
            Random seed (used if random_state is None)

        Notes
        -----
        Uses fixed hyperparameters (no MLE optimization) for:
        - Faster training (important for frequent retraining in SAMOS)
        - Stable behavior with small training sets (20-100 samples)
        - Predictable performance across generations

        Includes normalize_y=True and alpha=1e-10 for numerical stability.
        """
        self.length_scale = length_scale
        if random_state is None and seed is not None:
            self.random_state = RandomState(seed)
        elif random_state is None:
            raise EnvironmentError("random_state is None and no seed provided")
        else:
            self.random_state = random_state

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale, length_scale_bounds="fixed")
        self.name = f'Gaussian Process Regressor (length_scale={length_scale})'
        self.str = f'gpr_{self.length_scale}'
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self.random_state,
            normalize_y=True,  # Normalize outputs for numerical stability
            alpha=1e-10,       # Noise/regularization for numerical stability
        )
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
            "length_scale": self.length_scale,
        }
        return self.metadata