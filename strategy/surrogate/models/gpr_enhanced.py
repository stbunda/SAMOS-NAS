"""
Improved Gaussian Process implementation with best practices.
Optional enhanced version with normalize_y and alpha parameter.
"""
from numpy.random import RandomState
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor


class GPR_Enhanced:
    """
    Enhanced Gaussian Process Regressor with best practices.

    Improvements over basic GPR:
    - normalize_y=True for numerical stability
    - alpha parameter for noise/regularization
    - Same interface as RFR/KNN for drop-in replacement
    """

    def __init__(
        self,
        length_scale=1.0,
        random_state: RandomState = None,
        seed: int = None,
        normalize_y: bool = True,
        alpha: float = 1e-10,
    ):
        """
        Enhanced Gaussian Process Regressor wrapper.

        Parameters
        ----------
        length_scale : float
            Length scale for the RBF kernel (default: 1.0)
        random_state : RandomState, optional
            Numpy random state object
        seed : int, optional
            Random seed (used if random_state is None)
        normalize_y : bool
            Whether to normalize target values (recommended: True)
        alpha : float
            Value added to diagonal of kernel matrix for numerical stability
            Also represents noise level (default: 1e-10)
        """
        self.length_scale = length_scale
        self.normalize_y = normalize_y
        self.alpha = alpha

        if random_state is None and seed is not None:
            self.random_state = RandomState(seed)
        elif random_state is None:
            raise EnvironmentError("random_state is None and no seed provided")
        else:
            self.random_state = random_state

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            length_scale, length_scale_bounds="fixed"
        )

        self.name = f'Gaussian Process Regressor (length_scale={length_scale}, normalized={normalize_y})'
        self.str = f'gpr_{self.length_scale}'

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self.random_state,
            normalize_y=normalize_y,
            alpha=alpha,
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
            "normalize_y": self.normalize_y,
            "alpha": self.alpha,
        }
        return self.metadata

