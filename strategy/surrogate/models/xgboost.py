from numpy.random import RandomState

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")


class XGBoost:
    def __init__(self, n_estimators=100, random_state: RandomState = None, seed: int = None, **kwargs):
        """
        XGBoost Regressor surrogate model.

        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds (trees)
        random_state : RandomState
            Numpy random state
        seed : int
            Random seed for reproducibility
        **kwargs : dict
            Additional XGBoost parameters (e.g., max_depth, learning_rate, etc.)
        """
        self.n_estimators = n_estimators
        if random_state is None and seed is not None:
            self.random_state = RandomState(seed)
            self.seed = seed
        elif random_state is None:
            raise EnvironmentError("random_state is None")
        else:
            self.random_state = random_state
            self.seed = self.random_state.randint(0, 2**31 - 1)

        # Default XGBoost parameters optimized for surrogate modeling
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': self.seed,
            # Force CPU: the pip xgboost wheel is a CUDA build, and on a node
            # whose GPUs aren't masked it would otherwise create a CUDA context
            # and reserve VRAM. Keep training on CPU regardless of node type.
            'device': 'cpu',
            'tree_method': 'hist',
        }
        default_params.update(kwargs)

        self.params = default_params
        self.model = xgb.XGBRegressor(n_estimators=self.n_estimators, **self.params)
        self.name = f'XGBoost Regressor with {self.n_estimators} estimators'
        self.str = f'xgboost_{self.n_estimators}'
        self.metadata = {}

    def __str__(self):
        return self.name

    def fit(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_std(self, x):
        raise NotImplementedError(
            "XGBoost does not support predict_std(). "
            "Use a GPR or ensemble-based surrogate for uncertainty-aware ablations."
        )

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "n_estimators": self.n_estimators,
            "params": self.params,
        }
        return self.metadata

