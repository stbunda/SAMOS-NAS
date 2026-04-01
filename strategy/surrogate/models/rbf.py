"""RBF surrogate models wrapping scipy.interpolate.RBFInterpolator.

Six concrete kernel variants are provided:
  RBF_Cubic              ('cubic',             no epsilon)
  RBF_ThinPlateSpline    ('thin_plate_spline', no epsilon)
  RBF_Gaussian           ('gaussian',          epsilon=1.0)
  RBF_Multiquadric       ('multiquadric',      epsilon=1.0)
  RBF_InverseQuadratic   ('inverse_quadratic', epsilon=1.0)
  RBF_InverseMultiquadric('inverse_multiquadric', epsilon=1.0)

All models expose the standard fit / predict interface.
predict_std raises NotImplementedError (RBF gives no uncertainty estimate).
"""

import numpy as np
from scipy.interpolate import RBFInterpolator


class RBFSurrogate:
    """Base RBF surrogate wrapping RBFInterpolator.

    Parameters
    ----------
    kernel : str
        RBF kernel name accepted by RBFInterpolator.
    epsilon : float or None
        Shape parameter (ignored for kernels that don't use it).
    smoothing : float
        Regularisation term added to the diagonal of the kernel matrix.
        0.0 = exact interpolation; 1e-3 recommended for stability.
    """

    def __init__(self, kernel: str, epsilon: float = 1.0, smoothing: float = 1e-3):
        self.kernel    = kernel
        self.epsilon   = epsilon
        self.smoothing = smoothing
        self._rbf      = None
        self.name      = f'RBF({kernel}, eps={epsilon})'
        self.str       = f'rbf_{kernel}'

    def __str__(self):
        return self.name

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit the RBF interpolator.

        Parameters
        ----------
        x : (n, d) array of input points.
        y : (n,) array of target values.
        """
        y = np.asarray(y).ravel()
        kwargs = dict(kernel=self.kernel, smoothing=self.smoothing, degree=0)
        # epsilon is meaningless for cubic / thin_plate_spline
        if self.kernel not in ('linear', 'thin_plate_spline', 'cubic', 'quintic'):
            kwargs['epsilon'] = self.epsilon
        self._rbf = RBFInterpolator(x, y, **kwargs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._rbf is None:
            raise RuntimeError('Call fit() before predict().')
        return self._rbf(x).ravel()

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support uncertainty estimation.'
        )

    def to_config(self):
        return {
            'class':    self.__class__.__name__,
            'module':   self.__class__.__module__,
            'kernel':   self.kernel,
            'epsilon':  self.epsilon,
            'smoothing': self.smoothing,
        }


# ── Concrete kernel variants ──────────────────────────────────────────────────

class RBF_Cubic(RBFSurrogate):
    """φ(r) = r³  — smooth, no shape parameter."""
    def __init__(self, smoothing: float = 1e-3):
        super().__init__('cubic', smoothing=smoothing)
        self.name = 'RBF Cubic'
        self.str  = 'rbf_cubic'


class RBF_ThinPlateSpline(RBFSurrogate):
    """φ(r) = r² log(r)  — classic thin-plate spline, no shape parameter."""
    def __init__(self, smoothing: float = 1e-3):
        super().__init__('thin_plate_spline', smoothing=smoothing)
        self.name = 'RBF Thin-Plate Spline'
        self.str  = 'rbf_tps'


class RBF_Gaussian(RBFSurrogate):
    """φ(r) = exp(-(εr)²)."""
    def __init__(self, epsilon: float = 1.0, smoothing: float = 1e-3):
        super().__init__('gaussian', epsilon=epsilon, smoothing=smoothing)
        self.name = f'RBF Gaussian (ε={epsilon})'
        self.str  = 'rbf_gaussian'


class RBF_Multiquadric(RBFSurrogate):
    """φ(r) = −(1 + (εr)²)^(1/2)  — Multiquadric."""
    def __init__(self, epsilon: float = 1.0, smoothing: float = 1e-3):
        super().__init__('multiquadric', epsilon=epsilon, smoothing=smoothing)
        self.name = f'RBF Multiquadric (ε={epsilon})'
        self.str  = 'rbf_multiquadric'


class RBF_InverseQuadratic(RBFSurrogate):
    """φ(r) = 1 / (1 + (εr)²)."""
    def __init__(self, epsilon: float = 1.0, smoothing: float = 1e-3):
        super().__init__('inverse_quadratic', epsilon=epsilon, smoothing=smoothing)
        self.name = f'RBF Inverse Quadratic (ε={epsilon})'
        self.str  = 'rbf_invquad'


class RBF_InverseMultiquadric(RBFSurrogate):
    """φ(r) = 1 / (1 + (εr)²)^(1/2)."""
    def __init__(self, epsilon: float = 1.0, smoothing: float = 1e-3):
        super().__init__('inverse_multiquadric', epsilon=epsilon, smoothing=smoothing)
        self.name = f'RBF Inverse Multiquadric (ε={epsilon})'
        self.str  = 'rbf_invmultiquad'
