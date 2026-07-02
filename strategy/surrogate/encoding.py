"""strategy/surrogate/encoding.py --- Encoding-aware surrogate features (SAMOS2 C1).

Tree-based surrogates (XGBoost, RFR, ...) split on ``feature <= threshold``,
which silently assumes the encoded integers are ordered. For NAS search
spaces this is only true for width/depth/channel genes (NATS, ResNet-50D,
Transformer, MobileNetV3); *operator-choice* genes (NB201 ops, NB101 ops) are
unordered categories, and one-hot encoding them removes the fake order the
surrogate would otherwise split on.

``EncodingSpec`` reads the categorical/ordinal split from
``problem.evoxbench.benchmark_meta.OP_VAR_GROUPS`` and exposes
``transform(X) -> X_feat`` for use ahead of ``surrogate.fit`` / ``.predict``.
``EncodingAwareSurrogate`` wraps a base surrogate so the transform is applied
transparently -- the surrogate_problem_factory (e.g. SurrogateProblemEvox)
keeps calling plain ``.fit`` / ``.predict`` / ``.predict_std`` and never has
to know encoding is happening.
"""

from __future__ import annotations

import numpy as np

from problem.evoxbench.benchmark_meta import get_op_var_group


class EncodingSpec:
    """categorical_cols get one-hot encoded; every other column passes through."""

    def __init__(self, categorical_cols, n_var: int, cardinalities: dict):
        self.categorical_cols = sorted(set(int(c) for c in categorical_cols))
        self.n_var = n_var
        self.ordinal_cols = [c for c in range(n_var) if c not in self.categorical_cols]
        self.cardinalities = cardinalities

    @classmethod
    def from_search_space(cls, search_space_abbrev: str, n_var: int, xu) -> "EncodingSpec | None":
        """Build from OP_VAR_GROUPS metadata; None if the space has no
        categorical-op structure (encoding would be a no-op there)."""
        group = get_op_var_group(search_space_abbrev)
        if group is None:
            return None
        xu = np.asarray(xu, dtype=int)
        cols = list(range(n_var)) if group['all_vars_are_ops'] else list(group['op_var_indices'])
        cardinalities = {c: int(xu[c]) + 1 for c in cols}
        return cls(categorical_cols=cols, n_var=n_var, cardinalities=cardinalities)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_int = np.round(X).astype(int)
        blocks = []
        if self.ordinal_cols:
            blocks.append(X[:, self.ordinal_cols])
        for c in self.categorical_cols:
            card = self.cardinalities[c]
            onehot = np.zeros((len(X), card), dtype=float)
            vals = np.clip(X_int[:, c], 0, card - 1)
            onehot[np.arange(len(X)), vals] = 1.0
            blocks.append(onehot)
        return np.hstack(blocks) if blocks else X


class EncodingAwareSurrogate:
    """Wraps a base surrogate, applying EncodingSpec.transform before
    fit/predict/predict_std. Drop-in replacement for the base surrogate."""

    def __init__(self, base, encoding_spec: EncodingSpec):
        self.base = base
        self.encoding_spec = encoding_spec

    def fit(self, X, y):
        self.base.fit(self.encoding_spec.transform(X), y)
        return self

    def predict(self, X):
        return self.base.predict(self.encoding_spec.transform(X))

    def predict_std(self, X):
        return self.base.predict_std(self.encoding_spec.transform(X))

    def __str__(self):
        return f'EncodingAware({self.base})'
