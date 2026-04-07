"""
Tree updater for DeepGBoost.

Each ``TreeUpdater`` wraps a single ``DecisionTreeRegressor`` and trains it
on a 1-D pseudo-residual vector, implementing the bagging base learner
described in Algorithm 1 of the paper.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class TreeUpdater:
    """
    Wraps a single CART tree for gradient fitting.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the underlying decision tree.
    max_features : int, float, str or None
        Number of features to consider at each split (passed directly to
        ``DecisionTreeRegressor``).  Use ``"sqrt"`` for the Random Forest
        default.
    random_state : int or None
        Seed for the underlying tree splitter.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self._tree = DecisionTreeRegressor(
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
        )

    def fit(
        self,
        X_sub: np.ndarray,
        pseudo_y_sub: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "TreeUpdater":
        """
        Fit the tree on a bootstrap subsample.

        Parameters
        ----------
        X_sub : (n_sub, n_features)
        pseudo_y_sub : (n_sub,)
            1-D pseudo-residuals for this tree's slot.
        sample_weight : (n_sub,) or None
            Per-sample weights (e.g. hessian values) passed to the
            underlying tree.  When not None, sklearn's weighted least
            squares splitting is used, focusing splits on uncertain samples.

        Returns
        -------
        self
        """
        self._tree.fit(X_sub, pseudo_y_sub, sample_weight=sample_weight)
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict gradient components for all samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_trees)
        """
        out = self._tree.predict(X)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out

    @property
    def feature_importances_(self) -> np.ndarray:
        """Impurity-based feature importances from the underlying tree."""
        return self._tree.feature_importances_

    @property
    def n_features_in_(self) -> int:
        return self._tree.n_features_in_
