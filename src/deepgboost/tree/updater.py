"""
Tree updater for DeepGBoost.

Each ``TreeUpdater`` wraps a single ``DecisionTreeRegressor`` and trains it
on multi-output pseudo-residuals (shape n_sub × n_trees), implementing the
distributed gradient learning described in Algorithm 1 of the paper.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class TreeUpdater:
    """
    Wraps a single CART tree for multi-output gradient fitting.

    The tree learns T gradient components simultaneously (one per tree slot
    in the layer), which is the key operation behind the distributed gradient
    representation of DGBF.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the underlying decision tree.
    random_state : int or None
        Seed for the underlying tree splitter.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.random_state = random_state
        self._tree = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(
        self,
        X_sub: np.ndarray,
        pseudo_y_sub: np.ndarray,
    ) -> "TreeUpdater":
        """
        Fit the tree on a bootstrap subsample.

        Parameters
        ----------
        X_sub : (n_sub, n_features)
        pseudo_y_sub : (n_sub, n_trees)
            Multi-output pseudo-residuals — one column per tree slot.

        Returns
        -------
        self
        """
        self._tree.fit(X_sub, pseudo_y_sub)
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
