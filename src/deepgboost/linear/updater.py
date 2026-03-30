"""
Linear updater for DeepGBoost (XGBoost ``gblinear`` analogue).

When ``linear_projection=True`` is set on the model, each boosting layer
additionally trains a Ridge regression on the mean pseudo-residuals.  This
captures linear trends that axis-aligned decision trees can miss, in the
same spirit as XGBoost's ``booster='gblinear'`` option.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class LinearUpdater:
    """
    Ridge regression weak learner for linear gradient components.

    Fits a single Ridge model on the *mean* pseudo-residuals across all
    tree slots, producing a scalar correction per sample.  A learned mixing
    weight (``alpha_mix``) blends the linear correction into each tree
    slot's prediction during the forward pass.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength for Ridge (default 1.0).
    fit_intercept : bool
        Whether to fit an intercept in the Ridge model.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        self.alpha_mix_: float = 0.5

    def fit(
        self,
        X_sub: np.ndarray,
        pseudo_y_mean_sub: np.ndarray,
    ) -> "LinearUpdater":
        """
        Fit Ridge on a subsample.

        Parameters
        ----------
        X_sub : (n_sub, n_features)
        pseudo_y_mean_sub : (n_sub,)
            Mean pseudo-residuals across tree slots.

        Returns
        -------
        self
        """
        self._ridge.fit(X_sub, pseudo_y_mean_sub)
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict linear correction.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self._ridge.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self._ridge.coef_

    @property
    def intercept_(self) -> float:
        return float(self._ridge.intercept_)
