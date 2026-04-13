"""Regression objective functions."""

from __future__ import annotations

import numpy as np


class BaseObjective:
    """Abstract base for all DeepGBoost objective functions."""

    def gradient(
        self,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pseudo-residuals (negative gradient of the loss).

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            True target values.
        F : np.ndarray of shape (n_samples,)
            Current ensemble prediction.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        raise NotImplementedError

    def prior(
        self,
        y: np.ndarray,
    ) -> float:
        """Optimal constant prediction (F_0 in the paper)."""
        raise NotImplementedError

    def hessian(
        self,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """
        Diagonal of the loss Hessian w.r.t. F (second derivatives).

        Returns ones by default, which is exact for MSE and a safe constant
        for MAE (non-differentiable).  Override for objectives where the
        Hessian varies with F (e.g. logistic).
        """
        return np.ones_like(F)

    def transform(
        self,
        raw: np.ndarray,
    ) -> np.ndarray:
        """Map raw model output to prediction space (identity for regression)."""
        return raw


class RMSEObjective(BaseObjective):
    """
    Root Mean Squared Error objective.

    Loss: L(y, F) = (y - F)^2 / 2
    Gradient: g = y - F
    Prior: mean(y)
    """

    def gradient(
        self,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pseudo-residuals for MSE loss.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            True target values.
        F : np.ndarray of shape (n_samples,)
            Current ensemble predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return y - F

    def prior(self, y: np.ndarray) -> float:
        """Return mean of y as the optimal constant prediction."""
        return float(y.mean())


class MAEObjective(BaseObjective):
    """
    Mean Absolute Error objective.

    Loss: L(y, F) = |y - F|
    Gradient: g = sign(y - F)
    Prior: median(y)
    """

    def gradient(
        self,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pseudo-residuals for MAE loss.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            True target values.
        F : np.ndarray of shape (n_samples,)
            Current ensemble predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return np.sign(y - F)

    def prior(self, y: np.ndarray) -> float:
        """Return median of y as the optimal constant prediction."""
        return float(np.median(y))
