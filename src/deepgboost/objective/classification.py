"""Classification objective functions."""

from __future__ import annotations

import numpy as np

from .regression import BaseObjective
from ..common.utils import sigmoid, softmax


class LogisticObjective(BaseObjective):
    """
    Binary logistic (log-loss) objective.

    Training operates in log-odds space.
    Loss: L(y, F) = -[y*log(p) + (1-y)*log(1-p)],  p = sigmoid(F)
    Gradient: g = y - sigmoid(F)
    Prior: log(p_mean / (1 - p_mean))  (log-odds of the class rate)
    """

    def gradient(
        self,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pseudo-residuals for binary logistic loss.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            True binary targets in {0, 1}.
        F : np.ndarray of shape (n_samples,)
            Current raw ensemble predictions (log-odds).

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return y - sigmoid(F)

    def hessian(
        self,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """
        Diagonal of the loss Hessian: h_i = p_i * (1 - p_i), p_i = sigmoid(F_i).

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            True binary targets (unused; kept for API consistency).
        F : np.ndarray of shape (n_samples,)
            Current raw ensemble predictions (log-odds).

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        p = sigmoid(F)
        return p * (1.0 - p)

    def prior(
        self,
        y: np.ndarray,
    ) -> float:
        """
        Compute log-odds of the positive-class rate as the initial prediction.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            True binary targets in {0, 1}.

        Returns
        -------
        float
        """
        p = float(y.mean())
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(np.log(p / (1.0 - p)))

    def transform(
        self,
        raw: np.ndarray,
    ) -> np.ndarray:
        """Map log-odds to probabilities."""
        return sigmoid(raw)


class SoftmaxObjective(BaseObjective):
    """
    Multi-class softmax objective (used internally per class in OvR).

    Gradient: g = y_k - softmax(F)_k
    Prior: log-odds for each class.

    Note: DeepGBoostClassifier uses one-vs-rest (OvR) binary classifiers
    with LogisticObjective.  SoftmaxObjective is provided for direct use
    with multi-output targets.
    """

    def gradient(
        self,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-class pseudo-residuals.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples, n_classes)
            One-hot encoded targets.
        F : np.ndarray of shape (n_samples, n_classes)
            Raw log-scores.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        return y - softmax(F, axis=1)

    def hessian(
        self,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """
        Diagonal of the per-class Hessian: p_k * (1 - p_k).

        Parameters
        ----------
        y : np.ndarray of shape (n_samples, n_classes)
            One-hot encoded targets (unused; kept for API consistency).
        F : np.ndarray of shape (n_samples, n_classes)
            Raw log-scores.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        p = softmax(F, axis=1)
        return p * (1.0 - p)

    def prior(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """Log-odds prior for each class (shape n_classes)."""
        if y.ndim == 1:
            raise ValueError(
                "SoftmaxObjective requires one-hot encoded y (2-D).",
            )
        p = y.mean(axis=0)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1.0 - p))

    def transform(
        self,
        raw: np.ndarray,
    ) -> np.ndarray:
        """Map raw log-scores to class probabilities via softmax."""
        return softmax(raw, axis=1)
