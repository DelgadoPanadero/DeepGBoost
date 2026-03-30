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

    def gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        return y - sigmoid(F)

    def prior(self, y: np.ndarray) -> float:
        p = float(y.mean())
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(np.log(p / (1.0 - p)))

    def transform(self, raw: np.ndarray) -> np.ndarray:
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

    def gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        y : (n_samples, n_classes) one-hot encoded targets.
        F : (n_samples, n_classes) raw log-scores.
        """
        return y - softmax(F, axis=1)

    def prior(self, y: np.ndarray) -> np.ndarray:
        """Log-odds prior for each class (shape n_classes)."""
        if y.ndim == 1:
            raise ValueError("SoftmaxObjective requires one-hot encoded y (2-D).")
        p = y.mean(axis=0)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1.0 - p))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return softmax(raw, axis=1)
