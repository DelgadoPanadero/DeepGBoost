"""Classification evaluation metrics."""

from __future__ import annotations

import numpy as np

from .regression import BaseMetric


class AccuracyMetric(BaseMetric):
    """Classification accuracy."""

    name = "accuracy"
    higher_is_better = True

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))


class LogLossMetric(BaseMetric):
    """
    Binary cross-entropy (log-loss).

    y_pred should be probabilities in [0, 1].
    """

    name = "logloss"
    higher_is_better = False

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        p = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return float(
            -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        )


class AUCMetric(BaseMetric):
    """
    Area Under the ROC Curve (binary classification).

    Uses the trapezoidal rule.  y_pred should be probabilities.
    """

    name = "auc"
    higher_is_better = True

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        order = np.argsort(y_pred)[::-1]
        y_sorted = y_true[order]
        n_pos = y_sorted.sum()
        n_neg = len(y_sorted) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        tpr = tp / n_pos
        fpr = fp / n_neg
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        return float(trapz_fn(tpr, fpr))
