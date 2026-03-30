"""Regression evaluation metrics."""

from __future__ import annotations

import numpy as np


class BaseMetric:
    """Abstract base for all DeepGBoost evaluation metrics."""

    name: str = ""
    higher_is_better: bool = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RMSEMetric(BaseMetric):
    """Root Mean Squared Error."""

    name = "rmse"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class MAEMetric(BaseMetric):
    """Mean Absolute Error."""

    name = "mae"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))


class R2ScoreMetric(BaseMetric):
    """Coefficient of determination R²."""

    name = "r2"
    higher_is_better = True

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)
