"""
Callback system for DeepGBoost (mirrors XGBoost's callback module).

Callbacks hook into the training loop at four points:
  before_training / after_training — called once per fit() call.
  before_iteration / after_iteration — called once per boosting layer.

``before_iteration`` and ``after_iteration`` return a boolean; returning
``True`` signals early stopping.

Usage example::

    from deepgboost import DeepGBoostRegressor, EarlyStopping

    es = EarlyStopping(patience=5, metric="train_loss")
    reg = DeepGBoostRegressor(n_layers=50)
    reg.fit(X_train, y_train, callbacks=[es],
            evals=[(X_val, y_val, "val")])
"""

from __future__ import annotations

import copy
from typing import Any


class TrainingCallback:
    """
    Abstract base class for DeepGBoost training callbacks.

    Subclass this and override any of the four hook methods you need.
    All methods have safe default implementations (no-op / False).
    """

    def before_training(self, model) -> None:
        """Called once before the layer loop begins."""

    def after_training(self, model) -> None:
        """Called once after the layer loop ends (or early-stopping)."""

    def before_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        """
        Called at the start of each boosting layer.

        Returns
        -------
        bool
            ``True`` to stop training before fitting this layer.
        """
        return False

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        """
        Called at the end of each boosting layer.

        Parameters
        ----------
        evals_log : dict
            ``{dataset_name: {metric_name: latest_value}}`` for all eval sets.

        Returns
        -------
        bool
            ``True`` to stop training after this layer.
        """
        return False


class EarlyStopping(TrainingCallback):
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of layers with no improvement before stopping.
    metric : str
        Metric key to monitor inside ``evals_log`` values.
        The key is looked up in the *first* eval set in ``evals_log``.
    data : str or None
        Name of the eval dataset to monitor.  If ``None``, uses the first
        dataset found in ``evals_log``.
    restore_best : bool
        If ``True``, restores the model to the best-seen state when stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        metric: str = "train_loss",
        data: str | None = None,
        restore_best: bool = True,
        min_delta: float = 1e-6,
    ):
        self.patience = patience
        self.metric = metric
        self.data = data
        self.restore_best = restore_best
        self.min_delta = min_delta

        self._best_score: float | None = None
        self._best_epoch: int = 0
        self._best_graph: Any = None
        self._best_weights: Any = None
        self._best_linear: Any = None
        self._wait: int = 0

    def before_training(self, model) -> None:
        self._best_score = None
        self._best_epoch = 0
        self._wait = 0

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if not evals_log:
            return False

        # Pick dataset to monitor
        dataset = self.data or next(iter(evals_log))
        if dataset not in evals_log:
            return False

        score = evals_log[dataset].get(self.metric)
        if score is None:
            return False

        # Determine if improvement (lower is better for loss metrics)
        improved = (
            self._best_score is None
            or score < self._best_score - self.min_delta
        )

        if improved:
            self._best_score = score
            self._best_epoch = epoch
            self._wait = 0
            if self.restore_best:
                self._best_graph = copy.deepcopy(model.graph_)
                self._best_weights = copy.deepcopy(model.weights_)
                self._best_linear = copy.deepcopy(model.linear_models_)
        else:
            self._wait += 1

        if self._wait >= self.patience:
            if self.restore_best and self._best_graph is not None:
                model.graph_ = self._best_graph
                model.weights_ = self._best_weights
                model.linear_models_ = self._best_linear
            return True  # stop

        return False


class LearningRateScheduler(TrainingCallback):
    """
    Adjust ``model.learning_rate`` before each boosting layer.

    Parameters
    ----------
    schedule_fn : callable
        A function ``f(epoch: int) -> float`` that returns the new
        learning rate for that layer.

    Example::

        scheduler = LearningRateScheduler(lambda epoch: 0.1 * 0.95**epoch)
    """

    def __init__(self, schedule_fn):
        self.schedule_fn = schedule_fn

    def before_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        model.learning_rate = float(self.schedule_fn(epoch))
        return False


class EvaluationMonitor(TrainingCallback):
    """
    Print evaluation metrics to stdout after each layer.

    Parameters
    ----------
    period : int
        Print every ``period`` layers (default 1 = every layer).
    """

    def __init__(self, period: int = 1):
        self.period = period

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if (epoch + 1) % self.period == 0 and evals_log:
            parts = []
            for dataset, metrics in evals_log.items():
                for metric, val in metrics.items():
                    parts.append(f"{dataset}-{metric}: {val:.6f}")
            print(f"[{epoch + 1}]\t" + "\t".join(parts))
        return False
