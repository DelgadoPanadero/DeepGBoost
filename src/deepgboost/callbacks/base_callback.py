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

class TrainingCallback:
    """
    Abstract base class for DeepGBoost training callbacks.

    Subclass this and override any of the four hook methods you need.
    All methods have safe default implementations (no-op / False).
    """

    def before_training(
        self,
        model,
    ) -> None:
        """Called once before the layer loop begins."""

    def after_training(
        self,
        model,
    ) -> None:
        """Called once after the layer loop ends (or early-stopping)."""

    def before_iteration(
        self,
        model,
        epoch: int,
        evals_log: dict,
    ) -> bool:
        """
        Called at the start of each boosting layer.

        Returns
        -------
        bool
            ``True`` to stop training before fitting this layer.
        """
        return False

    def after_iteration(
        self,
        model,
        epoch: int,
        evals_log: dict,
    ) -> bool:
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
