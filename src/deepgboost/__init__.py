"""
DeepGBoost — Distributed Gradient Boosting Forest

A deep-graph tree ensemble algorithm combining boosting and bagging in a
neural-network-like layered architecture.

Based on the paper:
    "A generalized decision tree ensemble based on the NeuralNetworks
     architecture: Distributed Gradient Boosting Forest (DGBF)"
     Delgado-Panadero et al., Applied Intelligence 53, 22991–23003 (2023).
     https://doi.org/10.1007/s10489-023-04735-w

Quick start
-----------
Sklearn API::

    from deepgboost import DeepGBoostRegressor, DeepGBoostClassifier

    reg = DeepGBoostRegressor(n_layers=20, n_trees=10, learning_rate=0.05)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
"""

from .deepgboost_regressor import DeepGBoostRegressor
from .deepgboost_classifier import DeepGBoostClassifier
from .callbacks import TrainingCallback
from .callbacks import EarlyStoppingCallback
from .callbacks import LearningRateSchedulerCallback
from .callbacks import EvaluationMonitorCallback
from .plotting import plot_importance
from .gbm.dgbf import DGBFModel
from .objective import get_objective
from .metric import get_metric

__version__ = "0.1.0"

__all__ = [
    # Sklearn estimators
    "DeepGBoostRegressor",
    "DeepGBoostClassifier",
    # Callbacks
    "TrainingCallback",
    "EarlyStopping",
    "LearningRateScheduler",
    "EvaluationMonitor",
    # Plotting
    "plot_importance",
    # Low-level
    "DGBFModel",
    "get_objective",
    "get_metric",
    # Version
    "__version__",
]
