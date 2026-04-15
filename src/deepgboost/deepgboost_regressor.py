"""
Scikit-learn compatible estimators for DeepGBoost.

Mirrors ``xgboost.sklearn`` with ``DeepGBoostRegressor`` and
``DeepGBoostClassifier``.  Both classes follow the standard sklearn API
(``fit`` / ``predict`` / ``score`` / ``get_params`` / ``set_params``) and
are compatible with ``sklearn.clone``, ``GridSearchCV``, and pipelines.

Classification supports:
* Binary: predictions in log-odds space → sigmoid → probability.
* Multiclass: one-vs-rest binary classifiers → softmax-normalised probabilities.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .dgbf.dgbf import DGBFModel
from .callbacks.base_callback import TrainingCallback
from .common.categorical import CategoricalEncoderMixin


class DeepGBoostRegressor(
    CategoricalEncoderMixin,
    BaseEstimator,
    RegressorMixin,
):
    """
    DeepGBoost regressor — sklearn-compatible interface.

    Implements the DGBF algorithm (Delgado-Panadero et al., 2023) for
    regression tasks.

    Parameters
    ----------
    n_trees : int, default=10
        Number of trees (T) per boosting layer.
    n_layers : int, default=10
        Number of boosting layers (L).
    max_depth : int or None, default=None
        Maximum depth of each decision tree.
    max_features : int, float, str or None, default=None
        Number of features to consider at each split.  ``None`` uses all
        features (original DGBF behaviour).  Set to ``"sqrt"`` for the
        standard Random Forest feature subsampling; combined with
        ``n_layers=1`` the model becomes analogous to a RandomForest.
    learning_rate : float, default=0.1
        Shrinkage factor applied to pseudo-residuals each layer.
    subsample_min_frac : float, default=0.3
        Minimum subsample fraction at the first layer (grows to 1.0).
    weight_solver : str, default="nnls"
        How to combine the T bagged trees in each layer.  ``"nnls"`` finds
        optimal non-negative weights via Non-Negative Least Squares.
        ``"uniform"`` assigns equal weight to every tree (standard
        RandomForest averaging); with ``n_layers=1`` and
        ``learning_rate=1.0`` this makes the model exactly equivalent to a
        RandomForest.
    hessian_reg : float, default=0.0
        L2 regularisation added to the Hessian denominator of the Newton step:
        ``pseudo_y = g / (h + hessian_reg) * lr``.  Mirrors XGBoost's
        ``lambda`` parameter.  For MSE regression (h=1 everywhere) the
        effect is negligible at the default value of 0.0.
    linear_projection : bool, default=False
        Add a Ridge regression correction at each layer (XGBoost gblinear
        analogue) to capture linear trends that trees cannot model.
    linear_alpha : float, default=1.0
        L2 regularisation for the linear projection Ridge model.
    objective : str, default="reg:squarederror"
        Loss function.  Options: ``"reg:squarederror"``,
        ``"reg:absoluteerror"``.
    random_state : int or None, default=None
        Seed for reproducibility.
    n_jobs : int, default=1
        Reserved for future parallel tree fitting.
    early_stopping_rounds : int or None, default=None
        Stop if no improvement for this many rounds (requires ``eval_set``
        in ``fit``).
    eval_metric : str or None, default=None
        Metric to monitor for early stopping.  Defaults to ``"rmse"``.
    """

    def __init__(
        self,
        n_trees: int = 20,
        n_layers: int = 5,
        max_depth: int | None = None,
        max_features: int | float | str | None = None,
        learning_rate: float = 0.8,
        subsample_min_frac: float = 0.3,
        weight_solver: str = "nnls",
        hessian_reg: float = 0.0,
        linear_projection: bool = False,
        linear_alpha: float = 1.0,
        objective: str = "reg:squarederror",
        random_state: int | None = None,
        n_jobs: int = 1,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
    ):
        self.n_trees = n_trees
        self.n_layers = n_layers
        self.max_depth = max_depth
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.subsample_min_frac = subsample_min_frac
        self.weight_solver = weight_solver
        self.hessian_reg = hessian_reg
        self.linear_projection = linear_projection
        self.linear_alpha = linear_alpha
        self.objective = objective
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

    # ------------------------------------------------------------------
    # Sklearn interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        eval_set: list[tuple] | None = None,
        callbacks: Sequence[TrainingCallback] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> "DeepGBoostRegressor":
        """
        Fit the DeepGBoost regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : list of (X_val, y_val) tuples, optional
            Validation sets for early stopping / monitoring.
        callbacks : list of TrainingCallback, optional
        sample_weight : ignored (reserved for future use)

        Returns
        -------
        self
        """
        X = self._fit_transform_X(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.model_ = DGBFModel(
            n_trees=self.n_trees,
            n_layers=self.n_layers,
            max_depth=self.max_depth,
            max_features=self.max_features,
            learning_rate=self.learning_rate,
            subsample_min_frac=self.subsample_min_frac,
            weight_solver=self.weight_solver,
            hessian_reg=self.hessian_reg,
            linear_projection=self.linear_projection,
            linear_alpha=self.linear_alpha,
            objective=self.objective,
            random_state=self.random_state,
        )

        all_callbacks = list(callbacks or [])

        raw_evals = None
        if eval_set:
            raw_evals = [
                (
                    self._transform_X(Xv),
                    np.asarray(yv, dtype=np.float64).ravel(),
                    f"eval_{i}",
                )
                for i, (Xv, yv) in enumerate(eval_set)
            ]
            if self.early_stopping_rounds is not None:
                from .callbacks import EarlyStoppingCallback

                all_callbacks.append(
                    EarlyStoppingCallback(patience=self.early_stopping_rounds),
                )

        self.model_.fit(
            X=X,
            y=y,
            callbacks=all_callbacks,
            evals=raw_evals,
        )
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict regression targets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        X = self._transform_X(X)
        return self.model_.predict(X)

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        """Return the R² coefficient of determination."""
        check_is_fitted(self, "model_")
        from .metric.regression import R2ScoreMetric

        return R2ScoreMetric()(np.asarray(y).ravel(), self.predict(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Impurity-based feature importances from the underlying DGBF model."""
        check_is_fitted(self, "model_")

        if (feature_importances := self.model_.feature_importances_) is None:
            raise Exception()

        return feature_importances

    @property
    def evals_result_(self) -> dict:
        """Evaluation results logged during training, keyed by dataset name."""
        check_is_fitted(self, "model_")
        return self.model_.evals_result_
