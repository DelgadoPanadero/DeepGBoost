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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.preprocessing import LabelEncoder

from .gbm.dgbf import DGBFModel
from .callback import TrainingCallback
from .common.utils import sigmoid, softmax


# ---------------------------------------------------------------------------
# Shared parameter defaults (mirrors XGBModel grouping)
# ---------------------------------------------------------------------------

_TREE_PARAMS = ("n_trees", "n_layers", "max_depth")
_LEARNING_PARAMS = ("learning_rate", "subsample_min_frac")
_REGULARISATION_PARAMS = ("linear_projection", "linear_alpha")
_CONFIG_PARAMS = ("objective", "random_state", "n_jobs")
_CALLBACK_PARAMS = ("early_stopping_rounds", "eval_metric")


class DeepGBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    DeepGBoost classifier — sklearn-compatible interface.

    Supports binary and multiclass classification.

    * **Binary**: trains a single DGBF model in log-odds space
      (``LogisticObjective``).  ``predict_proba`` returns sigmoid outputs.
    * **Multiclass**: trains K binary classifiers (one-vs-rest), then
      normalises probabilities via softmax.

    Parameters
    ----------
    n_trees : int, default=10
        Number of trees per boosting layer.
    n_layers : int, default=10
        Number of boosting layers.
    max_depth : int or None, default=None
        Maximum depth of each decision tree.
    learning_rate : float, default=0.1
        Shrinkage factor.
    subsample_min_frac : float, default=0.3
        Minimum subsample fraction at layer 0.
    linear_projection : bool, default=False
        Add Ridge regression correction per layer.
    linear_alpha : float, default=1.0
        Ridge regularisation (only when ``linear_projection=True``).
    objective : str or None, default=None
        Override objective.  Auto-selected from number of classes if ``None``.
    random_state : int or None, default=None
        Seed for reproducibility.
    n_jobs : int, default=1
        Reserved for future use.
    early_stopping_rounds : int or None, default=None
        Early stopping patience (requires ``eval_set`` in ``fit``).
    eval_metric : str or None, default=None
        Metric for early stopping monitoring.
    """

    def __init__(
        self,
        n_trees: int = 10,
        n_layers: int = 10,
        max_depth: int | None = None,
        learning_rate: float = 0.1,
        subsample_min_frac: float = 0.3,
        linear_projection: bool = False,
        linear_alpha: float = 1.0,
        objective: str | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
    ):
        self.n_trees = n_trees
        self.n_layers = n_layers
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample_min_frac = subsample_min_frac
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
        X,
        y,
        *,
        eval_set: list[tuple] | None = None,
        callbacks: Sequence[TrainingCallback] | None = None,
        sample_weight=None,
    ) -> "DeepGBoostClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels (will be encoded internally).
        eval_set : list of (X_val, y_val) tuples, optional
        callbacks : list of TrainingCallback, optional
        sample_weight : ignored

        Returns
        -------
        self
        """
        X = check_array(X, dtype=np.float64)
        y_raw = np.asarray(y)

        # Encode labels to 0..K-1
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_raw)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        self.n_classes_ = n_classes
        self.n_features_in_ = X.shape[1]

        model_kw = dict(
            n_trees=self.n_trees,
            n_layers=self.n_layers,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            linear_projection=self.linear_projection,
            linear_alpha=self.linear_alpha,
            subsample_min_frac=self.subsample_min_frac,
            random_state=self.random_state,
        )

        all_callbacks = list(callbacks or [])
        if self.early_stopping_rounds is not None and eval_set:
            from .callback import EarlyStopping

            all_callbacks.append(
                EarlyStopping(patience=self.early_stopping_rounds)
            )

        if n_classes == 2:
            # Binary classification
            self._binary_model = self._fit_binary(
                X, y_enc.astype(np.float64), eval_set, all_callbacks, model_kw
            )
        else:
            # Multiclass: one-vs-rest
            self._ovr_models: list[DGBFModel] = []
            for k in range(n_classes):
                y_k = (y_enc == k).astype(np.float64)
                eval_set_k = None
                if eval_set:
                    eval_set_k = [
                        (
                            Xv,
                            (LabelEncoder().fit_transform(yv) == k).astype(
                                np.float64
                            ),
                        )
                        for Xv, yv in eval_set
                    ]
                model_k = self._fit_binary(
                    X, y_k, eval_set_k, all_callbacks, model_kw
                )
                self._ovr_models.append(model_k)

        return self

    def _fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set,
        callbacks: list,
        model_kw: dict,
    ) -> DGBFModel:
        objective = self.objective or "binary:logistic"
        model = DGBFModel(objective=objective, **model_kw)

        raw_evals = None
        if eval_set:
            raw_evals = [
                (
                    np.asarray(Xv, dtype=np.float64),
                    np.asarray(yv, dtype=np.float64).ravel(),
                    f"eval_{i}",
                )
                for i, (Xv, yv) in enumerate(eval_set)
            ]

        model.fit(X, y, callbacks=callbacks, evals=raw_evals)
        return model

    def predict_proba(self, X) -> np.ndarray:
        """
        Probability estimates.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, "classes_")
        X = check_array(X, dtype=np.float64)

        if self.n_classes_ == 2:
            raw = self._binary_model.predict_raw(X)  # log-odds
            p_pos = sigmoid(raw)
            return np.column_stack([1.0 - p_pos, p_pos])
        else:
            # OvR: collect raw log-odds from each binary model
            log_odds = np.column_stack(
                [m.predict_raw(X) for m in self._ovr_models]
            )  # (n_samples, K)
            return softmax(log_odds, axis=1)

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.

        Returns
        -------
        np.ndarray of shape (n_samples,) with original class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(indices)

    def score(self, X, y, sample_weight=None) -> float:
        """Return accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average feature importances across all binary sub-models."""
        check_is_fitted(self, "classes_")
        if self.n_classes_ == 2:
            return self._binary_model.feature_importances_
        importances = np.mean(
            [m.feature_importances_ for m in self._ovr_models], axis=0
        )
        return importances
