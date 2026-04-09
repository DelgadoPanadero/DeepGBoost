"""
DeepGBoost multi-output classifier.

Uses a single DGBFMultiOutputModel where all K classes' residuals are learned
jointly by multi-output decision trees, capturing cross-class split dependencies.

Compared to DeepGBoostClassifier (OvR), each tree split is chosen to minimise
residuals across all classes simultaneously rather than for one class in isolation.

Note: linear_projection is not supported in this classifier.  Use
DeepGBoostClassifier if you need linear correction layers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder, label_binarize

from .gbm.dgbf_multioutput import DGBFMultiOutputModel
from .callbacks.base_callback import TrainingCallback
from .common.utils import softmax
from .common.categorical import CategoricalEncoderMixin


class DeepGBoostMultiClassifier(
    CategoricalEncoderMixin, BaseEstimator, ClassifierMixin
):
    """
    DeepGBoost multi-output classifier — sklearn-compatible interface.

    Trains a single ``DGBFMultiOutputModel`` where each boosting tree learns
    residuals for all K classes simultaneously.  This contrasts with
    ``DeepGBoostClassifier`` which trains K independent OvR binary models.

    Parameters
    ----------
    n_trees : int, default=5
        Number of trees per boosting layer.
    n_layers : int, default=20
        Number of boosting layers.
    max_depth : int or None, default=None
        Maximum depth of each decision tree.
    max_features : int, float, str or None, default=None
        Features to consider at each split.  ``None`` uses all features.
        Set to ``"sqrt"`` for Random Forest-style feature subsampling.
    learning_rate : float, default=0.1
        Shrinkage factor.
    subsample_min_frac : float, default=0.3
        Minimum subsample fraction at layer 0.
    weight_solver : str, default="nnls"
        How to combine trees in each layer.  ``"nnls"`` finds optimal
        non-negative weights per class; ``"uniform"`` assigns equal weight.
    hessian_reg : float, default=0.0
        L2 regularisation added to the Hessian denominator.
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
        n_trees: int = 5,
        n_layers: int = 20,
        max_depth: int | None = None,
        max_features: int | float | str | None = None,
        learning_rate: float = 0.1,
        subsample_min_frac: float = 0.3,
        weight_solver: str = "nnls",
        hessian_reg: float = 0.0,
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
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_y(self, y_enc: np.ndarray) -> np.ndarray:
        """Encode integer class labels to one-hot (n_samples, K)."""
        y_onehot = label_binarize(y_enc, classes=np.arange(self.n_classes_))
        # label_binarize returns (n, 1) for binary — expand to (n, 2)
        if self.n_classes_ == 2:
            y_onehot = np.hstack([1 - y_onehot, y_onehot])
        return y_onehot.astype(np.float64)

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
    ) -> "DeepGBoostMultiClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels (encoded internally).
        eval_set : list of (X_val, y_val) tuples, optional
        callbacks : list of TrainingCallback, optional
        sample_weight : ignored

        Returns
        -------
        self
        """
        X = self._fit_transform_X(X)
        y_raw = np.asarray(y)

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_raw)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        y_onehot = self._encode_y(y_enc)  # (n_samples, K)

        raw_evals = None
        if eval_set:
            raw_evals = []
            for i, (X_val, y_val) in enumerate(eval_set):
                X_val_enc = self._transform_X(X_val)
                y_val_enc = self.label_encoder_.transform(np.asarray(y_val))
                y_val_onehot = self._encode_y(y_val_enc)
                raw_evals.append((X_val_enc, y_val_onehot, f"eval_{i}"))

        all_callbacks = list(callbacks or [])
        if self.early_stopping_rounds is not None and raw_evals:
            from .callbacks import EarlyStoppingCallback

            all_callbacks.append(
                EarlyStoppingCallback(patience=self.early_stopping_rounds)
            )

        self.model_ = DGBFMultiOutputModel(
            n_trees=self.n_trees,
            n_layers=self.n_layers,
            max_depth=self.max_depth,
            max_features=self.max_features,
            learning_rate=self.learning_rate,
            subsample_min_frac=self.subsample_min_frac,
            weight_solver=self.weight_solver,
            hessian_reg=self.hessian_reg,
            random_state=self.random_state,
        )
        self.model_.fit(X, y_onehot, callbacks=all_callbacks, evals=raw_evals)

        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Probability estimates.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Rows sum to 1.
        """
        check_is_fitted(self, "classes_")
        X = self._transform_X(X)
        F = self.model_.predict_raw(X)  # (n_samples, K)
        return softmax(F, axis=1)

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
        """Impurity-based feature importances averaged across all trees and layers."""
        check_is_fitted(self, "model_")
        return self.model_.feature_importances_
