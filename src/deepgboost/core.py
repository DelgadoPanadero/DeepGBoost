"""
Core public objects: DeepGBoostDMatrix and DeepGBoostBooster.

Mirrors XGBoost's ``core.py`` module.

* ``DeepGBoostDMatrix`` — validated data wrapper (like ``xgboost.DMatrix``).
* ``DeepGBoostBooster`` — high-level model object with save/load and a
  dict-based parameter interface (like ``xgboost.Booster``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .gbm.dgbf import DGBFModel
from .callback import TrainingCallback


class DeepGBoostDMatrix:
    """
    Validated data container for DeepGBoost (mirrors ``xgboost.DMatrix``).

    Accepts numpy arrays or pandas DataFrames and normalises them to float64
    numpy arrays for training and inference.

    Parameters
    ----------
    data : array-like or DataFrame
        Feature matrix of shape (n_samples, n_features).
    label : array-like or None
        Target vector of shape (n_samples,).  Required for training.
    feature_names : list of str or None
        Column names.  Inferred from DataFrame columns if not provided.
    weight : array-like or None
        Per-sample weights (reserved for future use).
    """

    def __init__(
        self,
        data,
        label=None,
        feature_names: list[str] | None = None,
        weight=None,
    ):
        if isinstance(data, pd.DataFrame):
            if feature_names is None:
                feature_names = list(data.columns)
            data = data.to_numpy()

        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {self.data.shape}")

        self.label: np.ndarray | None = None
        if label is not None:
            self.label = np.asarray(label, dtype=np.float64).ravel()
            if self.label.shape[0] != self.data.shape[0]:
                raise ValueError(
                    "data and label must have the same number of rows. "
                    f"Got {self.data.shape[0]} vs {self.label.shape[0]}."
                )

        self.feature_names: list[str] = (
            feature_names
            if feature_names is not None
            else [f"f{i}" for i in range(self.data.shape[1])]
        )

        self.weight: np.ndarray | None = (
            np.asarray(weight, dtype=np.float64).ravel()
            if weight is not None
            else None
        )

    @property
    def num_row(self) -> int:
        return self.data.shape[0]

    @property
    def num_col(self) -> int:
        return self.data.shape[1]

    def __repr__(self) -> str:
        label_str = (
            f", label={self.label.shape}" if self.label is not None else ""
        )
        return (
            f"DeepGBoostDMatrix(shape={self.data.shape}{label_str}, "
            f"features={self.feature_names})"
        )


class DeepGBoostBooster:
    """
    High-level model object (mirrors ``xgboost.Booster``).

    Wraps a ``DGBFModel`` with a dict-based parameter interface,
    ``save_model``/``load_model`` support, and feature-importance access.

    Parameters
    ----------
    params : dict or None
        Model hyper-parameters.  Keys match ``DGBFModel.__init__`` arguments.

    Example
    -------
    ::

        from deepgboost import DeepGBoostDMatrix, DeepGBoostBooster

        dtrain = DeepGBoostDMatrix(X_train, label=y_train)
        bst = DeepGBoostBooster(params={"n_layers": 20, "learning_rate": 0.05})
        bst.train(dtrain)
        preds = bst.predict(DeepGBoostDMatrix(X_test))
    """

    _PARAM_KEYS = {
        "n_trees",
        "n_layers",
        "max_depth",
        "learning_rate",
        "linear_projection",
        "linear_alpha",
        "subsample_min_frac",
        "weight_solver",
        "objective",
        "random_state",
    }

    def __init__(
        self,
        params: dict | None = None,
    ):
        self.params: dict = params or {}
        self._model: DGBFModel | None = None

    # ------------------------------------------------------------------
    # Training & inference
    # ------------------------------------------------------------------

    def train(
        self,
        dtrain: DeepGBoostDMatrix,
        callbacks: Sequence[TrainingCallback] | None = None,
        evals: list[tuple[DeepGBoostDMatrix, str]] | None = None,
    ) -> "DeepGBoostBooster":
        """
        Fit the model on ``dtrain``.

        Parameters
        ----------
        dtrain : DeepGBoostDMatrix with ``label`` set.
        callbacks : list of TrainingCallback, optional
        evals : list of (DeepGBoostDMatrix, name) tuples, optional

        Returns
        -------
        self
        """
        if dtrain.label is None:
            raise ValueError("dtrain must have a label to train.")

        model_params = {
            k: v for k, v in self.params.items() if k in self._PARAM_KEYS
        }
        self._model = DGBFModel(**model_params)

        raw_evals = None
        if evals:
            raw_evals = [
                (dm.data, dm.label, name)
                for dm, name in evals
                if dm.label is not None
            ]

        self._model.fit(
            dtrain.data,
            dtrain.label,
            callbacks=callbacks,
            evals=raw_evals,
        )
        return self

    def predict(
        self,
        dtest: DeepGBoostDMatrix,
    ) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        dtest : DeepGBoostDMatrix (label not required)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_is_trained()
        return self._model.predict(dtest.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | Path) -> None:
        """Serialise the booster to a file using pickle."""
        self._check_is_trained()
        with open(path, "wb") as f:
            pickle.dump({"params": self.params, "model": self._model}, f)

    def load_model(self, path: str | Path) -> "DeepGBoostBooster":
        """Load a previously saved booster from a file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.params = state["params"]
        self._model = state["model"]
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Normalised impurity-based feature importances (or None if not fitted)."""
        if self._model is None:
            return None
        return self._model.feature_importances_

    @property
    def evals_result_(self) -> dict:
        """Training / validation metric history."""
        if self._model is None:
            return {}
        return self._model.evals_result_

    def get_score(self, importance_type: str = "weight") -> dict[str, float]:
        """
        Return feature importance scores as a dict {feature_name: score}.

        Currently supports ``importance_type='weight'`` (impurity-based).
        """
        self._check_is_trained()
        fi = self._model.feature_importances_
        n_features = self._model.n_features_in_
        names = [f"f{i}" for i in range(n_features)]
        return {name: float(score) for name, score in zip(names, fi)}

    def set_param(self, key: str, value) -> None:
        """Update a single parameter (takes effect on next ``train`` call)."""
        self.params[key] = value

    def _check_is_trained(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "This DeepGBoostBooster is not trained yet. Call 'train' first."
            )

    def __repr__(self) -> str:
        trained = self._model is not None
        return f"DeepGBoostBooster(params={self.params}, trained={trained})"
