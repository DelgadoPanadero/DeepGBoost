"""
Multi-output Distributed Gradient Boosting Forest — core training loop.

Extends the DGBF algorithm to learn residuals for all K classes simultaneously
using per-class single-output decision trees.  Each class trains n_trees
independent trees on its own pseudo-residual slice.  NNLS weights are solved
per-class, giving each class the freedom to weight its trees differently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..common.utils import bootstrap_sampler, weight_solver
from ..tree.updater import TreeUpdater
from ..objective.classification import SoftmaxObjective

if TYPE_CHECKING:
    from ..callbacks.base_callback import TrainingCallback


class DGBFMultiOutputModel:
    """
    Multi-output Distributed Gradient Boosting Forest model.

    Operates entirely in 2-D space (n_samples, K).  Each layer fits K groups
    of T single-output trees (one group per class) on the per-class
    pseudo-residual slice.  NNLS weights are solved per-class.

    Parameters
    ----------
    n_trees : int
        Number of trees per boosting layer per class.
    n_layers : int
        Number of boosting layers.
    max_depth : int or None
        Maximum depth of each decision tree.
    max_features : int, float, str or None
        Features considered at each split.
    min_weight_fraction_leaf : float, default=0.0
        Minimum fraction of the total (weighted) number of samples required
        to be at a leaf node.  Prevents leaves whose accumulated Hessian
        mass is too small, analogous to XGBoost's ``min_child_weight``.
        The default ``0.0`` preserves the original behaviour exactly.
    learning_rate : float
        Shrinkage factor applied to pseudo-residuals.
    subsample_min_frac : float
        Minimum subsample fraction at the first layer.  Grows to 1.0 at
        the last layer (dynamic sampling, paper sec. 3.1.3).
    weight_solver : str
        ``"nnls"`` or ``"uniform"``.
    hessian_reg : float
        L2 regularisation added to the Hessian denominator (mirrors XGBoost λ).
    random_state : int or None
        Master seed.
    """

    def __init__(
        self,
        n_trees: int = 10,
        n_layers: int = 10,
        max_depth: int | None = None,
        max_features: int | float | str | None = None,
        min_weight_fraction_leaf: float = 0.0,
        learning_rate: float = 0.1,
        subsample_min_frac: float = 0.3,
        weight_solver: str = "nnls",
        hessian_reg: float = 0.0,
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.n_layers = n_layers
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.learning_rate = learning_rate
        self.subsample_min_frac = subsample_min_frac
        self.weight_solver = weight_solver
        self.hessian_reg = hessian_reg
        self.random_state = random_state

        # Fitted state
        self.graph_: list[list[list[TreeUpdater]]] = []
        # weights_[l] has shape (K, n_trees): per-class combination weights
        self.weights_: list[np.ndarray] = []
        self.prior_: np.ndarray = np.array([])
        self.feature_importances_: np.ndarray | None = None
        self.n_features_in_: int = 0
        self.evals_result_: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        callbacks: Sequence["TrainingCallback"] | None = None,
        evals: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
    ) -> "DGBFMultiOutputModel":
        """
        Fit the multi-output DGBF model.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples, K) one-hot encoded targets
        callbacks : list of TrainingCallback, optional
        evals : list of (X_val, y_val_onehot, name) tuples, optional

        Returns
        -------
        self
        """
        if y.ndim != 2:
            raise ValueError(
                "DGBFMultiOutputModel requires a 2-D one-hot target y of shape "
                "(n_samples, K)."
            )

        obj = SoftmaxObjective()
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        K = y.shape[1]

        # Initialise state
        self.graph_ = []
        self.weights_ = []
        self.prior_ = obj.prior(y)  # (K,)
        self.n_features_in_ = n_features
        self.evals_result_ = {}
        self._layer_cond_numbers_: list[float] = []
        feature_importance_accum = np.zeros(n_features)

        if evals:
            for _, _, name in evals:
                self.evals_result_[name] = {"logloss": []}

        callbacks = callbacks or []
        for cb in callbacks:
            cb.before_training(self)

        for layer_idx in range(self.n_layers):
            F_prev = self.predict_raw(X)  # (n_samples, K)

            g = obj.gradient(y, F_prev)   # (n_samples, K)
            h = obj.hessian(y, F_prev)    # (n_samples, K)

            pseudo_y = (
                g / np.maximum(h + self.hessian_reg, 1e-7)
            ) * self.learning_rate  # (n_samples, K)

            stop = False
            evals_log: dict = {}
            for cb in callbacks:
                if cb.before_iteration(self, layer_idx, evals_log):
                    stop = True
            if stop:
                break

            new_layer, new_weights, layer_cond = self._fit_layer(
                X, pseudo_y, layer_idx, rng, h, n_samples, K
            )
            self.graph_.append(new_layer)
            self.weights_.append(new_weights)
            self._layer_cond_numbers_.append(layer_cond)

            # new_layer is list[list[TreeUpdater]] of shape (K, n_trees)
            for class_trees in new_layer:
                for tree in class_trees:
                    feature_importance_accum += tree.feature_importances_

            if evals:
                for X_val, y_val, name in evals:
                    F_val = self.predict_raw(X_val)  # (n_val, K)
                    p_val = obj.transform(F_val)     # softmax → (n_val, K)
                    p_val = np.clip(p_val, 1e-7, 1.0 - 1e-7)
                    logloss = float(
                        -np.mean(np.sum(y_val * np.log(p_val), axis=1))
                    )
                    self.evals_result_.setdefault(name, {}).setdefault(
                        "logloss", []
                    ).append(logloss)
                    evals_log[name] = {"logloss": logloss}

            stop = False
            for cb in callbacks:
                if cb.after_iteration(self, layer_idx, evals_log):
                    stop = True
            if stop:
                break

        total = feature_importance_accum.sum()
        self.feature_importances_ = (
            feature_importance_accum / total
            if total > 0
            else feature_importance_accum
        )

        for cb in callbacks:
            cb.after_training(self)

        return self

    def _fit_layer(
        self,
        X: np.ndarray,
        pseudo_y: np.ndarray,
        layer_idx: int,
        rng: np.random.Generator,
        hessian: np.ndarray,
        n_samples: int,
        K: int,
    ) -> tuple[list[list[TreeUpdater]], np.ndarray, float]:
        """
        Fit K independent groups of n_trees single-output trees for one layer.

        Each class k trains n_trees trees on target ``pseudo_y[:, k]`` with
        ``sample_weight=hessian[:, k]``.  NNLS weights are solved per-class
        from the single-output predictions.

        Parameters
        ----------
        X : (n_samples, n_features)
        pseudo_y : (n_samples, K)
        layer_idx : int
        rng : np.random.Generator
        hessian : (n_samples, K)
            Per-sample per-class Hessian.
        n_samples : int
        K : int

        Returns
        -------
        new_layer : list[list[TreeUpdater]] of shape (K, n_trees)
        layer_weights : (K, n_trees)
        cond : float
            Mean condition number used as a diagnostic.
        """
        new_layer: list[list[TreeUpdater]] = []
        layer_weights = np.zeros((K, self.n_trees))
        cond_values: list[float] = []

        for k in range(K):
            class_trees: list[TreeUpdater] = []
            class_preds: list[np.ndarray] = []  # each (n_samples,)

            for t in range(self.n_trees):
                sample_idx = bootstrap_sampler(
                    n_samples=n_samples,
                    n_layers=self.n_layers,
                    layer_idx=layer_idx,
                    subsample_min_frac=self.subsample_min_frac,
                    rng=rng,
                )

                tree_seed = int(rng.integers(0, 2**31))
                tree = TreeUpdater(
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                    random_state=tree_seed,
                )

                # Exact per-class Hessian as sample_weight
                sw = hessian[sample_idx, k]  # (n_sub,)
                tree.fit(
                    X[sample_idx],
                    pseudo_y[sample_idx, k],  # (n_sub,) — single-output
                    sample_weight=sw,
                )

                # tree.predict returns (n_samples, 1) for single-output tree; flatten
                class_preds.append(tree.predict(X)[:, 0])
                class_trees.append(tree)

            # (n_samples, n_trees) predictor matrix for class k
            preds_k = np.column_stack(class_preds)
            cond_values.append(float(np.linalg.cond(preds_k)))

            layer_weights[k] = weight_solver(
                preds_k,           # (n_samples, n_trees)
                pseudo_y[:, k],    # (n_samples,)
                method=self.weight_solver,
                sample_weight=hessian[:, k],
            )
            new_layer.append(class_trees)

        cond = float(np.mean(cond_values))
        return new_layer, layer_weights, cond

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Raw ensemble output before softmax.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, K)
        """
        n_samples = X.shape[0]
        K = len(self.prior_)
        accum = np.tile(self.prior_, (n_samples, 1)).copy()  # (n_samples, K)

        for layer_idx, layer in enumerate(self.graph_):
            # layer is list[list[TreeUpdater]] of shape (K, n_trees)
            for k, class_trees in enumerate(layer):
                for t, tree in enumerate(class_trees):
                    # tree.predict returns (n_samples, 1) for single-output; flatten
                    accum[:, k] += (
                        self.weights_[layer_idx][k, t] * tree.predict(X)[:, 0]
                    )

        return accum  # (n_samples, K)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.graph_:
            raise RuntimeError(
                "This DGBFMultiOutputModel instance is not fitted yet. "
                "Call 'fit' first."
            )

    def get_params(self) -> dict:
        return {
            "n_trees": self.n_trees,
            "n_layers": self.n_layers,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "learning_rate": self.learning_rate,
            "subsample_min_frac": self.subsample_min_frac,
            "weight_solver": self.weight_solver,
            "hessian_reg": self.hessian_reg,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        p = self.get_params()
        parts = ", ".join(f"{k}={v!r}" for k, v in p.items())
        return f"DGBFMultiOutputModel({parts})"
