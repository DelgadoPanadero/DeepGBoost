"""
Distributed Gradient Boosting Forest (DGBF) — core training loop.

Implements Algorithm 1 from the paper:
    "A generalized decision tree ensemble based on the NeuralNetworks
     architecture: Distributed Gradient Boosting Forest (DGBF)"
     Delgado-Panadero et al., Applied Intelligence 2023.

The model follows a deep-graph architecture analogous to a Dense Neural
Network: each layer is a RandomForest ensemble, stacked via boosting, with
each tree independently learning a distributed gradient component.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..common.utils import bootstrap_sampler, weight_solver
from ..tree.updater import TreeUpdater
from ..linear.updater import LinearUpdater
from ..objective import get_objective
from ..objective.regression import BaseObjective
from ..predictor.predictor import DeepGBoostPredictor

if TYPE_CHECKING:
    from ..callback import TrainingCallback


class DGBFModel:
    """
    Distributed Gradient Boosting Forest model.

    This is the low-level training object.  End users should use
    ``DeepGBoostRegressor`` or ``DeepGBoostClassifier`` from ``sklearn.py``
    or the ``train()`` function from ``training.py``.

    Parameters
    ----------
    n_trees : int
        Number of trees (T) per boosting layer — analogous to the width of
        a neural network layer.
    n_layers : int
        Number of boosting layers (L) — analogous to the depth.
    max_depth : int or None
        Maximum depth of each decision tree.
    learning_rate : float
        Shrinkage factor applied to the pseudo-residuals before fitting each
        layer.
    linear_projection : bool
        If True, adds a Ridge regression correction at each layer (XGBoost
        ``gblinear`` analogue) to capture linear trends.
    linear_alpha : float
        L2 regularisation for the Ridge model (only used when
        ``linear_projection=True``).
    subsample_min_frac : float
        Minimum subsample fraction at the first layer.  Grows linearly to 1.0
        at the last layer (dynamic sampling, paper sec. 3.1.3).
    weight_solver : str
        Method for solving the tree-output weights.  ``"nnls"`` uses
        non-negative least squares (fast, no equality constraint needed;
        result is renormalised to sum=1).
    objective : str or BaseObjective
        Loss function.  String aliases: ``"reg:squarederror"``,
        ``"reg:absoluteerror"``, ``"binary:logistic"``, ``"multi:softmax"``.
    random_state : int or None
        Master seed; individual trees derive their seeds from this.
    """

    def __init__(
        self,
        n_trees: int = 10,
        n_layers: int = 10,
        max_depth: int | None = None,
        learning_rate: float = 0.1,
        linear_projection: bool = False,
        linear_alpha: float = 1.0,
        subsample_min_frac: float = 0.3,
        weight_solver: str = "nnls",
        objective: str | BaseObjective = "reg:squarederror",
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.n_layers = n_layers
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.linear_projection = linear_projection
        self.linear_alpha = linear_alpha
        self.subsample_min_frac = subsample_min_frac
        self.weight_solver = weight_solver
        self.objective = objective
        self.random_state = random_state

        # Fitted state (set during fit)
        self.graph_: list[list[TreeUpdater]] = []
        self.weights_: list[list[np.ndarray]] = []
        self.linear_models_: list[LinearUpdater] = []
        self.prior_: float = 0.0
        self.feature_importances_: np.ndarray | None = None
        self.n_features_in_: int = 0
        self.evals_result_: dict = {}

        self._predictor = DeepGBoostPredictor()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _objective(self) -> BaseObjective:
        if isinstance(self.objective, str):
            return get_objective(self.objective)
        return self.objective

    def _make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        callbacks: Sequence["TrainingCallback"] | None = None,
        evals: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
    ) -> "DGBFModel":
        """
        Fit the DGBF model (Algorithm 1 from the paper).

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        callbacks : list of TrainingCallback, optional
        evals : list of (X_val, y_val, name) tuples, optional
            Validation sets for evaluation monitoring.

        Returns
        -------
        self
        """
        obj = self._objective
        rng = self._make_rng()
        n_samples, n_features = X.shape

        # Initialise state
        self.graph_ = []
        self.weights_ = []
        self.linear_models_ = []
        self.prior_ = obj.prior(y)
        self.n_features_in_ = n_features
        self.evals_result_ = {}
        feature_importance_accum = np.zeros(n_features)

        # Initialise eval result containers
        if evals:
            for _, _, name in evals:
                self.evals_result_[name] = {"train_loss": []}

        # Before-training callbacks
        callbacks = callbacks or []
        for cb in callbacks:
            cb.before_training(self)

        for l in range(self.n_layers):
            # --- Compute current predictions and pseudo-residuals --------
            # F_{l-1}(X): raw ensemble prediction up to layer l-1
            F_prev = self._predictor.predict_raw(self, X)   # (n_samples,)

            # Gradient for each tree slot (paper eq. 8):
            # g'_{l,t}(x_i) = y_i - F_{l-1}(x_i)  (distributed: same for all t
            # in this implementation — each tree learns its own component via
            # independent bootstrap subsamples and multi-output regression)
            g_global = obj.gradient(y, F_prev)              # (n_samples,)

            # Pseudo-residuals matrix: (n_samples, n_trees)
            # Each column t contains the residuals for tree slot t
            pseudo_y = np.column_stack(
                [g_global * self.learning_rate] * self.n_trees
            )

            # Before-iteration callbacks
            stop = False
            evals_log: dict = {}
            for cb in callbacks:
                if cb.before_iteration(self, l, evals_log):
                    stop = True
            if stop:
                break

            # --- Fit this layer ------------------------------------------
            new_layer, new_weights = self._fit_layer(X, pseudo_y, l, rng)
            self.graph_.append(new_layer)
            self.weights_.append(new_weights)

            # --- Optional linear projection ------------------------------
            if self.linear_projection:
                sample_idx = bootstrap_sampler(
                    n_samples, self.n_layers, l,
                    self.subsample_min_frac, rng,
                )
                lin = LinearUpdater(alpha=self.linear_alpha)
                lin.fit(X[sample_idx], pseudo_y[sample_idx].mean(axis=1))
                # Mix weight: fraction of variance explained by linear model
                lin_pred_full = lin.predict(X)
                resid_full = pseudo_y.mean(axis=1)
                var_total = np.var(resid_full) + 1e-10
                var_lin = np.var(resid_full - lin_pred_full)
                lin.alpha_mix_ = float(np.clip(1.0 - var_lin / var_total, 0.0, 1.0))
                self.linear_models_.append(lin)

            # --- Accumulate feature importances -------------------------
            for tree in new_layer:
                feature_importance_accum += tree.feature_importances_

            # --- Eval sets ----------------------------------------------
            if evals:
                for X_val, y_val, name in evals:
                    F_val = self._predictor.predict_raw(self, X_val)
                    loss = float(np.sqrt(np.mean((y_val - F_val) ** 2)))
                    self.evals_result_.setdefault(name, {}).setdefault(
                        "train_loss", []
                    ).append(loss)
                    evals_log[name] = {"train_loss": loss}

            # After-iteration callbacks
            stop = False
            for cb in callbacks:
                if cb.after_iteration(self, l, evals_log):
                    stop = True
            if stop:
                break

        # Normalise feature importances
        total = feature_importance_accum.sum()
        self.feature_importances_ = (
            feature_importance_accum / total if total > 0
            else feature_importance_accum
        )

        # After-training callbacks
        for cb in callbacks:
            cb.after_training(self)

        return self

    def _fit_layer(
        self,
        X: np.ndarray,
        pseudo_y: np.ndarray,
        layer_idx: int,
        rng: np.random.Generator,
    ) -> tuple[list[TreeUpdater], list[np.ndarray]]:
        """
        Fit all T trees for layer ``layer_idx``.

        For each tree t:
        1. Draw a dynamic bootstrap subsample.
        2. Fit a multi-output CART on the subsample.
        3. Compute optimal output weights on the full training set.

        Returns
        -------
        new_layer : list of TreeUpdater, length n_trees
        new_weights : list of np.ndarray, length n_trees
        """
        n_samples = X.shape[0]
        new_layer: list[TreeUpdater] = []
        new_weights: list[np.ndarray] = []

        for t in range(self.n_trees):
            # Dynamic bootstrap subsample (paper sec. 3.1.3)
            sample_idx = bootstrap_sampler(
                n_samples, self.n_layers, layer_idx,
                self.subsample_min_frac, rng,
            )

            # Derive per-tree seed from master rng
            tree_seed = int(rng.integers(0, 2**31))

            tree = TreeUpdater(max_depth=self.max_depth, random_state=tree_seed)
            tree.fit(X[sample_idx], pseudo_y[sample_idx])

            # Compute output weights (paper eq. 11) on FULL dataset
            tree_out_full = tree.predict(X)      # (n_samples, n_trees)
            y_target = pseudo_y.mean(axis=1)    # (n_samples,) — mean gradient

            w = weight_solver(tree_out_full, y_target, method=self.weight_solver)

            new_layer.append(tree)
            new_weights.append(w)

        return new_layer, new_weights

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Raw ensemble prediction (in the objective's transformed space).

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        raw = self._predictor.predict_raw(self, X)
        return self._objective.transform(raw)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return predictions before applying the output transform."""
        self._check_is_fitted()
        return self._predictor.predict_raw(self, X)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.graph_:
            raise RuntimeError(
                "This DGBFModel instance is not fitted yet. Call 'fit' first."
            )

    def get_params(self) -> dict:
        return {
            "n_trees": self.n_trees,
            "n_layers": self.n_layers,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "linear_projection": self.linear_projection,
            "linear_alpha": self.linear_alpha,
            "subsample_min_frac": self.subsample_min_frac,
            "weight_solver": self.weight_solver,
            "objective": self.objective,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        p = self.get_params()
        parts = ", ".join(f"{k}={v!r}" for k, v in p.items())
        return f"DGBFModel({parts})"
