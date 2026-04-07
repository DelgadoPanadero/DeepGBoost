"""
Distributed Gradient Boosting Forest (DGBF) — core training loop.

Implements Algorithm 1 from the paper:
    "A generalized decision tree ensemble based on the NeuralNetworks
     architecture: Distributed Gradient Boosting Forest (DGBF)"
     Delgado-Panadero et al., Applied Intelligence 2023.

The model follows a deep-graph architecture analogous to a Dense Neural
Network: each layer is a bagged ensemble of trees (analogous to
RandomForest), stacked via boosting.  With ``n_layers=1`` and
``max_features="sqrt"`` the model reduces to a standard RandomForest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..common.utils import bootstrap_sampler, weight_solver
from ..tree.updater import TreeUpdater
from ..linear.updater import LinearUpdater
from ..objective import get_objective
from ..objective.regression import BaseObjective
from ..predictor.predictor import DeepGBoostPredictor

if TYPE_CHECKING:
    from ..callbacks.base_callback import TrainingCallback


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
    max_features : int, float, str or None
        Number of features considered at each split within a tree.  Mirrors
        ``sklearn``'s ``DecisionTreeRegressor.max_features``.  Defaults to
        ``None`` (all features), preserving the original DGBF behaviour.
        Set to ``"sqrt"`` for the standard Random Forest feature subsampling;
        combined with ``n_layers=1`` the model becomes analogous to a
        RandomForest.
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
        Method for combining the T bagged trees in each layer.
        ``"nnls"`` uses Non-Negative Least Squares to find optimal weights
        (result is renormalised to sum=1).  ``"uniform"`` assigns equal
        weight ``1/n_trees`` to every tree, exactly as in RandomForest.
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
        max_features: int | float | str | None = None,
        learning_rate: float = 0.1,
        linear_projection: bool = False,
        linear_alpha: float = 1.0,
        subsample_min_frac: float = 0.3,
        weight_solver: str = "nnls",
        hessian_reg: float = 0.0,
        objective: str | BaseObjective = "reg:squarederror",
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.n_layers = n_layers
        self.max_depth = max_depth
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.linear_projection = linear_projection
        self.linear_alpha = linear_alpha
        self.subsample_min_frac = subsample_min_frac
        self.weight_solver = weight_solver
        self.hessian_reg = hessian_reg
        self.objective = objective
        self.random_state = random_state

        # Fitted state (set during fit)
        self.graph_: list[list[TreeUpdater]] = []
        # weights_[l] is a 1-D array of shape (n_trees,): the combination
        # weights for the T bagged trees in layer l.
        self.weights_: list[np.ndarray] = []
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

        for layer_idx in range(self.n_layers):
            # --- Compute current predictions and pseudo-residuals --------
            # F_{l-1}(X): raw ensemble prediction up to layer l-1
            F_prev = self._predictor.predict_raw(self, X)  # (n_samples,)

            # Gradient for each tree slot (paper eq. 8):
            # g'_{l,t}(x_i) = y_i - F_{l-1}(x_i)  (distributed: same for all t
            # in this implementation — each tree learns its own component via
            # independent bootstrap subsamples and multi-output regression)
            g_global = obj.gradient(y, F_prev)  # (n_samples,)
            h_global = obj.hessian(y, F_prev)   # (n_samples,)

            # Newton step: g/(h + reg) scales the update correctly for
            # objectives whose Hessian varies with F (e.g. logistic).
            # ``hessian_reg`` (λ) mirrors XGBoost's L2 leaf regularisation:
            # it prevents extreme Newton steps when h is small (e.g. near the
            # prior for imbalanced datasets) and improves convergence stability.
            # For MSE (h=1 everywhere) this term has minimal effect.
            pseudo_y = (g_global / np.maximum(h_global + self.hessian_reg, 1e-7)) * self.learning_rate  # (n_samples,)

            # Before-iteration callbacks
            stop = False
            evals_log: dict = {}
            for cb in callbacks:
                if cb.before_iteration(self, layer_idx, evals_log):
                    stop = True
            if stop:
                break

            # --- Fit this layer ------------------------------------------
            new_layer, new_weights = self._fit_layer(
                X, pseudo_y, layer_idx, rng, h_global
            )
            self.graph_.append(new_layer)
            self.weights_.append(new_weights)

            # --- Optional linear projection ------------------------------
            if self.linear_projection:
                sample_idx = bootstrap_sampler(
                    n_samples,
                    self.n_layers,
                    layer_idx,
                    self.subsample_min_frac,
                    rng,
                )
                lin = LinearUpdater(alpha=self.linear_alpha)
                lin.fit(X[sample_idx], pseudo_y[sample_idx])
                # Mix weight: fraction of variance explained by linear model
                lin_pred_full = lin.predict(X)
                var_total = np.var(pseudo_y) + 1e-10
                var_lin = np.var(pseudo_y - lin_pred_full)
                lin.alpha_mix_ = float(
                    np.clip(1.0 - var_lin / var_total, 0.0, 1.0)
                )
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
                if cb.after_iteration(self, layer_idx, evals_log):
                    stop = True
            if stop:
                break

        # Normalise feature importances
        total = feature_importance_accum.sum()
        self.feature_importances_ = (
            feature_importance_accum / total
            if total > 0
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
        hessian: np.ndarray | None = None,
    ) -> tuple[list[TreeUpdater], np.ndarray]:
        """
        Fit all T trees for layer ``layer_idx`` using bagging.

        Each tree is trained independently on:
        * a dynamic bootstrap row-subsample (paper sec. 3.1.3), and
        * a random feature subset governed by ``max_features``.

        When ``hessian`` is provided, each tree is fit with per-sample
        weights equal to the Hessian values.  This mirrors XGBoost's
        behaviour: splits focus on uncertain samples (h_i > 0) and
        down-weight already-confident predictions (h_i ≈ 0).

        After fitting, a single weight vector of length T is computed via
        NNLS on the full dataset to optimally combine the T tree predictions.

        Parameters
        ----------
        X : (n_samples, n_features)
        pseudo_y : (n_samples,)
            Shrunk pseudo-residuals for this layer.
        layer_idx : int
        rng : np.random.Generator
        hessian : (n_samples,) or None
            Per-sample Hessian values used as tree fitting sample weights.

        Returns
        -------
        new_layer : list of TreeUpdater, length n_trees
        layer_weights : np.ndarray of shape (n_trees,)
            One combination weight per bagged tree.
        """
        n_samples = X.shape[0]
        new_layer: list[TreeUpdater] = []
        tree_preds: list[np.ndarray] = []

        for t in range(self.n_trees):
            # Dynamic bootstrap subsample (paper sec. 3.1.3)
            sample_idx = bootstrap_sampler(
                n_samples=n_samples,
                n_layers=self.n_layers,
                layer_idx=layer_idx,
                subsample_min_frac=self.subsample_min_frac,
                rng=rng,
            )

            # Derive per-tree seed from master rng
            tree_seed = int(rng.integers(0, 2**31))

            tree = TreeUpdater(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            sw = hessian[sample_idx] if hessian is not None else None
            tree.fit(X[sample_idx], pseudo_y[sample_idx], sample_weight=sw)

            new_layer.append(tree)
            tree_preds.append(tree.predict(X)[:, 0])  # (n_samples,)

        # Stack per-tree predictions: (n_samples, n_trees)
        all_preds = np.column_stack(tree_preds)

        # Single weight vector for the layer (paper eq. 11)
        layer_weights = weight_solver(
            all_preds,
            pseudo_y,
            method=self.weight_solver,
        )

        return new_layer, layer_weights

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
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

    def predict_raw(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
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
            "max_features": self.max_features,
            "learning_rate": self.learning_rate,
            "linear_projection": self.linear_projection,
            "linear_alpha": self.linear_alpha,
            "subsample_min_frac": self.subsample_min_frac,
            "weight_solver": self.weight_solver,
            "hessian_reg": self.hessian_reg,
            "objective": self.objective,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        p = self.get_params()
        parts = ", ".join(f"{k}={v!r}" for k, v in p.items())
        return f"DGBFModel({parts})"
