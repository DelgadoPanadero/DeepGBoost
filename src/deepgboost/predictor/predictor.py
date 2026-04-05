"""
Inference module for DeepGBoost.

Decouples the forward pass from the training loop, mirroring XGBoost's
``predictor`` module.  ``DeepGBoostPredictor`` knows only about the trained
``graph_``, ``weights_``, and optional ``linear_models_`` — it does not
touch any training state.
"""

from __future__ import annotations

import numpy as np


class DeepGBoostPredictor:
    """
    Stateless inference engine for a trained DeepGBoost model.

    All state is read from the ``model`` object passed into each method, so
    a single predictor instance can be reused across multiple trained models.
    """

    def predict_raw(
        self,
        model,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute raw (untransformed) ensemble predictions.

        Implements the forward pass of Algorithm 1 (paper sec. 3):

        1. Initialise a scalar accumulator at ``model.prior_``.
        2. For each layer l:
           a. Collect the T bagged trees' 1-D predictions.
           b. Apply the layer's learned weight vector to combine them.
           c. Add the weighted sum to the accumulator.
        3. If ``linear_projection`` is active, add the linear correction.

        Parameters
        ----------
        model : DGBFModel
            A fitted model with ``graph_``, ``weights_``, ``prior_``,
            and optionally ``linear_models_``.
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        n_samples = X.shape[0]

        # Scalar accumulator initialised at the prior (F_0 = mean(y))
        accum = np.full(n_samples, model.prior_, dtype=np.float64)

        for layer_idx, layer in enumerate(model.graph_):
            # layer_weights: (n_trees,) — one weight per bagged tree
            layer_weights = model.weights_[layer_idx]

            # Stack each tree's 1-D prediction → (n_samples, n_trees)
            tree_preds = np.column_stack(
                [tree.predict(X)[:, 0] for tree in layer]
            )
            # Weighted sum across the T bagged trees → (n_samples,)
            accum += (tree_preds * layer_weights).sum(axis=1)

            if model.linear_projection and model.linear_models_:
                lin = model.linear_models_[layer_idx]
                accum += lin.predict(X) * lin.alpha_mix_

        return accum
