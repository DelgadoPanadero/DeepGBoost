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

        1. Initialise per-tree accumulators at ``model.prior_``.
        2. For each layer l and tree t:
           a. Obtain T-dimensional tree output.
           b. Apply learned weights to collapse to a scalar per sample.
           c. Accumulate into tree-slot t.
        3. If ``linear_projection`` is active, add the linear correction.
        4. Return the mean over all T accumulators as the ensemble output.

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
        n_trees = model.n_trees

        # Per-tree accumulator initialised at the prior (F_0 = mean(y))
        accum = np.full((n_samples, n_trees), model.prior_, dtype=np.float64)

        for l, layer in enumerate(model.graph_):
            layer_weights = model.weights_[l]

            for t, tree in enumerate(layer):
                # tree.predict returns (n_samples, n_trees)
                tree_out = tree.predict(X)  # (n_samples, n_trees)
                w = layer_weights[t]  # (n_trees,)
                # Weighted combination → scalar per sample
                tree_scalar = (tree_out * w).sum(axis=1)  # (n_samples,)
                accum[:, t] += tree_scalar

            if model.linear_projection and model.linear_models_:
                lin = model.linear_models_[l]
                lin_correction = lin.predict(X)  # (n_samples,)
                # Broadcast linear correction to all tree slots
                accum += lin_correction[:, np.newaxis] * lin.alpha_mix_

        return accum.mean(axis=1)

    def predict_raw_per_slot(
        self,
        model,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return the full (n_samples, n_trees) accumulator without averaging.

        Useful for inspecting per-slot contributions.
        """
        n_samples = X.shape[0]
        n_trees = model.n_trees
        accum = np.full((n_samples, n_trees), model.prior_, dtype=np.float64)

        for l, layer in enumerate(model.graph_):
            layer_weights = model.weights_[l]
            for t, tree in enumerate(layer):
                tree_out = tree.predict(X)
                w = layer_weights[t]
                accum[:, t] += (tree_out * w).sum(axis=1)

            if model.linear_projection and model.linear_models_:
                lin = model.linear_models_[l]
                accum += lin.predict(X)[:, np.newaxis] * lin.alpha_mix_

        return accum
