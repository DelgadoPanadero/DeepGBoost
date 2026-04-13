"""
Shared utility functions for DeepGBoost.

Includes bootstrap sampling, weight solving, and numeric helpers
used across the gbm, tree, and objective modules.
"""

import numpy as np
from scipy.optimize import nnls


def bootstrap_sampler(
    n_samples: int,
    n_layers: int,
    layer_idx: int,
    subsample_min_frac: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Dynamic bootstrap sampler (paper sec. 3.1.3).

    Sample size grows linearly from ``subsample_min_frac * n_samples`` at
    layer 0 to ``n_samples`` at the last layer.  This avoids over-fitting in
    early boosting steps while allowing the final layers to see the whole
    dataset.

    Parameters
    ----------
    n_samples : int
        Total number of training samples.
    n_layers : int
        Total number of boosting layers.
    layer_idx : int
        Index of the current layer (0-based).
    subsample_min_frac : float
        Minimum fraction of samples used at layer 0.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray of shape (size,)
        Row indices to use for training.
    """
    if rng is None:
        rng = np.random.default_rng()

    min_size = max(1, int(subsample_min_frac * n_samples))
    if n_layers <= 1:
        size = n_samples
    else:
        size = int(
            min_size + (n_samples - min_size) * layer_idx / (n_layers - 1)
        )
        size = min(size, n_samples)

    return rng.choice(n_samples, size=size, replace=True)


def weight_solver(
    tree_pred: np.ndarray,
    y_real: np.ndarray,
    method: str = "nnls",
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute combination weights for the T bagged trees in a layer.

    Parameters
    ----------
    tree_pred : np.ndarray of shape (n_samples, n_trees)
        Each column is one bagged tree's prediction on the full dataset.
    y_real : np.ndarray of shape (n_samples,)
        Pseudo-residuals target (used only when ``method="nnls"``).
    method : {"nnls", "uniform"}
        ``"nnls"``    — Non-Negative Least Squares: solves
                        ``min_w ||y - tree_pred @ w||`` s.t. ``w >= 0``,
                        then normalises to ``sum(w) = 1``.  Gives each tree
                        an optimal, data-driven weight.
        ``"uniform"`` — Equal weight ``1/n_trees`` for every tree, exactly
                        as in a standard RandomForest.  Combined with
                        ``n_layers=1`` and ``learning_rate=1.0`` this makes
                        DeepGBoost mathematically equivalent to
                        RandomForest.
    sample_weight : np.ndarray of shape (n_samples,) or None
        Optional per-sample weights.  When provided, both ``tree_pred`` and
        ``y_real`` are pre-multiplied by ``sqrt(sample_weight)`` before NNLS
        so the solver minimises the Hessian-weighted residual norm.

    Returns
    -------
    np.ndarray of shape (n_trees,)
        Non-negative weights summing to 1.
    """
    n_outputs = tree_pred.shape[1]

    if method == "uniform":
        return np.full(n_outputs, 1.0 / n_outputs)

    if method == "nnls":
        if sample_weight is not None:
            sw = np.sqrt(np.clip(sample_weight, 1e-6, None))
            A = tree_pred * sw[:, np.newaxis]
            b = y_real * sw
        else:
            A, b = tree_pred, y_real
        weights, _ = nnls(A, b)
    else:
        raise ValueError(
            f"Unknown weight_solver method: '{method}'. "
            "Valid options are 'nnls' and 'uniform'."
        )

    total = weights.sum()
    if total == 0.0:
        return np.full(n_outputs, 1.0 / n_outputs)
    return weights / total


def sigmoid(
    x: np.ndarray,
) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def softmax(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Row-wise softmax with numerical stability."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
