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
        size = int(min_size + (n_samples - min_size) * layer_idx / (n_layers - 1))
        size = min(size, n_samples)

    return rng.choice(n_samples, size=size, replace=True)


def weight_solver(
    tree_pred: np.ndarray,
    y_real: np.ndarray,
    method: str = "nnls",
) -> np.ndarray:
    """
    Compute optimal combination weights for tree outputs (paper eq. 10–11).

    Solves ``min_w ||y - tree_pred @ w||`` subject to ``w >= 0``, then
    normalises so ``sum(w) = 1``.  Falls back to uniform weights when
    the solver produces a zero vector.

    Parameters
    ----------
    tree_pred : np.ndarray of shape (n_samples, n_outputs)
        Predicted values for each tree output.
    y_real : np.ndarray of shape (n_samples,)
        Target values (mean pseudo-residuals).
    method : {"nnls"}
        Solver method.  Only "nnls" is currently supported.

    Returns
    -------
    np.ndarray of shape (n_outputs,)
        Non-negative weights summing to 1.
    """
    n_outputs = tree_pred.shape[1]

    if method == "nnls":
        weights, _ = nnls(tree_pred, y_real)
    else:
        raise ValueError(f"Unknown weight_solver method: '{method}'")

    total = weights.sum()
    if total == 0.0:
        return np.full(n_outputs, 1.0 / n_outputs)

    return weights / total


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Row-wise softmax with numerical stability."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
