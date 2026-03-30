"""
Plotting utilities for DeepGBoost (mirrors ``xgboost.plotting``).

Requires ``matplotlib`` (optional dependency: ``pip install deepgboost[plotting]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes


def plot_importance(
    model,
    *,
    max_features: int = 20,
    importance_type: str = "gain",
    feature_names: list[str] | None = None,
    title: str = "Feature Importance",
    xlabel: str = "Importance Score",
    ax=None,
    figsize: tuple[float, float] = (8, 6),
    color: str = "#3498db",
) -> "matplotlib.axes.Axes":
    """
    Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    model : DGBFModel, DeepGBoostBooster, DeepGBoostRegressor, or DeepGBoostClassifier
        Any fitted DeepGBoost object with a ``feature_importances_`` attribute.
    max_features : int
        Maximum number of features to display (sorted by importance).
    importance_type : str
        Currently only ``"gain"`` (impurity-based) is supported.
    feature_names : list of str or None
        Names for each feature.  Falls back to ``f0``, ``f1``, … if not given.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.  A new figure is created if ``None``.
    figsize : tuple
        Figure size when creating a new figure.
    color : str
        Bar colour.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install deepgboost[plotting]"
        ) from exc

    # Retrieve importances from various model types
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "_model") and model._model is not None:
        importances = model._model.feature_importances_
    else:
        raise ValueError(
            "Cannot extract feature importances from the provided model. "
            "Make sure the model is fitted."
        )

    if importances is None:
        raise ValueError("Model has not been fitted or feature importances are None.")

    n_features = len(importances)

    if feature_names is None:
        # Try to retrieve from the model object
        if hasattr(model, "_model") and model._model is not None:
            feature_names = [f"f{i}" for i in range(n_features)]
        else:
            feature_names = [f"f{i}" for i in range(n_features)]

    # Sort and trim
    indices = np.argsort(importances)[-max_features:]
    selected_names = [feature_names[i] for i in indices]
    selected_scores = importances[indices]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.barh(range(len(indices)), selected_scores, color=color, edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(selected_names)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    return ax
