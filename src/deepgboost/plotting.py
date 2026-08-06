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
    X,
    *,
    max_features: int = 20,
    importance_type: str = "gain",
    feature_names: list[str] | None = None,
    title: str = "Feature Importance",
    xlabel: str = "Mean Contribution",
    ax=None,
    figsize: tuple[float, float] | None = None,
    positive_color: str = "#ff0051",
    negative_color: str = "#008bfb",
) -> "tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]":
    """
    Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    model : DeepGBoostRegressor, DeepGBoostClassifier, or DeepGBoostMultiClassifier
        Any fitted DeepGBoost estimator that exposes ``feature_contributions``.
    X : array-like of shape (n_samples, n_features)
        Sample matrix used to compute per-feature contributions.  A representative
        subset (e.g. the training set) is recommended.
    max_features : int
        Maximum number of features to display (sorted by importance).
    importance_type : str
        Reserved for future use; currently ignored.
    feature_names : list of str or None
        Names for each feature.  Falls back to ``f0``, ``f1``, … if not given.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.  A new figure is created if ``None``.
    figsize : tuple or None
        Figure size when creating a new figure.  If ``None``, the height is
        derived from the number of features shown (~0.35 in per bar).
    positive_color : str
        Bar colour for features with a positive mean contribution.
    negative_color : str
        Bar colour for features with a negative mean contribution.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install deepgboost[plotting]",
        ) from exc

    if not hasattr(model, "feature_contributions"):
        raise ValueError(
            "model does not expose feature_contributions(). "
            "Make sure the model is fitted.",
        )

    _, contributions = model.feature_contributions(np.asarray(X))
    mean_contributions = contributions.mean(axis=0)

    n_features = len(mean_contributions)

    if feature_names is None:
        # Try to retrieve from the model object
        if hasattr(model, "_model") and model._model is not None:
            feature_names = [f"f{i}" for i in range(n_features)]
        else:
            feature_names = [f"f{i}" for i in range(n_features)]

    # Sort by magnitude and trim, but keep the signed value for plotting
    indices = np.argsort(np.abs(mean_contributions))[-max_features:]
    selected_names = [feature_names[i] for i in indices]
    selected_scores = mean_contributions[indices]
    bar_colors = [
        positive_color if v >= 0 else negative_color for v in selected_scores
    ]

    if figsize is None:
        figsize = (8, max(2.0, 0.35 * len(indices)))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.barh(
        range(len(indices)),
        selected_scores,
        color=bar_colors,
        edgecolor="white",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(selected_names)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    return fig, ax
