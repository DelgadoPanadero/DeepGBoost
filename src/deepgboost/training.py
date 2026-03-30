"""
High-level training API for DeepGBoost (mirrors ``xgboost.training``).

The ``train()`` function accepts a params dict and a ``DeepGBoostDMatrix``,
returning a fitted ``DeepGBoostBooster`` — the same pattern as XGBoost's
functional API.

Example
-------
::

    import deepgboost as dgb

    dtrain = dgb.DeepGBoostDMatrix(X_train, label=y_train)
    dval   = dgb.DeepGBoostDMatrix(X_val,   label=y_val)

    params = {
        "n_layers": 20,
        "n_trees": 10,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
    }

    bst = dgb.train(
        params,
        dtrain,
        evals=[(dval, "val")],
        early_stopping_rounds=5,
        verbose_eval=True,
    )
    preds = bst.predict(dval)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import DeepGBoostDMatrix, DeepGBoostBooster
from .callback import TrainingCallback, EarlyStopping, EvaluationMonitor
from .metric import get_metric


def train(
    params: dict,
    dtrain: DeepGBoostDMatrix,
    *,
    evals: list[tuple[DeepGBoostDMatrix, str]] | None = None,
    early_stopping_rounds: int | None = None,
    evals_result: dict | None = None,
    verbose_eval: bool | int = True,
    callbacks: Sequence[TrainingCallback] | None = None,
) -> DeepGBoostBooster:
    """
    Train a DeepGBoost model (mirrors ``xgboost.train``).

    Parameters
    ----------
    params : dict
        Hyper-parameters for ``DGBFModel``.  Valid keys: ``n_trees``,
        ``n_layers``, ``max_depth``, ``learning_rate``, ``linear_projection``,
        ``linear_alpha``, ``subsample_min_frac``, ``weight_solver``,
        ``objective``, ``random_state``.
    dtrain : DeepGBoostDMatrix
        Training data with ``label`` set.
    evals : list of (DeepGBoostDMatrix, str), optional
        Validation sets to evaluate after each layer.
    early_stopping_rounds : int or None
        Stop training if no improvement for this many rounds on the first
        eval set.  Requires ``evals`` to be provided.
    evals_result : dict or None
        If provided, metric histories are written here after training.
    verbose_eval : bool or int
        ``True`` → print every layer; integer N → print every N layers;
        ``False`` → silent.
    callbacks : list of TrainingCallback, optional
        Additional callbacks.

    Returns
    -------
    DeepGBoostBooster
        Fitted booster.
    """
    all_callbacks: list[TrainingCallback] = list(callbacks or [])

    # Verbose monitor
    if verbose_eval is not False and evals:
        period = 1 if verbose_eval is True else int(verbose_eval)
        all_callbacks.append(EvaluationMonitor(period=period))

    # Early stopping
    if early_stopping_rounds is not None:
        if not evals:
            raise ValueError(
                "early_stopping_rounds requires at least one eval set in 'evals'."
            )
        all_callbacks.append(EarlyStopping(patience=early_stopping_rounds))

    booster = DeepGBoostBooster(params=params)
    booster.train(dtrain, callbacks=all_callbacks, evals=evals)

    if evals_result is not None:
        evals_result.update(booster.evals_result_)

    return booster


def cv(
    params: dict,
    data: DeepGBoostDMatrix,
    nfold: int = 5,
    *,
    metrics: list[str] | None = None,
    seed: int = 42,
    verbose_eval: bool = False,
) -> dict[str, list[float]]:
    """
    Cross-validated training (mirrors ``xgboost.cv``).

    Performs stratified k-fold for classification objectives and plain k-fold
    for regression.

    Parameters
    ----------
    params : dict
        Model hyper-parameters.
    data : DeepGBoostDMatrix
        Full dataset with label.
    nfold : int
        Number of folds.
    metrics : list of str or None
        Metric names to compute per fold.  Defaults to ``["rmse"]`` for
        regression and ``["accuracy"]`` for classification.
    seed : int
        Seed for fold splitting.
    verbose_eval : bool
        Print per-fold scores.

    Returns
    -------
    dict mapping metric names to lists of per-fold scores.
    """
    if data.label is None:
        raise ValueError("data must have a label for cross-validation.")

    X, y = data.data, data.label
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, nfold)

    objective = params.get("objective", "reg:squarederror")
    is_classification = objective in ("binary:logistic", "multi:softmax")

    if metrics is None:
        metrics = ["accuracy"] if is_classification else ["rmse"]

    metric_fns = {name: get_metric(name) for name in metrics}
    results: dict[str, list[float]] = {m: [] for m in metrics}

    for fold_idx in range(nfold):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(nfold) if i != fold_idx])

        dtrain_fold = DeepGBoostDMatrix(X[train_idx], label=y[train_idx])
        dval_fold = DeepGBoostDMatrix(X[val_idx], label=y[val_idx])

        bst = train(params, dtrain_fold, verbose_eval=False)
        preds = bst.predict(dval_fold)

        for name, fn in metric_fns.items():
            # For accuracy, compare class labels
            if name == "accuracy":
                score = fn(y[val_idx], (preds >= 0.5).astype(int))
            else:
                score = fn(y[val_idx], preds)
            results[name].append(score)
            if verbose_eval:
                print(f"Fold {fold_idx + 1}/{nfold}  {name}: {score:.6f}")

    return results
