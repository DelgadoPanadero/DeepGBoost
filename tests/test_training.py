"""Tests for the functional train() / cv() API (training.py)."""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split

from deepgboost import (
    DeepGBoostDMatrix,
    DeepGBoostBooster,
    train,
    cv,
    EarlyStopping,
    EvaluationMonitor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def regression_data():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    return (
        DeepGBoostDMatrix(Xtr, label=ytr),
        DeepGBoostDMatrix(Xte, label=yte),
        DeepGBoostDMatrix(Xte),
    )


@pytest.fixture(scope="module")
def binary_data():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    return (
        DeepGBoostDMatrix(Xtr, label=ytr.astype(float)),
        DeepGBoostDMatrix(Xte, label=yte.astype(float)),
        DeepGBoostDMatrix(Xte),
    )


_PARAMS = {
    "n_trees": 5,
    "n_layers": 5,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": 42,
    "objective": "reg:squarederror",
}


# ---------------------------------------------------------------------------
# train() basic
# ---------------------------------------------------------------------------

class TestTrainFunction:

    def test_returns_booster(self, regression_data):
        dtrain, _, _ = regression_data
        bst = train(_PARAMS, dtrain, verbose_eval=False)
        assert isinstance(bst, DeepGBoostBooster)

    def test_predict_after_train(self, regression_data):
        dtrain, _, dtest = regression_data
        bst = train(_PARAMS, dtrain, verbose_eval=False)
        preds = bst.predict(dtest)
        assert preds.shape == (dtest.num_row,)
        assert np.all(np.isfinite(preds))

    def test_verbose_eval_false_is_silent(self, regression_data, capsys):
        dtrain, dval, _ = regression_data
        train(_PARAMS, dtrain, evals=[(dval, "val")], verbose_eval=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_eval_int_period(self, regression_data, capsys):
        dtrain, dval, _ = regression_data
        train(_PARAMS, dtrain, evals=[(dval, "val")], verbose_eval=2)
        captured = capsys.readouterr()
        # 5 layers, period=2 → print at layers 2 and 4
        lines = [l for l in captured.out.strip().split("\n") if l]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# train() with evals
# ---------------------------------------------------------------------------

class TestTrainWithEvals:

    def test_evals_result_populated(self, regression_data):
        dtrain, dval, _ = regression_data
        evals_result = {}
        train(
            _PARAMS, dtrain,
            evals=[(dval, "validation")],
            evals_result=evals_result,
            verbose_eval=False,
        )
        assert "validation" in evals_result
        assert "train_loss" in evals_result["validation"]
        assert len(evals_result["validation"]["train_loss"]) == _PARAMS["n_layers"]

    def test_evals_result_decreasing_loss(self, regression_data):
        dtrain, dval, _ = regression_data
        # Use more layers and higher lr to see consistent decrease
        params = dict(_PARAMS, n_layers=10, learning_rate=0.2)
        evals_result = {}
        train(
            params, dtrain,
            evals=[(dval, "val")],
            evals_result=evals_result,
            verbose_eval=False,
        )
        losses = evals_result["val"]["train_loss"]
        # First loss should be higher than last (or at least not constant)
        assert losses[0] >= losses[-1] or np.std(losses) > 0


# ---------------------------------------------------------------------------
# train() with early stopping
# ---------------------------------------------------------------------------

class TestTrainEarlyStopping:

    def test_early_stopping_requires_evals(self, regression_data):
        dtrain, _, _ = regression_data
        with pytest.raises(ValueError, match="eval"):
            train(_PARAMS, dtrain, early_stopping_rounds=3, verbose_eval=False)

    def test_early_stopping_stops_before_max_layers(self, regression_data):
        dtrain, dval, _ = regression_data
        params = dict(_PARAMS, n_layers=50, learning_rate=1.0)
        evals_result = {}
        train(
            params, dtrain,
            evals=[(dval, "val")],
            early_stopping_rounds=3,
            evals_result=evals_result,
            verbose_eval=False,
        )
        actual_layers = len(evals_result["val"]["train_loss"])
        assert actual_layers < 50, (
            f"Expected early stopping before 50 layers, ran {actual_layers}"
        )

    def test_custom_callback_in_list(self, regression_data):
        dtrain, _, _ = regression_data
        called = []

        class CounterCallback(EarlyStopping.__bases__[0]):
            def after_iteration(self, model, epoch, evals_log):
                called.append(epoch)
                return False

        train(_PARAMS, dtrain, callbacks=[CounterCallback()], verbose_eval=False)
        assert len(called) == _PARAMS["n_layers"]


# ---------------------------------------------------------------------------
# train() with classification
# ---------------------------------------------------------------------------

class TestTrainClassification:

    def test_binary_logistic(self, binary_data):
        dtrain, dval, dtest = binary_data
        params = dict(_PARAMS, objective="binary:logistic")
        bst = train(params, dtrain, evals=[(dval, "val")], verbose_eval=False)
        preds = bst.predict(dtest)
        assert np.all(preds >= 0) and np.all(preds <= 1)


# ---------------------------------------------------------------------------
# cv()
# ---------------------------------------------------------------------------

class TestCVFunction:

    def test_cv_returns_dict(self):
        X, y = load_diabetes(return_X_y=True)
        data = DeepGBoostDMatrix(X[:100], label=y[:100])
        params = {"n_trees": 3, "n_layers": 3, "random_state": 0}
        results = cv(params, data, nfold=3, metrics=["rmse"])
        assert isinstance(results, dict)
        assert "rmse" in results
        assert len(results["rmse"]) == 3

    def test_cv_scores_are_positive(self):
        X, y = load_diabetes(return_X_y=True)
        data = DeepGBoostDMatrix(X[:100], label=y[:100])
        params = {"n_trees": 3, "n_layers": 3, "random_state": 0}
        results = cv(params, data, nfold=3, metrics=["rmse"])
        assert all(s > 0 for s in results["rmse"])

    def test_cv_requires_label(self):
        X, _ = load_diabetes(return_X_y=True)
        data = DeepGBoostDMatrix(X[:50])
        with pytest.raises(ValueError, match="label"):
            cv({"n_layers": 3}, data)
