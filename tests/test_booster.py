"""Tests for DeepGBoostDMatrix and DeepGBoostBooster (core.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from deepgboost import DeepGBoostDMatrix, DeepGBoostBooster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def diabetes():
    X, y = load_diabetes(return_X_y=True)
    return X, y


@pytest.fixture(scope="module")
def dtrain_diabetes(diabetes):
    X, y = diabetes
    return DeepGBoostDMatrix(X[:300], label=y[:300])


@pytest.fixture(scope="module")
def dtest_diabetes(diabetes):
    X, _ = diabetes
    return DeepGBoostDMatrix(X[300:])


# ---------------------------------------------------------------------------
# DeepGBoostDMatrix tests
# ---------------------------------------------------------------------------

class TestDeepGBoostDMatrix:

    def test_from_numpy_array(self, diabetes):
        X, y = diabetes
        dm = DeepGBoostDMatrix(X, label=y)
        assert dm.num_row == X.shape[0]
        assert dm.num_col == X.shape[1]
        assert dm.label is not None
        assert dm.label.shape == (X.shape[0],)

    def test_from_dataframe(self, diabetes):
        X, y = diabetes
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        dm = DeepGBoostDMatrix(df, label=y)
        assert dm.feature_names == [f"feat_{i}" for i in range(X.shape[1])]
        assert dm.num_col == X.shape[1]

    def test_without_label(self, diabetes):
        X, _ = diabetes
        dm = DeepGBoostDMatrix(X)
        assert dm.label is None

    def test_auto_feature_names(self, diabetes):
        X, _ = diabetes
        dm = DeepGBoostDMatrix(X)
        assert dm.feature_names == [f"f{i}" for i in range(X.shape[1])]

    def test_label_shape_mismatch_raises(self, diabetes):
        X, y = diabetes
        with pytest.raises(ValueError, match="same number of rows"):
            DeepGBoostDMatrix(X, label=y[:-5])

    def test_1d_data_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            DeepGBoostDMatrix(np.ones(10))

    def test_repr(self, diabetes):
        X, y = diabetes
        dm = DeepGBoostDMatrix(X, label=y)
        assert "DeepGBoostDMatrix" in repr(dm)
        assert "shape=" in repr(dm)


# ---------------------------------------------------------------------------
# DeepGBoostBooster tests
# ---------------------------------------------------------------------------

class TestDeepGBoostBooster:

    _PARAMS = {
        "n_trees": 5,
        "n_layers": 5,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 42,
    }

    def test_train_and_predict(self, dtrain_diabetes, dtest_diabetes):
        bst = DeepGBoostBooster(params=self._PARAMS)
        bst.train(dtrain_diabetes)
        preds = bst.predict(dtest_diabetes)
        assert preds.shape == (dtest_diabetes.num_row,)
        assert np.all(np.isfinite(preds))

    def test_predict_before_train_raises(self, dtest_diabetes):
        bst = DeepGBoostBooster()
        with pytest.raises(RuntimeError, match="not trained"):
            bst.predict(dtest_diabetes)

    def test_train_without_label_raises(self, dtest_diabetes):
        bst = DeepGBoostBooster(params=self._PARAMS)
        with pytest.raises(ValueError, match="label"):
            bst.train(dtest_diabetes)

    def test_feature_importances_shape(self, dtrain_diabetes):
        bst = DeepGBoostBooster(params=self._PARAMS)
        bst.train(dtrain_diabetes)
        fi = bst.feature_importances_
        assert fi is not None
        assert fi.shape == (dtrain_diabetes.num_col,)
        assert abs(fi.sum() - 1.0) < 1e-6

    def test_save_and_load_model(self, dtrain_diabetes, dtest_diabetes):
        bst = DeepGBoostBooster(params=self._PARAMS)
        bst.train(dtrain_diabetes)
        preds_before = bst.predict(dtest_diabetes)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            bst.save_model(path)
            bst2 = DeepGBoostBooster()
            bst2.load_model(path)
            preds_after = bst2.predict(dtest_diabetes)
            np.testing.assert_array_almost_equal(preds_before, preds_after)
        finally:
            path.unlink(missing_ok=True)

    def test_get_score_returns_dict(self, dtrain_diabetes):
        bst = DeepGBoostBooster(params=self._PARAMS)
        bst.train(dtrain_diabetes)
        score = bst.get_score()
        assert isinstance(score, dict)
        assert all(v >= 0 for v in score.values())

    def test_repr(self):
        bst = DeepGBoostBooster(params={"n_layers": 5})
        assert "DeepGBoostBooster" in repr(bst)
        assert "trained=False" in repr(bst)
