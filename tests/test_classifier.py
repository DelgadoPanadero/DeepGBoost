"""Tests for DeepGBoostClassifier (sklearn.py)."""

import io
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

from deepgboost import DeepGBoostClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def multiclass_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

class TestDeepGBoostClassifierBinary:

    _CLF = DeepGBoostClassifier(
        n_trees=5, n_layers=10, max_depth=4, learning_rate=0.15, random_state=42
    )

    def test_fit_returns_self(self, binary_split):
        X_train, _, y_train, _ = binary_split
        clf = clone(self._CLF)
        result = clf.fit(X_train, y_train)
        assert result is clf

    def test_classes_attribute(self, binary_split):
        X_train, _, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2

    def test_n_classes(self, binary_split):
        X_train, _, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        assert clf.n_classes_ == 2

    def test_predict_shape(self, binary_split):
        X_train, X_test, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (X_test.shape[0],)

    def test_predict_labels_are_valid(self, binary_split):
        X_train, X_test, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert set(np.unique(preds)).issubset(set(clf.classes_))

    def test_predict_proba_shape(self, binary_split):
        X_train, X_test, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)

    def test_predict_proba_sums_to_one(self, binary_split):
        X_train, X_test, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, binary_split):
        X_train, X_test, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_accuracy_above_threshold(self, binary_split):
        X_train, X_test, y_train, y_test = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        assert acc > 0.85, f"Binary accuracy should be > 0.85, got {acc:.4f}"

    def test_feature_importances_shape(self, binary_split):
        X_train, _, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        fi = clf.feature_importances_
        assert fi.shape == (X_train.shape[1],)
        assert abs(fi.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------

class TestDeepGBoostClassifierMulticlass:

    _CLF = DeepGBoostClassifier(
        n_trees=5, n_layers=8, max_depth=3, learning_rate=0.1, random_state=0
    )

    def test_fit_returns_self(self, multiclass_split):
        X_train, _, y_train, _ = multiclass_split
        clf = clone(self._CLF)
        result = clf.fit(X_train, y_train)
        assert result is clf

    def test_classes_attribute_multiclass(self, multiclass_split):
        X_train, _, y_train, _ = multiclass_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        assert len(clf.classes_) == 3
        assert clf.n_classes_ == 3

    def test_predict_shape_multiclass(self, multiclass_split):
        X_train, X_test, y_train, _ = multiclass_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (X_test.shape[0],)

    def test_predict_labels_are_valid_multiclass(self, multiclass_split):
        X_train, X_test, y_train, _ = multiclass_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert set(np.unique(preds)).issubset(set(clf.classes_))

    def test_predict_proba_shape_multiclass(self, multiclass_split):
        X_train, X_test, y_train, _ = multiclass_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 3)

    def test_predict_proba_sums_to_one_multiclass(self, multiclass_split):
        X_train, X_test, y_train, _ = multiclass_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_accuracy_above_threshold_multiclass(self, multiclass_split):
        X_train, X_test, y_train, y_test = multiclass_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        assert acc > 0.85, f"Multiclass accuracy should be > 0.85, got {acc:.4f}"

    def test_feature_importances_multiclass(self, multiclass_split):
        X_train, _, y_train, _ = multiclass_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        fi = clf.feature_importances_
        assert fi.shape == (X_train.shape[1],)
        assert np.all(fi >= 0)


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------

class TestDeepGBoostClassifierSklearnCompat:

    def test_clone_preserves_params(self):
        clf = DeepGBoostClassifier(n_layers=10, learning_rate=0.05, random_state=7)
        clf2 = clone(clf)
        assert clf2.n_layers == 10
        assert clf2.learning_rate == 0.05
        assert clf2.random_state == 7
        assert not hasattr(clf2, "_binary_model")

    def test_get_params(self):
        clf = DeepGBoostClassifier(n_trees=8, n_layers=12)
        params = clf.get_params()
        assert params["n_trees"] == 8
        assert params["n_layers"] == 12

    def test_set_params(self):
        clf = DeepGBoostClassifier()
        clf.set_params(n_layers=15, linear_projection=True)
        assert clf.n_layers == 15
        assert clf.linear_projection is True

    def test_predict_before_fit_raises(self):
        clf = DeepGBoostClassifier()
        with pytest.raises(Exception):
            clf.predict(np.ones((5, 4)))

    def test_predict_proba_before_fit_raises(self):
        clf = DeepGBoostClassifier()
        with pytest.raises(Exception):
            clf.predict_proba(np.ones((5, 4)))

    def test_string_labels(self, multiclass_split):
        X_train, X_test, y_train, y_test = multiclass_split
        y_str_train = np.array(["class_a", "class_b", "class_c"])[y_train]
        clf = DeepGBoostClassifier(n_trees=3, n_layers=5, random_state=0)
        clf.fit(X_train, y_str_train)
        preds = clf.predict(X_test)
        assert set(np.unique(preds)).issubset({"class_a", "class_b", "class_c"})


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

@pytest.fixture
def cat_binary_numpy():
    """Binary classification dataset with one categorical column (numpy)."""
    X = np.array([
        ["red",   1.0],
        ["blue",  2.0],
        ["red",   1.5],
        ["green", 2.5],
        ["blue",  2.0],
        ["green", 1.0],
        ["red",   1.8],
        ["blue",  2.3],
    ], dtype=object)
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    return X, y


@pytest.fixture
def cat_multiclass_pandas():
    """Multiclass classification dataset as a pandas DataFrame."""
    X = pd.DataFrame({
        "color":  ["red", "blue", "green", "red", "blue", "green",
                   "red", "blue", "green", "red", "blue", "green"],
        "weight": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5,
                   1.2, 2.2, 3.2, 1.8, 2.8, 3.8],
    })
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    return X, y


class TestDeepGBoostClassifierCategorical:

    _CLF = DeepGBoostClassifier(n_trees=3, n_layers=5, random_state=0)

    def test_numpy_detects_categorical_column(self, cat_binary_numpy):
        X, y = cat_binary_numpy
        clf = clone(self._CLF)
        clf.fit(X, y)
        assert clf.categorical_columns_ == [0]
        assert clf.numerical_columns_ == [1]

    def test_pandas_detects_categorical_column(self, cat_multiclass_pandas):
        X, y = cat_multiclass_pandas
        clf = clone(self._CLF)
        clf.fit(X, y)
        assert clf.categorical_columns_ == [0]
        assert clf.numerical_columns_ == [1]

    def test_ohe_is_fitted(self, cat_binary_numpy):
        X, y = cat_binary_numpy
        clf = clone(self._CLF)
        clf.fit(X, y)
        assert clf.ohe_ is not None
        assert set(clf.ohe_.categories_[0]) == {"red", "blue", "green"}

    def test_no_encoder_when_all_numeric(self, binary_split):
        X_train, _, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        assert clf.ohe_ is None
        assert clf.categorical_columns_ == []

    def test_binary_predict_shape(self, cat_binary_numpy):
        X, y = cat_binary_numpy
        clf = clone(self._CLF)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_binary_predict_proba_valid(self, cat_binary_numpy):
        X, y = cat_binary_numpy
        clf = clone(self._CLF)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass_predict_shape(self, cat_multiclass_pandas):
        X, y = cat_multiclass_pandas
        clf = clone(self._CLF)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_multiclass_predict_proba_valid(self, cat_multiclass_pandas):
        X, y = cat_multiclass_pandas
        clf = clone(self._CLF)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_pickle_preserves_encoder(self, cat_binary_numpy):
        X, y = cat_binary_numpy
        clf = clone(self._CLF)
        clf.fit(X, y)

        buf = io.BytesIO()
        pickle.dump(clf, buf)
        buf.seek(0)
        clf_loaded = pickle.load(buf)

        assert clf_loaded.categorical_columns_ == clf.categorical_columns_
        assert clf_loaded.numerical_columns_ == clf.numerical_columns_
        assert clf_loaded.ohe_ is not None
        np.testing.assert_array_equal(
            clf_loaded.ohe_.categories_[0], clf.ohe_.categories_[0]
        )

    def test_pickle_identical_predictions_binary(self, cat_binary_numpy):
        X, y = cat_binary_numpy
        clf = clone(self._CLF)
        clf.fit(X, y)
        preds_before = clf.predict(X)
        proba_before = clf.predict_proba(X)

        buf = io.BytesIO()
        pickle.dump(clf, buf)
        buf.seek(0)
        clf_loaded = pickle.load(buf)

        np.testing.assert_array_equal(preds_before, clf_loaded.predict(X))
        np.testing.assert_array_equal(proba_before, clf_loaded.predict_proba(X))

    def test_pickle_identical_predictions_multiclass(self, cat_multiclass_pandas):
        X, y = cat_multiclass_pandas
        clf = clone(self._CLF)
        clf.fit(X, y)
        preds_before = clf.predict(X)
        proba_before = clf.predict_proba(X)

        buf = io.BytesIO()
        pickle.dump(clf, buf)
        buf.seek(0)
        clf_loaded = pickle.load(buf)

        np.testing.assert_array_equal(preds_before, clf_loaded.predict(X))
        np.testing.assert_array_equal(proba_before, clf_loaded.predict_proba(X))

    def test_pickle_no_encoder_identical_predictions(self, binary_split):
        X_train, X_test, y_train, _ = binary_split
        clf = clone(self._CLF)
        clf.fit(X_train, y_train)
        preds_before = clf.predict(X_test)

        buf = io.BytesIO()
        pickle.dump(clf, buf)
        buf.seek(0)
        clf_loaded = pickle.load(buf)

        np.testing.assert_array_equal(preds_before, clf_loaded.predict(X_test))
