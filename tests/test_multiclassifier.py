"""Tests for DeepGBoostMultiClassifier."""

import io
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deepgboost import DeepGBoostMultiClassifier, DeepGBoostClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multiclass_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def binary_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# Multiclass (3-class iris)
# ---------------------------------------------------------------------------


class TestDeepGBoostMultiClassifierMulticlass:
    _CLF = DeepGBoostMultiClassifier(
        n_trees=5, n_layers=10, max_depth=4, random_state=42
    )

    def test_fit_returns_self(self, multiclass_split):
        X_train, _, y_train, _ = multiclass_split
        clf = clone(self._CLF)
        assert clf.fit(X_train, y_train) is clf

    def test_classes_set(self, multiclass_split):
        X_train, _, y_train, _ = multiclass_split
        clf = clone(self._CLF).fit(X_train, y_train)
        assert clf.n_classes_ == 3
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_predict_proba_shape_and_sum(self, multiclass_split):
        X_train, X_test, y_train, _ = multiclass_split
        clf = clone(self._CLF).fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_labels_in_classes(self, multiclass_split):
        X_train, X_test, y_train, _ = multiclass_split
        clf = clone(self._CLF).fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert set(preds).issubset(set(clf.classes_))

    def test_accuracy_threshold(self, multiclass_split):
        X_train, X_test, y_train, y_test = multiclass_split
        clf = clone(self._CLF).fit(X_train, y_train)
        assert clf.score(X_test, y_test) > 0.85

    def test_internal_state_shapes(self, multiclass_split):
        X_train, _, y_train, _ = multiclass_split
        clf = clone(self._CLF).fit(X_train, y_train)
        K = clf.n_classes_
        T = clf.n_trees
        # prior shape
        assert clf.model_.prior_.shape == (K,)
        # weights shape per layer
        for w in clf.model_.weights_:
            assert w.shape == (K, T)

    def test_trees_are_multi_output(self, multiclass_split):
        X_train, _, y_train, _ = multiclass_split
        clf = clone(self._CLF).fit(X_train, y_train)
        first_tree = clf.model_.graph_[0][0]
        # sklearn's n_outputs_ == K confirms multi-output training
        assert first_tree._tree.n_outputs_ == clf.n_classes_

    def test_feature_importances(self, multiclass_split):
        X_train, _, y_train, _ = multiclass_split
        clf = clone(self._CLF).fit(X_train, y_train)
        fi = clf.feature_importances_
        assert fi.shape == (X_train.shape[1],)
        np.testing.assert_allclose(fi.sum(), 1.0, atol=1e-6)

    def test_pickle_roundtrip(self, multiclass_split):
        X_train, X_test, y_train, _ = multiclass_split
        clf = clone(self._CLF).fit(X_train, y_train)
        buf = io.BytesIO()
        pickle.dump(clf, buf)
        buf.seek(0)
        clf2 = pickle.load(buf)
        np.testing.assert_array_equal(clf.predict(X_test), clf2.predict(X_test))

    def test_clone_and_get_set_params(self):
        clf = self._CLF
        params = clf.get_params()
        assert params["n_trees"] == 5
        cloned = clone(clf)
        assert cloned.get_params() == params

    def test_predict_proba_raises_before_fit(self):
        clf = DeepGBoostMultiClassifier()
        X = np.random.rand(10, 4)
        with pytest.raises(Exception):
            clf.predict_proba(X)

    def test_pipeline_compatible(self, multiclass_split):
        X_train, X_test, y_train, _ = multiclass_split
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DeepGBoostMultiClassifier(n_trees=3, n_layers=5, random_state=0)),
        ])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)

    def test_competitive_with_ovr(self, multiclass_split):
        """Multi-output accuracy should be within 5 pp of OvR on iris."""
        X_train, X_test, y_train, y_test = multiclass_split
        clf_ovr = DeepGBoostClassifier(
            n_trees=5, n_layers=10, max_depth=4, random_state=42
        ).fit(X_train, y_train)
        clf_multi = clone(self._CLF).fit(X_train, y_train)
        assert clf_multi.score(X_test, y_test) >= clf_ovr.score(X_test, y_test) - 0.05


# ---------------------------------------------------------------------------
# Binary case (K=2 edge case via one-hot expansion)
# ---------------------------------------------------------------------------


class TestDeepGBoostMultiClassifierBinary:
    _CLF = DeepGBoostMultiClassifier(
        n_trees=5, n_layers=5, max_depth=4, random_state=42
    )

    def test_binary_predict_proba_shape(self, binary_split):
        X_train, X_test, y_train, _ = binary_split
        clf = clone(self._CLF).fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_binary_internal_state(self, binary_split):
        X_train, _, y_train, _ = binary_split
        clf = clone(self._CLF).fit(X_train, y_train)
        assert clf.model_.prior_.shape == (2,)
        for w in clf.model_.weights_:
            assert w.shape == (2, clf.n_trees)

    def test_binary_trees_are_multi_output(self, binary_split):
        X_train, _, y_train, _ = binary_split
        clf = clone(self._CLF).fit(X_train, y_train)
        assert clf.model_.graph_[0][0]._tree.n_outputs_ == 2
