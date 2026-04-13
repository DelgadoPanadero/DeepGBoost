"""Tests for adaptive layer-width scheduling via NNLS condition number.

Covers:
- _layer_cond_numbers_ is always populated after fit (length == n_layers).
- _layer_n_trees_ is populated when cond_threshold is set.
- Adaptive halving logic: n_trees is reduced when cond exceeds threshold.
- Both DGBFModel (regressor objective) and DGBFModel (binary classifier
  objective) variants.

Tree budget: n_layers * n_trees <= 100 in every test.
Data: n_samples <= 200, n_features <= 10.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from deepgboost.dgbf.dgbf import DGBFModel


# ---------------------------------------------------------------------------
# Shared small fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reg_data():
    """Regression dataset: 150 samples, 8 features."""
    X, y = make_regression(n_samples=150, n_features=8, noise=0.1, random_state=0)
    return X, y.astype(np.float64)


@pytest.fixture(scope="module")
def clf_data():
    """Binary classification dataset: 150 samples, 8 features."""
    X, y = make_classification(
        n_samples=150, n_features=8, n_informative=4, random_state=0
    )
    return X, y.astype(np.float64)


# ---------------------------------------------------------------------------
# Helper: build a DGBFModel with small budget
# ---------------------------------------------------------------------------

_BASE_PARAMS = dict(
    n_trees=5,
    n_layers=4,
    max_depth=3,
    random_state=42,
    objective="reg:squarederror",
)  # budget: 5 * 4 = 20 <= 100


def _make_model(**overrides) -> DGBFModel:
    params = {**_BASE_PARAMS, **overrides}
    return DGBFModel(**params)


# ---------------------------------------------------------------------------
# 1. _layer_cond_numbers_ always populated (cond_threshold=None)
# ---------------------------------------------------------------------------


class TestCondNumbersDiagnostic:
    """_layer_cond_numbers_ must be populated for every layer, always."""

    @pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
    def test_length_equals_n_layers(self, reg_data, clf_data, objective):
        X, y = reg_data if objective == "reg:squarederror" else clf_data
        model = _make_model(objective=objective)
        model.fit(X, y)
        assert len(model._layer_cond_numbers_) == model.n_layers

    def test_all_values_are_positive_floats(self, reg_data):
        X, y = reg_data
        model = _make_model()
        model.fit(X, y)
        for cond in model._layer_cond_numbers_:
            assert isinstance(cond, float)
            assert cond > 0.0

    def test_populated_with_cond_threshold_none(self, reg_data):
        """Explicit cond_threshold=None still populates the diagnostic."""
        X, y = reg_data
        model = _make_model(cond_threshold=None)
        model.fit(X, y)
        assert len(model._layer_cond_numbers_) == model.n_layers

    def test_populated_for_binary_classifier(self, clf_data):
        X, y = clf_data
        model = _make_model(objective="binary:logistic")
        model.fit(X, y)
        assert len(model._layer_cond_numbers_) == model.n_layers


# ---------------------------------------------------------------------------
# 2. _layer_n_trees_ populated when cond_threshold is set
# ---------------------------------------------------------------------------


class TestLayerNTreesDiagnostic:
    """_layer_n_trees_ must be populated (length == n_layers) when adaptation
    is enabled."""

    def test_length_equals_n_layers_regression(self, reg_data):
        X, y = reg_data
        model = _make_model(cond_threshold=1e4)
        model.fit(X, y)
        assert len(model._layer_n_trees_) == model.n_layers

    def test_length_equals_n_layers_classifier(self, clf_data):
        X, y = clf_data
        model = _make_model(objective="binary:logistic", cond_threshold=1e4)
        model.fit(X, y)
        assert len(model._layer_n_trees_) == model.n_layers

    def test_first_layer_n_trees_equals_n_trees(self, reg_data):
        """The first layer always uses the configured n_trees."""
        X, y = reg_data
        model = _make_model(cond_threshold=1e4)
        model.fit(X, y)
        assert model._layer_n_trees_[0] == model.n_trees

    def test_values_are_positive_ints(self, reg_data):
        X, y = reg_data
        model = _make_model(cond_threshold=1e4)
        model.fit(X, y)
        for t in model._layer_n_trees_:
            assert isinstance(t, int)
            assert t >= 1


# ---------------------------------------------------------------------------
# 3. Adaptive halving: when threshold is very low, n_trees is reduced
# ---------------------------------------------------------------------------


class TestAdaptiveHalving:
    """With a very low cond_threshold, at least one layer must have reduced
    n_trees (because almost any real-data predictor matrix will be
    ill-conditioned relative to threshold=1.0).

    Tree budget accounting: even if every layer halves, min n_trees per
    layer is 1, so total <= n_layers * n_trees (configured) <= 100.
    """

    def test_halving_occurs_with_very_low_threshold(self, reg_data):
        """cond_threshold=1.0 causes halving from layer 0 → 1.

        The first layer uses n_trees=8 (multi-column matrix, cond > 1.0).
        Because the threshold is exceeded at layer 0, the second layer must
        use max(1, 8 // 2) = 4 trees.  We cannot assert cond > 1.0 for all
        layers because when n_trees is halved all the way to 1 the single-
        column predictor matrix has cond=1.0 exactly.
        """
        X, y = reg_data
        model = _make_model(n_trees=8, n_layers=4, cond_threshold=1.0)
        # budget: 8 * 4 = 32 <= 100
        model.fit(X, y)
        # The first layer always uses the configured n_trees=8
        assert model._layer_n_trees_[0] == 8
        # The first-layer condition number must exceed 1.0 (multi-column matrix)
        assert model._layer_cond_numbers_[0] > 1.0
        # The second layer must be halved from the first
        assert model._layer_n_trees_[1] == max(1, model._layer_n_trees_[0] // 2)

    def test_n_trees_never_below_one(self, reg_data):
        """Adaptive width must not go below 1 regardless of threshold."""
        X, y = reg_data
        model = _make_model(n_trees=2, n_layers=5, cond_threshold=1.0)
        # budget: 2 * 5 = 10 <= 100
        model.fit(X, y)
        assert all(t >= 1 for t in model._layer_n_trees_)

    def test_halving_not_triggered_with_high_threshold(self, reg_data):
        """With a very high threshold, no halving should occur."""
        X, y = reg_data
        model = _make_model(n_trees=5, n_layers=4, cond_threshold=1e18)
        model.fit(X, y)
        # No layer's cond should exceed 1e18 on small data
        assert all(t == model.n_trees for t in model._layer_n_trees_)

    def test_recovered_width_when_cond_drops(self, reg_data):
        """After a high-cond layer reduces width, if cond drops on the next
        layer, width resets to n_trees for the following layer.

        We verify the invariant: when _layer_n_trees_[i] < n_trees, the
        entry at [i+1] (if it exists) is either n_trees (recovered) or
        max(1, _layer_n_trees_[i] // 2) (further halved) — never some other
        arbitrary value.
        """
        X, y = reg_data
        model = _make_model(n_trees=6, n_layers=5, cond_threshold=1e4)
        # budget: 6 * 5 = 30 <= 100
        model.fit(X, y)
        n_trees_config = model.n_trees
        cond_thresh = model.cond_threshold
        for i, (cond, actual) in enumerate(
            zip(model._layer_cond_numbers_, model._layer_n_trees_)
        ):
            if i + 1 < len(model._layer_n_trees_):
                next_actual = model._layer_n_trees_[i + 1]
                if cond > cond_thresh:
                    assert next_actual == max(1, actual // 2)
                else:
                    assert next_actual == n_trees_config


