"""Tests for DeepGBoostRegressor (sklearn.py)."""

import io
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from deepgboost import DeepGBoostRegressor
from deepgboost.gbm.dgbf import DGBFModel
from deepgboost.predictor.predictor import DeepGBoostPredictor
from deepgboost.tree.updater import TreeUpdater


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diabetes_split():
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestDeepGBoostRegressorBasic:
    _REG = DeepGBoostRegressor(
        n_trees=5,
        n_layers=5,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )

    def test_fit_returns_self(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        reg = clone(self._REG)
        result = reg.fit(X_train, y_train)
        assert result is reg

    def test_predict_shape(self, diabetes_split):
        X_train, X_test, y_train, _ = diabetes_split
        reg = clone(self._REG)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        assert preds.shape == (X_test.shape[0],)

    def test_predictions_are_finite(self, diabetes_split):
        X_train, X_test, y_train, _ = diabetes_split
        reg = clone(self._REG)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        assert np.all(np.isfinite(preds))

    def test_r2_score_positive(self, diabetes_split):
        X_train, X_test, y_train, y_test = diabetes_split
        reg = clone(self._REG)
        reg.fit(X_train, y_train)
        r2 = reg.score(X_test, y_test)
        assert r2 > 0.3, f"R² should be > 0.3, got {r2:.4f}"

    def test_predict_before_fit_raises(self, diabetes_split):
        _, X_test, _, _ = diabetes_split
        reg = DeepGBoostRegressor()
        with pytest.raises(Exception):
            reg.predict(X_test)


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


class TestDeepGBoostRegressorFeatureImportances:
    def test_importances_shape(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        reg = DeepGBoostRegressor(n_trees=3, n_layers=3, random_state=0)
        reg.fit(X_train, y_train)
        fi = reg.feature_importances_
        assert fi.shape == (X_train.shape[1],)

    def test_importances_sum_to_one(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        reg = DeepGBoostRegressor(n_trees=3, n_layers=3, random_state=0)
        reg.fit(X_train, y_train)
        assert abs(reg.feature_importances_.sum() - 1.0) < 1e-6

    def test_importances_non_negative(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        reg = DeepGBoostRegressor(n_trees=3, n_layers=3, random_state=0)
        reg.fit(X_train, y_train)
        assert np.all(reg.feature_importances_ >= 0)


# ---------------------------------------------------------------------------
# Linear projection
# ---------------------------------------------------------------------------


class TestDeepGBoostRegressorLinearProjection:
    def test_linear_projection_fits_and_predicts(self, diabetes_split):
        X_train, X_test, y_train, _ = diabetes_split
        reg = DeepGBoostRegressor(
            n_trees=5,
            n_layers=5,
            linear_projection=True,
            linear_alpha=1.0,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        assert preds.shape == (X_test.shape[0],)
        assert np.all(np.isfinite(preds))

    def test_linear_projection_has_linear_models(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        reg = DeepGBoostRegressor(
            n_trees=3, n_layers=3, linear_projection=True, random_state=0
        )
        reg.fit(X_train, y_train)
        assert len(reg.model_.linear_models_) == 3

    def test_linear_projection_false_has_no_linear_models(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        reg = DeepGBoostRegressor(
            n_trees=3, n_layers=3, linear_projection=False, random_state=0
        )
        reg.fit(X_train, y_train)
        assert len(reg.model_.linear_models_) == 0


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------


class TestDeepGBoostRegressorSklearnCompat:
    def test_clone(self):
        reg = DeepGBoostRegressor(n_layers=7, learning_rate=0.05)
        reg2 = clone(reg)
        assert reg2.n_layers == 7
        assert reg2.learning_rate == 0.05
        assert not hasattr(reg2, "model_")

    def test_get_params(self):
        reg = DeepGBoostRegressor(n_layers=15, n_trees=8)
        params = reg.get_params()
        assert params["n_layers"] == 15
        assert params["n_trees"] == 8

    def test_set_params(self):
        reg = DeepGBoostRegressor()
        reg.set_params(n_layers=20, learning_rate=0.05)
        assert reg.n_layers == 20
        assert reg.learning_rate == 0.05

    def test_eval_set_and_evals_result(self, diabetes_split):
        X_train, X_test, y_train, y_test = diabetes_split
        reg = DeepGBoostRegressor(n_trees=3, n_layers=5, random_state=0)
        reg.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        result = reg.evals_result_
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


@pytest.fixture
def cat_data_numpy():
    """Small regression dataset with one categorical column (numpy object array)."""
    X = np.array(
        [
            [1.0, "red", 10.0],
            [2.0, "blue", 20.0],
            [3.0, "red", 30.0],
            [4.0, "green", 40.0],
            [5.0, "blue", 50.0],
            [6.0, "red", 60.0],
        ],
        dtype=object,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return X, y


@pytest.fixture
def cat_data_pandas():
    """Small regression dataset as a pandas DataFrame with one string column."""
    X = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50],
            "city": [
                "Madrid",
                "Barcelona",
                "Madrid",
                "Sevilla",
                "Barcelona",
                "Sevilla",
            ],
            "income": [30000, 45000, 50000, 60000, 55000, 70000],
        }
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return X, y


class TestDeepGBoostRegressorCategorical:
    _REG = DeepGBoostRegressor(n_trees=3, n_layers=5, random_state=0)

    def test_numpy_detects_categorical_column(self, cat_data_numpy):
        X, y = cat_data_numpy
        reg = clone(self._REG)
        reg.fit(X, y)
        assert reg.categorical_columns_ == [1]
        assert reg.numerical_columns_ == [0, 2]

    def test_pandas_detects_categorical_column(self, cat_data_pandas):
        X, y = cat_data_pandas
        reg = clone(self._REG)
        reg.fit(X, y)
        assert reg.categorical_columns_ == [1]
        assert reg.numerical_columns_ == [0, 2]

    def test_ohe_is_fitted(self, cat_data_numpy):
        X, y = cat_data_numpy
        reg = clone(self._REG)
        reg.fit(X, y)
        assert reg.ohe_ is not None
        assert len(reg.ohe_.categories_) == 1
        assert set(reg.ohe_.categories_[0]) == {"red", "blue", "green"}

    def test_no_encoder_when_all_numeric(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        reg = clone(self._REG)
        reg.fit(X_train, y_train)
        assert reg.ohe_ is None
        assert reg.categorical_columns_ == []

    def test_predictions_finite(self, cat_data_numpy):
        X, y = cat_data_numpy
        reg = clone(self._REG)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))

    def test_eval_set_with_categorical(self, cat_data_numpy):
        X, y = cat_data_numpy
        reg = clone(self._REG)
        reg.fit(X, y, eval_set=[(X, y)])
        assert isinstance(reg.evals_result_, dict)

    def test_pickle_preserves_encoder(self, cat_data_numpy):
        X, y = cat_data_numpy
        reg = clone(self._REG)
        reg.fit(X, y)

        buf = io.BytesIO()
        pickle.dump(reg, buf)
        buf.seek(0)
        reg_loaded = pickle.load(buf)

        assert reg_loaded.categorical_columns_ == reg.categorical_columns_
        assert reg_loaded.numerical_columns_ == reg.numerical_columns_
        assert reg_loaded.ohe_ is not None
        np.testing.assert_array_equal(
            reg_loaded.ohe_.categories_[0], reg.ohe_.categories_[0]
        )

    def test_pickle_identical_predictions(self, cat_data_numpy):
        X, y = cat_data_numpy
        reg = clone(self._REG)
        reg.fit(X, y)
        preds_before = reg.predict(X)

        buf = io.BytesIO()
        pickle.dump(reg, buf)
        buf.seek(0)
        reg_loaded = pickle.load(buf)

        preds_after = reg_loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_pickle_no_encoder_identical_predictions(self, diabetes_split):
        X_train, X_test, y_train, _ = diabetes_split
        reg = clone(self._REG)
        reg.fit(X_train, y_train)
        preds_before = reg.predict(X_test)

        buf = io.BytesIO()
        pickle.dump(reg, buf)
        buf.seek(0)
        reg_loaded = pickle.load(buf)

        preds_after = reg_loaded.predict(X_test)
        np.testing.assert_array_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# RandomForest equivalence
# ---------------------------------------------------------------------------


class TestRandomForestEquivalence:
    """
    With n_layers=1, weight_solver="uniform" and learning_rate=1.0,
    DeepGBoost must implement the exact RandomForest prediction formula:

        prediction(X) = prior + (1/T) * Σ_t  tree_t(X)

    where each tree_t was fit on a bootstrap sample of the pseudo-residuals
    (y - prior).  This is mathematically equivalent to plain RF averaging
    because CART is translation-equivariant: leaf values shift by ``prior``
    when the target shifts by ``prior``, so
    ``prior + tree_on_(y-prior)(X) == tree_on_y(X)``.

    Two levels of verification:

    1. **Formula test** — verified from DGB's own fitted components using the
       diabetes dataset.

    2. **sklearn RF injection test** — sklearn's RandomForestRegressor trees
       are injected directly into DGB's predictor using centered targets
       (mean=0) so that DGB and RF train on the *identical* targets.
       Predictions must therefore be bit-for-bit identical.
    """

    N_TREES = 20
    MAX_FEATURES = "sqrt"
    RANDOM_STATE = 42

    # -- Fixtures -------------------------------------------------------------

    @pytest.fixture(scope="class")
    def rf_model(self, diabetes_split):
        """Fitted DeepGBoost in RF mode (n_layers=1, uniform weights)."""
        X_train, _, y_train, _ = diabetes_split
        reg = DeepGBoostRegressor(
            n_trees=self.N_TREES,
            n_layers=1,
            max_features=self.MAX_FEATURES,
            learning_rate=1.0,
            weight_solver="uniform",
            random_state=self.RANDOM_STATE,
        )
        reg.fit(X_train, y_train)
        return reg

    @pytest.fixture(scope="class")
    def centered_data(self):
        """Synthetic dataset with mean(y_train) exactly 0.

        When prior = 0, DeepGBoost trains its trees on ``y - 0 = y``, the
        same targets that sklearn's RF uses.  Injecting RF's own fitted trees
        into DGB's predictor must then give bit-for-bit identical results.
        """
        rng = np.random.default_rng(0)
        X_train = rng.random((200, 8))
        X_test = rng.random((50, 8))
        y_raw = rng.standard_normal(200) * 50
        y_train = y_raw - y_raw.mean()
        return X_train, X_test, y_train

    # -- Tests ----------------------------------------------------------------

    def test_prior_equals_training_mean(self, rf_model, diabetes_split):
        """prior_ must equal mean(y_train), as in a RandomForest."""
        _, _, y_train, _ = diabetes_split
        assert rf_model.model_.prior_ == pytest.approx(y_train.mean(), rel=1e-12)

    def test_weights_are_uniform(self, rf_model):
        """Layer weights must be exactly 1/T for every tree (RF averaging)."""
        layer_weights = rf_model.model_.weights_[0]
        expected = np.full(self.N_TREES, 1.0 / self.N_TREES)
        np.testing.assert_allclose(layer_weights, expected, rtol=1e-12)

    def test_exact_rf_formula(self, rf_model, diabetes_split):
        """
        model.predict(X) must equal prior + (1/T) * Σ_t tree_t.predict(X),
        verified directly from DGB's own fitted components.
        """
        _, X_test, _, _ = diabetes_split
        model = rf_model.model_

        tree_preds = np.column_stack(
            [t.predict(X_test)[:, 0] for t in model.graph_[0]]
        )
        expected = model.prior_ + tree_preds.mean(axis=1)

        np.testing.assert_allclose(
            rf_model.predict(X_test),
            expected,
            rtol=1e-12,
            err_msg=(
                "DeepGBoost(n_layers=1, weight_solver='uniform', lr=1.0) "
                "must implement the RandomForest prediction formula exactly."
            ),
        )

    def test_sklearn_rf_injection(self, centered_data):
        """
        Inject sklearn's RandomForestRegressor trees into DGB's predictor.

        With prior=0 (centred y), DGB and RF train trees on the identical
        targets, so injecting RF's fitted trees into DGB's prediction formula
        must reproduce RF's own output exactly.

        This is the strongest test: it uses sklearn's RF *object* as the
        ground truth and verifies DGB's formula against it directly.
        """
        X_train, X_test, y_train = centered_data

        rf = RandomForestRegressor(
            n_estimators=self.N_TREES,
            max_features=self.MAX_FEATURES,
            random_state=self.RANDOM_STATE,
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # Build a minimal DGB model shell and inject RF's trees.
        # Setting prior=0 matches the centred-data assumption so DGB adds
        # nothing on top of the plain tree average.
        model = DGBFModel.__new__(DGBFModel)
        model.prior_ = 0.0
        model.n_trees = self.N_TREES
        model.n_layers = 1
        model.linear_projection = False
        model.linear_models_ = []
        model._predictor = DeepGBoostPredictor()

        # Wrap each sklearn DecisionTreeRegressor in a TreeUpdater shell so
        # DGB's predictor can call tree.predict(X)[:, 0] on it.
        wrapped = []
        for sklearn_tree in rf.estimators_:
            tu = TreeUpdater.__new__(TreeUpdater)
            tu._tree = sklearn_tree
            wrapped.append(tu)

        model.graph_ = [wrapped]
        model.weights_ = [np.full(self.N_TREES, 1.0 / self.N_TREES)]

        dgb_pred = model._predictor.predict_raw(model, X_test)

        np.testing.assert_allclose(
            dgb_pred,
            rf_pred,
            rtol=1e-10,
            err_msg=(
                "DGB predictor with injected sklearn RF trees must reproduce "
                "RandomForestRegressor.predict() exactly."
            ),
        )
