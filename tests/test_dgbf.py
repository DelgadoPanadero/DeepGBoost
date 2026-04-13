"""Tests for DGBFModel — core algorithm behaviour.

This module tests the internals of the DGBF training loop and prediction
formula.  Tests here operate directly on ``DGBFModel``, ``TreeUpdater``,
and ``DeepGBoostPredictor`` rather than through the sklearn wrapper.

Interface-level behaviour (fit/predict/score/clone/GridSearchCV/…) is
covered in ``test_regressor.py`` and ``test_classifier.py``.
"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from deepgboost import DeepGBoostRegressor
from deepgboost.dgbf.dgbf import DGBFModel
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
        assert rf_model.model_.prior_ == pytest.approx(
            y_train.mean(), rel=1e-12
        )

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
