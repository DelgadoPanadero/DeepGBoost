"""Tests for training callbacks (callback.py)."""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from deepgboost import (
    DeepGBoostRegressor,
    EarlyStopping,
    LearningRateScheduler,
    EvaluationMonitor,
    TrainingCallback,
)
from deepgboost.gbm.dgbf import DGBFModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def diabetes_split():
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0)


# ---------------------------------------------------------------------------
# TrainingCallback base
# ---------------------------------------------------------------------------

class TestTrainingCallbackBase:

    def test_default_methods_return_false(self):
        cb = TrainingCallback()
        model = object()
        assert cb.before_iteration(model, 0, {}) is False
        assert cb.after_iteration(model, 0, {}) is False

    def test_default_before_after_training_do_not_raise(self):
        cb = TrainingCallback()
        model = object()
        cb.before_training(model)
        cb.after_training(model)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_stops_when_no_improvement_unit(self):
        """Unit test: EarlyStopping returns True after patience exhausted."""
        es = EarlyStopping(patience=3, metric="train_loss", data="val", restore_best=False)

        class FakeModel:
            graph_ = []
            weights_ = []
            linear_models_ = []

        model = FakeModel()
        es.before_training(model)

        # Simulate: improvement at epoch 0, then no improvement for 3 epochs
        improving_log = {"val": {"train_loss": 1.0}}
        stale_log     = {"val": {"train_loss": 1.5}}

        assert es.after_iteration(model, 0, improving_log) is False  # improves
        assert es.after_iteration(model, 1, stale_log) is False       # wait=1
        assert es.after_iteration(model, 2, stale_log) is False       # wait=2
        assert es.after_iteration(model, 3, stale_log) is True        # wait=3 → stop

    def test_training_stops_when_callback_returns_true(self, diabetes_split):
        """Integration test: training loop stops when any callback returns True."""
        X_train, _, y_train, _ = diabetes_split
        layers_run = []

        class StopAtEpoch5(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                layers_run.append(epoch)
                return epoch >= 4  # stop after layer 5 (0-indexed)

        model = DGBFModel(n_trees=3, n_layers=20, random_state=0)
        model.fit(X_train, y_train, callbacks=[StopAtEpoch5()])

        assert len(layers_run) == 5, (
            f"Training should have stopped at layer 5, ran {len(layers_run)}"
        )

    def test_restore_best_restores_model(self, diabetes_split):
        X_train, X_val, y_train, y_val = diabetes_split
        es = EarlyStopping(patience=3, restore_best=True)
        reg = DeepGBoostRegressor(
            n_trees=3, n_layers=20, max_depth=3, learning_rate=0.05, random_state=1
        )
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[es])
        # Model should still be able to predict after restore
        preds = reg.predict(X_val)
        assert np.all(np.isfinite(preds))

    def test_no_eval_set_does_not_stop(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        es = EarlyStopping(patience=3)
        layers_run = []

        class Counter(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                layers_run.append(epoch)
                return False

        reg = DeepGBoostRegressor(n_trees=3, n_layers=5, random_state=0)
        reg.fit(X_train, y_train, callbacks=[es, Counter()])
        assert len(layers_run) == 5


# ---------------------------------------------------------------------------
# LearningRateScheduler
# ---------------------------------------------------------------------------

class TestLearningRateScheduler:

    def test_scheduler_changes_learning_rate(self, diabetes_split):
        X_train, X_val, y_train, y_val = diabetes_split

        rates_seen = []

        class LRRecorder(TrainingCallback):
            def before_iteration(self, model, epoch, evals_log):
                rates_seen.append(model.learning_rate)
                return False

        # Decay by 10% each layer
        scheduler = LearningRateScheduler(lambda epoch: 0.1 * (0.9 ** epoch))
        recorder = LRRecorder()

        model = DGBFModel(n_trees=3, n_layers=5, random_state=0)
        model.fit(X_train, y_train, callbacks=[scheduler, recorder])

        # Learning rate should decrease each layer (scheduler runs before recorder)
        assert len(rates_seen) == 5
        # After scheduler, rates are set; verify they're different
        unique_rates = set(round(r, 6) for r in rates_seen)
        assert len(unique_rates) > 1, "Learning rates should change across layers"

    def test_constant_schedule(self, diabetes_split):
        X_train, _, y_train, _ = diabetes_split
        scheduler = LearningRateScheduler(lambda epoch: 0.05)
        model = DGBFModel(n_trees=3, n_layers=3, random_state=0)
        model.fit(X_train, y_train, callbacks=[scheduler])
        assert model.learning_rate == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# EvaluationMonitor
# ---------------------------------------------------------------------------

class TestEvaluationMonitor:

    def test_monitor_prints_every_period(self, diabetes_split, capsys):
        X_train, X_val, y_train, y_val = diabetes_split
        monitor = EvaluationMonitor(period=2)
        model = DGBFModel(n_trees=3, n_layers=4, random_state=0)
        model.fit(
            X_train, y_train,
            callbacks=[monitor],
            evals=[(X_val, y_val, "val")],
        )
        captured = capsys.readouterr()
        # With 4 layers and period=2, should print at layers 2 and 4
        lines = [line for line in captured.out.strip().split("\n") if line]
        assert len(lines) == 2

    def test_monitor_silent_without_evals(self, diabetes_split, capsys):
        X_train, _, y_train, _ = diabetes_split
        monitor = EvaluationMonitor(period=1)
        model = DGBFModel(n_trees=3, n_layers=3, random_state=0)
        model.fit(X_train, y_train, callbacks=[monitor])
        captured = capsys.readouterr()
        assert captured.out == ""
