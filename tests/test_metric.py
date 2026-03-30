"""Tests for evaluation metrics (metric/)."""

import numpy as np
import pytest

from deepgboost.metric.regression import RMSEMetric, MAEMetric, R2ScoreMetric
from deepgboost.metric.classification import AccuracyMetric, LogLossMetric, AUCMetric
from deepgboost.metric import get_metric


class TestRMSEMetric:

    def test_zero_error(self):
        metric = RMSEMetric()
        y = np.array([1.0, 2.0, 3.0])
        assert metric(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        metric = RMSEMetric()
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        assert metric(y_true, y_pred) == pytest.approx(1.0)

    def test_non_negative(self):
        metric = RMSEMetric()
        y = np.random.default_rng(0).standard_normal(100)
        yp = np.random.default_rng(1).standard_normal(100)
        assert metric(y, yp) >= 0

    def test_higher_is_better_false(self):
        assert RMSEMetric.higher_is_better is False


class TestMAEMetric:

    def test_zero_error(self):
        metric = MAEMetric()
        y = np.array([1.0, 2.0, 3.0])
        assert metric(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        metric = MAEMetric()
        y_true = np.array([0.0, 2.0])
        y_pred = np.array([1.0, 0.0])
        assert metric(y_true, y_pred) == pytest.approx(1.5)

    def test_higher_is_better_false(self):
        assert MAEMetric.higher_is_better is False


class TestR2ScoreMetric:

    def test_perfect_prediction(self):
        metric = R2ScoreMetric()
        y = np.array([1.0, 2.0, 3.0])
        assert metric(y, y) == pytest.approx(1.0)

    def test_mean_prediction_is_zero(self):
        metric = R2ScoreMetric()
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.full_like(y, y.mean())
        assert metric(y, y_pred) == pytest.approx(0.0)

    def test_negative_for_bad_predictions(self):
        metric = R2ScoreMetric()
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 1.0, 0.0])
        assert metric(y, y_pred) < 0

    def test_constant_target(self):
        metric = R2ScoreMetric()
        y = np.ones(5)
        assert metric(y, y) == pytest.approx(1.0)

    def test_higher_is_better_true(self):
        assert R2ScoreMetric.higher_is_better is True


class TestAccuracyMetric:

    def test_perfect_accuracy(self):
        metric = AccuracyMetric()
        y = np.array([0, 1, 1, 0])
        assert metric(y, y) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        metric = AccuracyMetric()
        y = np.array([0, 0, 1, 1])
        yp = np.array([1, 1, 0, 0])
        assert metric(y, yp) == pytest.approx(0.0)

    def test_half_accuracy(self):
        metric = AccuracyMetric()
        y = np.array([0, 1, 0, 1])
        yp = np.array([0, 0, 1, 1])
        assert metric(y, yp) == pytest.approx(0.5)

    def test_higher_is_better_true(self):
        assert AccuracyMetric.higher_is_better is True


class TestLogLossMetric:

    def test_perfect_predictions(self):
        metric = LogLossMetric()
        y = np.array([0.0, 1.0])
        # Clip avoids log(0); near-perfect probs
        p = np.array([1e-7, 1 - 1e-7])
        assert metric(y, p) < 0.01

    def test_uniform_predictions(self):
        metric = LogLossMetric()
        y = np.array([0.0, 1.0])
        p = np.array([0.5, 0.5])
        assert metric(y, p) == pytest.approx(-np.log(0.5), rel=1e-4)

    def test_non_negative(self):
        metric = LogLossMetric()
        y = np.array([0.0, 1.0, 1.0])
        p = np.array([0.3, 0.7, 0.9])
        assert metric(y, p) >= 0

    def test_higher_is_better_false(self):
        assert LogLossMetric.higher_is_better is False


class TestAUCMetric:

    def test_perfect_auc(self):
        metric = AUCMetric()
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        assert metric(y, p) == pytest.approx(1.0)

    def test_random_auc(self):
        metric = AUCMetric()
        y = np.array([0, 1, 0, 1])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        assert 0.0 <= metric(y, p) <= 1.0

    def test_auc_all_same_class(self):
        metric = AUCMetric()
        y = np.ones(5)
        p = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        assert metric(y, p) == pytest.approx(0.5)

    def test_higher_is_better_true(self):
        assert AUCMetric.higher_is_better is True


class TestGetMetric:

    def test_known_metrics(self):
        for name in ["rmse", "mae", "r2", "accuracy", "logloss", "auc"]:
            m = get_metric(name)
            assert callable(m)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("f1_score")
