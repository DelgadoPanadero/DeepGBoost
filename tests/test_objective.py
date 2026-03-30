"""Tests for objective functions (objective/)."""

import numpy as np
import pytest

from deepgboost.objective.regression import RMSEObjective, MAEObjective
from deepgboost.objective.classification import LogisticObjective, SoftmaxObjective
from deepgboost.objective import get_objective
from deepgboost.common.utils import sigmoid


class TestRMSEObjective:

    def test_gradient_shape(self):
        obj = RMSEObjective()
        y = np.array([1.0, 2.0, 3.0])
        F = np.array([1.5, 1.5, 3.5])
        g = obj.gradient(y, F)
        assert g.shape == y.shape

    def test_gradient_values(self):
        obj = RMSEObjective()
        y = np.array([1.0, 2.0, 3.0])
        F = np.array([0.0, 0.0, 0.0])
        g = obj.gradient(y, F)
        np.testing.assert_array_almost_equal(g, y)

    def test_prior_is_mean(self):
        obj = RMSEObjective()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert obj.prior(y) == pytest.approx(2.5)

    def test_transform_is_identity(self):
        obj = RMSEObjective()
        raw = np.array([1.0, -1.0, 0.0])
        np.testing.assert_array_equal(obj.transform(raw), raw)


class TestMAEObjective:

    def test_gradient_shape(self):
        obj = MAEObjective()
        y = np.array([1.0, 2.0, 3.0])
        F = np.array([1.5, 1.5, 3.5])
        g = obj.gradient(y, F)
        assert g.shape == y.shape

    def test_gradient_values_are_signs(self):
        obj = MAEObjective()
        y = np.array([1.0, 2.0, 3.0])
        F = np.array([2.0, 1.0, 3.0])
        g = obj.gradient(y, F)
        expected = np.array([-1.0, 1.0, 0.0])
        np.testing.assert_array_equal(g, expected)

    def test_prior_is_median(self):
        obj = MAEObjective()
        y = np.array([1.0, 3.0, 2.0, 10.0])
        assert obj.prior(y) == pytest.approx(float(np.median(y)))


class TestLogisticObjective:

    def test_gradient_shape(self):
        obj = LogisticObjective()
        y = np.array([0.0, 1.0, 1.0, 0.0])
        F = np.zeros(4)
        g = obj.gradient(y, F)
        assert g.shape == y.shape

    def test_gradient_at_zero_logits(self):
        obj = LogisticObjective()
        y = np.array([1.0])
        F = np.array([0.0])
        g = obj.gradient(y, F)
        # sigmoid(0) = 0.5, so gradient = 1 - 0.5 = 0.5
        assert g[0] == pytest.approx(0.5)

    def test_gradient_range(self):
        obj = LogisticObjective()
        y = np.array([0.0, 1.0])
        F = np.array([100.0, -100.0])
        g = obj.gradient(y, F)
        # Both should be close to -1 (wrong direction)
        assert np.all(g >= -1.0) and np.all(g <= 1.0)

    def test_prior_is_log_odds(self):
        obj = LogisticObjective()
        y = np.array([1.0, 1.0, 0.0, 0.0])
        prior = obj.prior(y)
        # p = 0.5 → log(0.5 / 0.5) = 0
        assert prior == pytest.approx(0.0, abs=1e-6)

    def test_transform_applies_sigmoid(self):
        obj = LogisticObjective()
        raw = np.array([0.0, 1.0, -1.0])
        transformed = obj.transform(raw)
        expected = sigmoid(raw)
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_transform_in_range(self):
        obj = LogisticObjective()
        raw = np.linspace(-10, 10, 50)
        transformed = obj.transform(raw)
        assert np.all(transformed >= 0) and np.all(transformed <= 1)


class TestSoftmaxObjective:

    def test_gradient_shape(self):
        obj = SoftmaxObjective()
        y = np.eye(3)[[0, 1, 2, 0]]     # one-hot, (4, 3)
        F = np.zeros((4, 3))
        g = obj.gradient(y, F)
        assert g.shape == y.shape

    def test_gradient_values_at_uniform_logits(self):
        obj = SoftmaxObjective()
        y = np.eye(3)[[0]]              # one-hot for class 0, (1, 3)
        F = np.zeros((1, 3))
        g = obj.gradient(y, F)
        # softmax([0,0,0]) = [1/3, 1/3, 1/3]
        # gradient[0] = 1 - 1/3 = 2/3; gradient[1,2] = 0 - 1/3 = -1/3
        np.testing.assert_array_almost_equal(
            g, np.array([[2 / 3, -1 / 3, -1 / 3]])
        )

    def test_prior_raises_for_1d(self):
        obj = SoftmaxObjective()
        with pytest.raises(ValueError, match="one-hot"):
            obj.prior(np.array([0, 1, 2]))

    def test_1d_input_raises(self):
        obj = SoftmaxObjective()
        with pytest.raises(ValueError):
            obj.prior(np.array([0, 1, 2]))


class TestGetObjective:

    def test_known_objectives(self):
        for name in ["reg:squarederror", "reg:absoluteerror",
                     "binary:logistic", "multi:softmax"]:
            obj = get_objective(name)
            assert obj is not None

    def test_unknown_objective_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            get_objective("unknown:objective")
