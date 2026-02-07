"""
Tests for predicting_unpredictable.metrics module.
"""

import numpy as np
import pytest

from predicting_unpredictable.metrics import (
    accuracy_from_logits_numpy,
    mae_numpy,
    macro_f1_numpy,
)


class TestMaeNumpy:
    """Tests for mae_numpy function."""

    def test_mae_identical_arrays(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert mae_numpy(arr, arr) == 0.0

    def test_mae_simple_case(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([2.0, 3.0, 4.0])
        # MAE = mean(|1-2|, |2-3|, |3-4|) = mean(1, 1, 1) = 1.0
        assert mae_numpy(pred, target) == 1.0

    def test_mae_with_negative_values(self):
        pred = np.array([-1.0, 0.0, 1.0])
        target = np.array([1.0, 0.0, -1.0])
        # MAE = mean(|-1-1|, |0-0|, |1-(-1)|) = mean(2, 0, 2) = 4/3
        expected = 4.0 / 3.0
        assert abs(mae_numpy(pred, target) - expected) < 1e-6

    def test_mae_2d_arrays(self):
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        target = np.array([[2.0, 2.0], [3.0, 5.0]])
        # MAE = mean(1, 0, 0, 1) = 0.5
        assert mae_numpy(pred, target) == 0.5

    def test_mae_returns_float(self):
        pred = np.array([1.0])
        target = np.array([2.0])
        result = mae_numpy(pred, target)
        assert isinstance(result, float)

    def test_mae_shape_mismatch_raises(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="shape mismatch"):
            mae_numpy(pred, target)

    def test_mae_different_ndim_raises(self):
        pred = np.array([[1.0, 2.0]])
        target = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="shape mismatch"):
            mae_numpy(pred, target)

    def test_mae_large_array(self):
        np.random.seed(42)
        pred = np.random.randn(100, 100)
        target = np.random.randn(100, 100)
        result = mae_numpy(pred, target)
        assert result > 0
        assert isinstance(result, float)


class TestAccuracyFromLogitsNumpy:
    """Tests for accuracy_from_logits_numpy function."""

    def test_accuracy_all_correct(self):
        # Logits where argmax matches targets
        logits = np.array([
            [10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 10.0]
        ])
        targets = np.array([0, 1, 2])
        assert accuracy_from_logits_numpy(logits, targets) == 1.0

    def test_accuracy_all_wrong(self):
        logits = np.array([
            [10.0, 1.0, 1.0], [10.0, 1.0, 1.0], [10.0, 1.0, 1.0]
        ])
        targets = np.array([1, 2, 2])
        assert accuracy_from_logits_numpy(logits, targets) == 0.0

    def test_accuracy_half_correct(self):
        logits = np.array([[10.0, 1.0], [1.0, 10.0], [10.0, 1.0], [1.0, 10.0]])
        targets = np.array([0, 0, 0, 0])
        # Correct: idx 0 and 2, Wrong: idx 1 and 3
        assert accuracy_from_logits_numpy(logits, targets) == 0.5

    def test_accuracy_returns_float(self):
        logits = np.array([[1.0, 0.0]])
        targets = np.array([0])
        result = accuracy_from_logits_numpy(logits, targets)
        assert isinstance(result, float)

    def test_accuracy_logits_wrong_ndim_raises(self):
        logits = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D
        targets = np.array([0])
        with pytest.raises(ValueError, match="logits must be 2D"):
            accuracy_from_logits_numpy(logits, targets)

    def test_accuracy_targets_wrong_ndim_raises(self):
        logits = np.array([[1.0, 2.0]])
        targets = np.array([[0]])  # 2D instead of 1D
        with pytest.raises(ValueError, match="targets must be 1D"):
            accuracy_from_logits_numpy(logits, targets)

    def test_accuracy_single_sample(self):
        logits = np.array([[0.1, 0.9]])
        targets = np.array([1])
        assert accuracy_from_logits_numpy(logits, targets) == 1.0

    def test_accuracy_with_negative_logits(self):
        logits = np.array([[-1.0, -2.0, -0.5]])  # argmax = 2 (-0.5 is largest)
        targets = np.array([2])
        assert accuracy_from_logits_numpy(logits, targets) == 1.0


class TestMacroF1Numpy:
    """Tests for macro_f1_numpy function."""

    def test_macro_f1_perfect_predictions(self):
        targets = np.array([0, 1, 2, 0, 1, 2])
        preds = np.array([0, 1, 2, 0, 1, 2])
        assert macro_f1_numpy(targets, preds) == 1.0

    def test_macro_f1_all_wrong(self):
        targets = np.array([0, 0, 0])
        preds = np.array([1, 1, 1])
        # All predictions are wrong
        result = macro_f1_numpy(targets, preds)
        assert result == 0.0

    def test_macro_f1_returns_float(self):
        targets = np.array([0, 1])
        preds = np.array([0, 1])
        result = macro_f1_numpy(targets, preds)
        assert isinstance(result, float)

    def test_macro_f1_binary_classification(self):
        # Balanced binary classification
        targets = np.array([0, 0, 1, 1])
        preds = np.array([0, 1, 1, 0])
        # Each class: 1 TP, 1 FN, 1 FP
        # Precision = 1/2, Recall = 1/2, F1 = 0.5 for each class
        # Macro F1 = 0.5
        result = macro_f1_numpy(targets, preds)
        assert abs(result - 0.5) < 1e-6

    def test_macro_f1_with_labels_parameter(self):
        targets = np.array([0, 1, 2])
        preds = np.array([0, 1, 2])
        # Only consider labels 0 and 1
        result = macro_f1_numpy(targets, preds, labels=[0, 1])
        assert result == 1.0

    def test_macro_f1_multiclass(self):
        # 3-class problem with varying performance
        targets = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([0, 0, 1, 2, 2, 1])
        # Class 0: 2 correct, F1 = 1.0
        # Class 1: 1 TP, 1 FN (predicted 2), 1 FP (from class 2)
        # Class 2: 1 TP, 1 FN (predicted 1), 1 FP (from class 1)
        result = macro_f1_numpy(targets, preds)
        assert 0.0 < result < 1.0

    def test_macro_f1_single_class(self):
        targets = np.array([0, 0, 0])
        preds = np.array([0, 0, 0])
        result = macro_f1_numpy(targets, preds)
        assert result == 1.0

    def test_macro_f1_imbalanced_classes(self):
        # Imbalanced: many class 0, few class 1
        targets = np.array([0, 0, 0, 0, 0, 1])
        preds = np.array([0, 0, 0, 0, 0, 1])
        result = macro_f1_numpy(targets, preds)
        # Both classes perfectly predicted
        assert result == 1.0
