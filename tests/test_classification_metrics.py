import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.function_classification_metrics import calculate_classification_metrics

def test_perfect_predictions():
    """
    Tests that all classification metrics return a score of 1.0 for perfect predictions,
    where predicted values exactly match the true values.
    """

    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert all(np.isclose(value, 1.0) for value in metrics.values()), "All metrics should be 1.0 for perfect predictions."

def test_all_incorrect_predictions():
    """
    Tests that the accuracy is 0 for a case where all predictions are incorrect.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics['Accuracy'] == 0, "Accuracy should be 0 for all incorrect predictions."

def test_mismatched_array_lengths():
    """
    Tests that a ValueError is given when the lengths of y_true and y_pred arrays do not match.
    """

    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])
    with pytest.raises(ValueError):
        calculate_classification_metrics(y_true, y_pred)

def test_non_numeric_inputs():
    """
    Tests that a TypeError is given when inputs are not numeric.
    """

    y_true = ['a', 'b', 'c']
    y_pred = ['a', 'b', 'd']
    with pytest.raises(TypeError):
        calculate_classification_metrics(y_true, y_pred)

def test_empty_arrays():
    """
    Tests that a ValueError is given when inputs are empty arrays.
    """
    y_true = np.array([])
    y_pred = np.array([])
    with pytest.raises(ValueError):
        calculate_classification_metrics(y_true, y_pred) 