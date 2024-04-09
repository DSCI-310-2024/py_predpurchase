import numpy as np
import pytest
import sys
import os

from py_predpurchase.function_classification_metrics import calculate_classification_metrics

def test_perfect_predictions():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert all(np.isclose(value, 1.0) for value in metrics.values()), "All metrics should be 1.0 for perfect predictions."

def test_all_incorrect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics['Accuracy'] == 0, "Accuracy should be 0 for all incorrect predictions."

def test_mismatched_array_lengths():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])
    with pytest.raises(ValueError):
        calculate_classification_metrics(y_true, y_pred)

def test_non_numeric_inputs():
    y_true = ['a', 'b', 'c']
    y_pred = ['a', 'b', 'd']
    with pytest.raises(TypeError):
        calculate_classification_metrics(y_true, y_pred)

def test_empty_arrays():
    y_true = np.array([])
    y_pred = np.array([])
    with pytest.raises(ValueError):
        calculate_classification_metrics(y_true, y_pred)