import pandas as pd
import pytest
import sys
import os

from src.py_predpurchase import model_cross_validation

@pytest.fixture
def test_data():
    # Create a mock dataframe with the expected structure
    data = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5, 2, 2, 3, 3, 3, 3],
    "feature2": [5, 4, 3, 2, 1, 4, 0, 5, 5, 5, 5],
    'target': [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
})
    return(data)


def test_model_cross_validation_results(test_data):
    """
    This test checks if the model_cross_validation function returns results 
    for all the expected models: 'dummy', 'knn', 'SVM', and 'random_forest'.
    """
    results = model_cross_validation(test_data, test_data, 'target', 3, 0.01)

    assert all(model in results for model in ['dummy', 'knn', 'SVM', 'random_forest']), "All expected models should be in the output."

def test_model_cross_validation_scores(test_data):
    """
    This test verifies that the model_cross_validation function returns scores for each model,
    including 'fit_time', 'score_time', 'test_score', and 'train_score'. It also checks if the
    test score falls within the valid range of 0 to 1.
    """
    results = model_cross_validation(test_data, test_data, 'target', 3, 0.01)

    for model, scores in results.items():
        assert 'fit_time' in scores, f"Fit time should be in the {model} model scores."
        assert 'score_time' in scores, f"Score time should be in the {model} model scores."
        assert 'test_score' in scores, f"Test score should be in the {model} model scores."
        assert 'train_score' in scores, f"Train score should be in the {model} model scores."
        assert 0 <= float(scores['test_score'].split()[0]) <= 1, f"Test score for {model} should be between 0 and 1."

def test_invalid_parameters(test_data):
    """
    This test uses pytest.raises to check if model_cross_validation raises a ValueError
    when provided with invalid parameters, such as a negative 'k' value or a negative 'gamma' value.
    """
    with pytest.raises(ValueError):
        model_cross_validation(test_data, test_data, 'target', -1, 0.01)

    with pytest.raises(ValueError):
        model_cross_validation(test_data, test_data, 'target', 3, -0.01)

if __name__ == '__main__':
    pytest.main([__file__])