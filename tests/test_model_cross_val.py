import pandas as pd
import pytest
import sys
import os

from py_predpurchase.function_model_cross_val import model_cross_validation

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
    results = model_cross_validation(test_data, test_data, 'target', 3, 0.01)

    assert all(model in results for model in ['dummy', 'knn', 'SVM', 'random_forest']), "All expected models should be in the output."

def test_model_cross_validation_scores(test_data):
    #data = test_data
    results = model_cross_validation(test_data, test_data, 'target', 3, 0.01)

    for model, scores in results.items():
        assert 'fit_time' in scores, f"Fit time should be in the {model} model scores."
        assert 'score_time' in scores, f"Score time should be in the {model} model scores."
        assert 'test_score' in scores, f"Test score should be in the {model} model scores."
        assert 'train_score' in scores, f"Train score should be in the {model} model scores."
        assert 0 <= float(scores['test_score'].split()[0]) <= 1, f"Test score for {model} should be between 0 and 1."

def test_invalid_parameters(test_data):
    with pytest.raises(ValueError):
        model_cross_validation(test_data, test_data, 'target', -1, 0.01)

    with pytest.raises(ValueError):
        model_cross_validation(test_data, test_data, 'target', 3, -0.01)

if __name__ == '__main__':
    pytest.main([__file__])