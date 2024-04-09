import pandas as pd
import pytest
import sys
import os


from py_predpurchase.function_model_cross_val import model_cross_validation


     
def test_model_cross_validation_results():
    test_file = 'data/cross_val_test_data.csv'
    results = model_cross_validation(test_file, test_file, 'target', 3, 0.01)

    assert all(model in results for model in ['dummy', 'knn', 'SVM', 'random_forest']), "All expected models should be in the output."

def test_model_cross_validation_scores():
    test_file = 'data/cross_val_test_data.csv'
    results = model_cross_validation(test_file, test_file, 'target', 3, 0.01)

    for model, scores in results.items():
        assert 'fit_time' in scores, f"Fit time should be in the {model} model scores."
        assert 'score_time' in scores, f"Score time should be in the {model} model scores."
        assert 'test_score' in scores, f"Test score should be in the {model} model scores."
        assert 'train_score' in scores, f"Train score should be in the {model} model scores."
        assert 0 <= float(scores['test_score'].split()[0]) <= 1, f"Test score for {model} should be between 0 and 1."

def test_invalid_parameters():
    test_file = 'data/cross_val_test_data.csv'

    with pytest.raises(ValueError):
        model_cross_validation(test_file, test_file, 'target', -1, 0.01)

    with pytest.raises(ValueError):
        model_cross_validation(test_file, test_file, 'target', 3, -0.01)

if __name__ == '__main__':
    pytest.main([__file__])