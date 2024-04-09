import numpy as np
import pytest
import pandas as pd
import sys
import os 
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


from py_predpurchase.function_feature_importance import get_feature_importances 

# creating a mock model

class DummyModel:
    def __init__(self, feature_importances):
        self.feature_importances_ = feature_importances

def test_fitted_model_uniform_feature_importances():
    # creating a dummy model with uniform feature importances
    feature_importances = np.array([1/3, 1/3, 1/3])
    model = DummyModel(feature_importances)
    feature_names = ['feature_1', 'feature_2', 'feature_3']
    
    # assume get_feature_importances is adjusted to not require fitting
    importances_df = get_feature_importances(model, feature_names)
    
    # perform checks
    assert isinstance(importances_df, pd.DataFrame), "Output should be a pandas DataFrame"
    assert not importances_df.empty, "The DataFrame should not be empty"
    assert importances_df.shape[1] == 1, "DataFrame should have one column for importances"
    assert importances_df.shape[0] == 3, "DataFrame should have a row for each feature"
    # This check ensures that each feature's importance is approximately 1/3, since this is checking uniform case
    assert all(np.isclose(importances_df.iloc[:, 0], 1/3)), "All features should have equal importance"
    
def test_unfitted_model_error():
    model = RandomForestClassifier(random_state=42)
    feature_names = [f'feature_{i}' for i in range(5)]
    
    with pytest.raises(ValueError):
        get_feature_importances(model, feature_names)

def test_non_tree_based_model_error():
    model = LinearRegression()
    model.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))  # just dummy fit here to avoid unfitted model error
    feature_names = ['feature_0']
    
    with pytest.raises(ValueError):
        get_feature_importances(model, feature_names)

def test_invalid_model_type_error():
    model = "Not a model"
    feature_names = ['feature_0']
    with pytest.raises(ValueError):  
        get_feature_importances(model, feature_names)

def test_feature_importances_sorted_correctly():
    feature_importances = np.array([0.1, 0.4, 0.2, 0.3])
    model = DummyModel(feature_importances)
    feature_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
    importances_df = get_feature_importances(model, feature_names)
    
    sorted_importances = np.sort(feature_importances)[::-1]  # sort in descending order
    assert np.array_equal(importances_df['Importance'], sorted_importances), "Importances should be sorted in descending order"

def test_non_string_feature_names():
    feature_importances = np.array([0.5, 0.5])
    model = DummyModel(feature_importances)
    feature_names = [True, False]  # using boolean values as feature names
    importances_df = get_feature_importances(model, feature_names)
    
    assert list(importances_df.index) == feature_names, "dataFrame index should handle non-string feature names"

def test_empty_feature_names():
    model = DummyModel(np.array([]))
    feature_names = []
    with pytest.raises(ValueError):
        get_feature_importances(model, feature_names)