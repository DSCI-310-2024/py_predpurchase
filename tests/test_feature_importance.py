import numpy as np
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from py_predpurchase.function_feature_importance import get_feature_importances

class DummyModel:
    """ A dummy model to simulate a fitted model with feature importances. """
    def __init__(self, feature_importances):
        self.feature_importances_ = feature_importances

def test_fitted_model_with_feature_importances():
    """Tests correct output format and values when given a fitted model with feature importances."""
    feature_importances = np.array([0.2, 0.3, 0.5])
    model = DummyModel(feature_importances)
    feature_names = pd.Index(['feature_1', 'feature_2', 'feature_3'])

    importances_df = get_feature_importances(model, feature_names)
    
    assert isinstance(importances_df, pd.DataFrame), "Output should be a pandas DataFrame"
    assert not importances_df.empty, "The DataFrame should not be empty"
    assert importances_df.shape == (3, 1), "DataFrame should have one column and three rows"
    assert list(importances_df.index) == ['feature_3', 'feature_2', 'feature_1'], "Feature names should match index of DataFrame based on sorted importances"
    assert list(importances_df['Importance']) == [0.5, 0.3, 0.2], "Importances should be sorted in descending order"


def test_unfitted_model_error():
    """Ensures that a ValueError is raised when an unfitted model is provided."""
    model = RandomForestClassifier()
    feature_names = pd.Index(['feature_1', 'feature_2', 'feature_3'])
    
    with pytest.raises(ValueError):
        get_feature_importances(model, feature_names)

def test_non_tree_based_model_error():
    """Ensures that a ValueError is raised when a non-tree-based model is provided."""
    model = LinearRegression()
    feature_names = pd.Index(['feature_1'])
    
    with pytest.raises(ValueError):
        get_feature_importances(model, feature_names)

def test_invalid_model_type_error():
    """Ensures that a ValueError is raised when a non-model object is given."""
    model = "Not a model"
    feature_names = pd.Index(['feature_1'])
    
    with pytest.raises(ValueError, match="This model does not have the 'feature_importances_' attribute"):
        get_feature_importances(model, feature_names)

def test_empty_feature_names_error():
    """Ensures that a ValueError is raised when no feature names are provided."""
    feature_importances = np.array([0.1, 0.2])
    model = DummyModel(feature_importances)
    feature_names = pd.Index([])

    with pytest.raises(ValueError):
        get_feature_importances(model, feature_names)

def test_feature_importances_sorted_correctly():
    """Tests that the feature importances are correctly sorted in descending order."""
    feature_importances = np.array([0.1, 0.4, 0.2, 0.3])
    model = DummyModel(feature_importances)
    feature_names = pd.Index(['feature_0', 'feature_1', 'feature_2', 'feature_3'])
    
    importances_df = get_feature_importances(model, feature_names)
    sorted_importances = np.sort(feature_importances)[::-1]  # sort in descending order
    assert np.array_equal(importances_df['Importance'], sorted_importances), "Importances should be sorted in descending order"

def test_non_string_feature_names():
    """Tests that function can handle feature names that are not strings."""
    feature_importances = np.array([0.5, 0.5])
    model = DummyModel(feature_importances)
    feature_names = pd.Index([True, False])  # using boolean values as feature names
    
    importances_df = get_feature_importances(model, feature_names)
    assert list(importances_df.index) == [True, False], "DataFrame index should handle non-string feature names"
