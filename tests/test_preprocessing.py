import pandas as pd
import numpy as np
import pytest
import sys
import os


from py_predpurchase.function_preprocessing import numerical_categorical_preprocess


@pytest.fixture
def sample_data():
    
    numeric_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
    categorical_features = ['Month', 'VisitorType']
    
    # Sample data setup
    X_train = pd.DataFrame({
        'Administrative': [1, 2, 3],
        'Administrative_Duration': [10, 20, 30],
        'Informational': [4, 5, 6],
        'Informational_Duration': [40, 50, 60],
        'ProductRelated': [7, 8, 9],
        'ProductRelated_Duration': [70, 80, 90],
        'BounceRates': [0.1, 0.2, 0.3],
        'ExitRates': [0.4, 0.5, 0.6],
        'PageValues': [0.7, 0.8, 0.9],
        'SpecialDay': [0.0, 0.1, 0.2],
        'Month': ['Jan', 'Feb', 'Mar'],
        'VisitorType': ['New_Visitor', 'Returning_Visitor', 'Other'],
        
    })

    X_test = pd.DataFrame({
        'Administrative': [4, 5, 6],
        'Administrative_Duration': [40, 50, 60],
        'Informational': [7, 8, 9],
        'Informational_Duration': [70, 80, 90],
        'ProductRelated': [10, 11, 12],
        'ProductRelated_Duration': [100, 110, 120],
        'BounceRates': [0.4, 0.5, 0.6],
        'ExitRates': [0.7, 0.8, 0.9],
        'PageValues': [1.0, 1.1, 1.2],
        'SpecialDay': [0.3, 0.4, 0.5],
        'Month': ['Apr', 'May', 'Jun'],
        'VisitorType': ['Returning_Visitor', 'New_Visitor', 'Other'],
        
    })

    y_train = pd.Series([0, 1, 0], name='Revenue')
    y_test = pd.Series([1, 0, 1], name='Revenue')

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

def test_shape(sample_data):
    """
    Tests that the transformed training and testing data retain the same number of rows
    to ensure no data loss occured during preprocessing.
    
    """
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = sample_data
    train_transformed, test_transformed, _ = numerical_categorical_preprocess(X_train, X_test, y_train, y_test, numeric_features, categorical_features)
    
    assert train_transformed.shape[0] == X_train.shape[0], "Train data row count mismatch after transformation."
    assert test_transformed.shape[0] == X_test.shape[0], "Test data row count mismatch after transformation."

def test_null_values(sample_data):
    """
    Ensures there are no null values in the transformed datasets

    """
    X_train, X_test, _, _, numeric_features, categorical_features = sample_data
    train_transformed, test_transformed, _ = numerical_categorical_preprocess(X_train, X_test, None, None, numeric_features, categorical_features)
    assert not train_transformed.isnull().any().any(), "Null values found in transformed training data"
    assert not test_transformed.isnull().any().any(), "Null values found in transformed test data"


def test_revenue_preservation(sample_data):
    """
    Tests that the 'Revenue' target column is kept unaltered after preprocessing.

    """
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = sample_data
    train_transformed, test_transformed, _ = numerical_categorical_preprocess(X_train, X_test, y_train, y_test, numeric_features, categorical_features)
    
    assert np.array_equal(train_transformed['Revenue'], y_train), "Revenue data altered in training set."
    assert np.array_equal(test_transformed['Revenue'], y_test), "Revenue data altered in testing set."

def test_numerical_features_transformation(sample_data):
    """
    Tests that all specified numeric features are included in the transformed data
    with the correct application of scaling.

    """
    X_train, X_test, y_train, y_test, numeric_features, _ = sample_data
    train_transformed, test_transformed, transformed_columns = numerical_categorical_preprocess(X_train, X_test, y_train, y_test, numeric_features, [])
    
    for feature in numeric_features:
        assert any(col.startswith(f'numeric__{feature}') for col in transformed_columns), f"Numeric feature '{feature}' not found in transformed columns."

def test_categorical_features_transformation(sample_data):
    """
    Tests that categorical features are correctly one-hot encoded and included in the transformed
    data, which will be indicted by the presence of transformed column names.
    
    """
    X_train, X_test, y_train, y_test, _, categorical_features = sample_data
    train_transformed, test_transformed, transformed_columns = numerical_categorical_preprocess(X_train, X_test, y_train, y_test, [], categorical_features)
    
    for feature in categorical_features:
        feature_columns = [col for col in transformed_columns if col.startswith(f'categorical__{feature}')]
        assert len(feature_columns) > 0, f"Categorical feature '{feature}' not found in transformed columns."

if __name__ == "__main__":
    pytest.main()