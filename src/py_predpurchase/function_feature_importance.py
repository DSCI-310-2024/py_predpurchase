import pandas as pd

def get_feature_importances(model, X_columns):
    """
    Fits the model, extracts feature importances, sorts them, and returns them as a DataFrame.
    
    Parameters:
    ----------
    - model: The trained machine learning model with a feature_importances_ attribute.
    - X_columns: The column names of the features in entered dataset.

    Returns:
    ----------
    pandas.DataFrame
    - DataFrame: 
        - 1 column containing the feature importances sorted in descending order. 
        - index of DataFrame has feature names

    Examples:
    --------
    - Assuming:
        - random_forest is your RandomForestClassifier instance
        - X_train.columns is your feature names
        >>> feature_importances_df = get_feature_importances(random_forest, X_train.columns)


    Notes: 
    -------
    This function uses the pandas library produce the results as a pandas DataFrame.

    """

    # checking if model has been fitted and has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("This model does not have the 'feature_importances_' attribute. Make sure your model is tree-based, and has been fitted on the data.")
    
    # check if X_columns is not empty
    if not X_columns:
        raise ValueError("The list of feature names (X_columns) cannot be empty.")

    # obtaining feature importances
    feature_importances_raw = pd.Series(model.feature_importances_, index=X_columns)

    # sorting feature importances
    feature_importances_sorted = feature_importances_raw.sort_values(ascending=False)
    
    # Return as DataFrame 
    return feature_importances_sorted.to_frame(name='Importance')

    