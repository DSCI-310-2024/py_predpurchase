import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def numerical_categorical_preprocess(X_train, X_test, y_train, y_test, numeric_features, categorical_features):
    """
    Applies preprocessing transformations to the data, including scaling, encoding, and passing through features as specified.
    This function requires target data to be provided and includes it in the output DataFrames.

    Parameters:
    ----------
    X_train : DataFrame
        Training feature data.
    X_test : DataFrame
        Testing feature data.
    y_train : DataFrame or Series
        Training target data.
    y_test : DataFrame or Series
        Testing target data.
    numeric_features : list
        Names of numeric features to scale.
    categorical_features : list
        Names of categorical features to encode.
    
    Returns:
    ----------
    Tuple
        Contains preprocessed training and testing DataFrames including target data, 
        and transformed column names.
   
    Examples:
    --------
    Assume you want to transform the following features and your data set has already been split
    into train and test

    >>> numeric_features = ['feature1', 'feature2']
    >>> categorical_features = ['feature3', feature4']
    >>> train_transformed, test_transformed, transformed_columns = numerical_categorical_preprocess(
            X_train, X_test, y_train, y_test, numeric_features, categorical_features)
    
    The function will transform feature1,2,3,4 accordingly, carrying out scaling and one-hot encoding and 
    storing the preprocessed data in 'train_transformed' and 'test_transformed'. Column names will also be stored in 
    'transformed_columns'.
    
    """
    
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    # Defining preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features),
        ]
    )
    
    # Apply preprocesor on train and test data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Retrieve transformed column names 
    transformed_columns = preprocessor.get_feature_names_out()
    
    # Reconstruct df with transformed data and target columns
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=transformed_columns)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=transformed_columns)
    
    if y_train is not None:
        y_train = y_train.reset_index(drop=True)
        train_transformed = pd.concat([X_train_transformed_df, y_train], axis=1)
    else:
        train_transformed = X_train_transformed_df

    if y_test is not None:
        y_test = y_test.reset_index(drop=True)
        test_transformed = pd.concat([X_test_transformed_df, y_test], axis=1)
    else:
        test_transformed = X_test_transformed_df

    return train_transformed, test_transformed, transformed_columns