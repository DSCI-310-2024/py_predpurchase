import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def numerical_categorical_preprocess(X_train, X_test, y_train, y_test, numeric_features, categorical_features):
    """
    Applies preprocessing transformations to the data, including scaling, encoding, and passing through features as specified.
    This function requires target data to be provided and includes it in the output DataFrames.

    Parameters:
    - X_train: DataFrame, training feature data
    - X_test: DataFrame, testing feature data
    - y_train: DataFrame, training target data
    - y_test: DataFrame, testing target data
    - numeric_features: list, names of numeric features to scale
    - categorical_features: list, names of categorical features to encode
    
    
    Returns:
    - Tuple containing preprocessed training and testing DataFrames including target data, and transformed column names
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