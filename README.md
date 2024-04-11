# py_predpurchase

```py_predpurchase``` is a package for predicting online shopper purchasing intentions, whether an online shopper will make a purchase from their current browsing session or not. This package contains functions to aid with the data analysis processes including conducting data preprocessing as well as calculating classification metrics, cross validation scores and feature importances.

## Installation

```bash
$ pip install py_predpurchase
```

## Usage

```py_pred``` can be used to:

* Apply preprocessing transformations to the data, including scaling, encoding, and passing through features as specified.
* Calculate the cross validation results for a four common off-the-shelf models (Dummy, KNN, SVM and RandomForests)
* Fit a given model, and extract feature importances, sorted in descending order, and returns them as a DataFrame.
* Calculate the classification metrics for model predictions including precision, recall, accuracy and F1 scores.

... as follows:

``` python
from py_predpurchase.function_preprocessing import numerical_categorical_preprocess

from py_prepurchase.function_model_cross_val import model_cross_validation

from py_prepurchase.function_feature_importance import get_feature_importances

from py_prepurchase.function_classification_metrics import calculate_classification_metrics

import pandas as pd
import numpy as np

from pycounts.pycounts import count_words
from pycounts.plotting import plot_words
import matplotlib.pyplot as plt

x_train = "X_train.csv" # path to your x train data
X_test = "X_test.csv" # path to your x test data
numeric_features = "numeric_features" # path to, or object (eg. list of strings) containing names of numerical features 
categorical_features = "categorical_features"  # path to, or object (eg. list of strings) containing categorical features 

preprocessed_data = numerical_categorical_preprocess(x_train, x_test, numeric_features, categorical_features)

preprocessed_training_data = "processed_training.csv" # path to preprocessed training data
preprocessed_testing_data = "processed_testing.csv" # path to preprocessed testing data
k = 5 # k value hyperparameter for KNearestNeighbor
target: "target_name" # str target column name
gamma: 10 # gamma value hyperparameter for SVM


cross_val_results = model_cross_validation(preprocessed_training_data, preprocessed_testing_data, target, k, gamma):


model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X, y) # tree-based model, fitted on data
X_columns = ['a', 'b', 'c'] # names of features from the model

get_feature_importances(model, X_columns)

y_true = "y_test.csv" # path to y_test data (true y values)
y_pred = "y_pred.csv" # path to predicted target, given explanatory (X) features that the model was fitted on.  

calculate_classification_metrics(y_true, y_pred)


```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`py_predpurchase` was created by Nour Abdelfattah, Sana Shams, Calvin Choi, Sai Pusuluri. It is licensed under the terms of the MIT license.

## Credits

`py_predpurchase` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
