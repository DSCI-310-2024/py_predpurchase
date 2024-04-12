# py_predpurchase

[![codecov](https://codecov.io/gh/DSCI-310-2024/py_predpurchase/graph/badge.svg?token=ykj5GDrW0K)](https://codecov.io/gh/DSCI-310-2024/py_predpurchase)

```py_predpurchase``` is a package for predicting online shopper purchasing intentions, whether an online shopper will purchase their current browsing session or not. This package contains functions to aid with the data analysis processes including conducting data preprocessing as well as calculating classification metrics, cross-validation scores and feature importances.

**Full Documentation hosted on Read the Docs**: https://py-predpurchase.readthedocs.io/en/latest/index.html

## Installation

```bash
$ pip install py_predpurchase
```

## Usage

```py_predpurchase``` can be used to:

* Apply preprocessing transformations to the data, including scaling, encoding, and passing through features as specified.
* Calculate the cross-validation results for four common off-the-shelf models (Dummy, KNN, SVM and RandomForests)
* Fit a given model, extract feature importances, sort in descending order, and return them as a DataFrame.
* Calculate the classification metrics for model predictions including precision, recall, accuracy and F1 scores.

*Please refer to the 'Example usage' page on the [Read the Docs](https://py-predpurchase.readthedocs.io/en/latest/index.html) package documentation for a step-by-step, demonstration of each function in this package.*

Below is an example usage for one of our functions, `calculate_classification_metrics` 

``` python

import numpy as np
from py_predpurchase.function_classification_metrics import calculate_classification_metrics

# dummy data
y_true = [1,0,1,1,1,0,0,1,0,1]
y_pred = [1,1,1,0,1,0,0,1,0,0]

# using the function
calculate_classification_metrics(y_true, y_pred)

```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a [Code of Conduct](https://github.com/DSCI-310-2024/py_predpurchase/blob/main/CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## License

`py_predpurchase` was created by Nour Abdelfattah, Sana Shams, Calvin Choi, Sai Pusuluri. It is licensed under the terms of the MIT license.

## Other packages

`pandas`: Pandas is an extensive tool for data manipulation, py_predpurchase specializes in applying machine learning with basic data manipulation, offering functionalities to utilize off-the-shelf machine learning models. When comparing it to something like the [E-Commerce Tools Package](https://pypi.org/project/ecommercetools/0.42.9/), our use of `pandas` along with `sklearn` allows us to manipulate and analyze data in a more primitive setting. The E-Commerce Tools Package is catered more towards transactional data with tools and functions for stock management and ledger items. `pandas` provides a simpler solution suited for the dataset used in py_predpurchase as the dataset pertains to consumer behaviour and E-Commerce marketing metrics which are less sophisticated.  

`scikit-learn`: Scikit-learn excels in model building, but py_predpurchase extends its offerings by providing advanced tools for interpreting model outcomes. Unlike scikit-learn's broader approach, our package includes specific methods for detailing the impact of each predictor on the purchasing decision, allowing for a deeper understanding of model dynamics and more accurate validation scores. py_predpurchase benefits from these specialized insights and improves your model's predictive performance in the context of online shopping.

## Credits

`py_predpurchase` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
