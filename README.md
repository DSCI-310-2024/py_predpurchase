# py_predpurchase

[![codecov](https://codecov.io/gh/DSCI-310-2024/py_predpurchase/graph/badge.svg?token=ykj5GDrW0K)](https://codecov.io/gh/DSCI-310-2024/py_predpurchase)

```py_predpurchase``` is a package for predicting online shopper purchasing intentions, whether an online shopper will make a purchase from their current browsing session or not. This package contains functions to aid with the data analysis processes including conducting data preprocessing as well as calculating classification metrics, cross validation scores and feature importances.

**Full Documentation hosted on Read the Docs**: https://py-predpurchase.readthedocs.io/en/latest/index.html

## Installation

```bash
$ pip install py_predpurchase
```

## Usage

```py_predpurchase``` can be used to:

* Apply preprocessing transformations to the data, including scaling, encoding, and passing through features as specified.
* Calculate the cross validation results for a four common off-the-shelf models (Dummy, KNN, SVM and RandomForests)
* Fit a given model, and extract feature importances, sorted in descending order, and returns them as a DataFrame.
* Calculate the classification metrics for model predictions including precision, recall, accuracy and F1 scores.

*Please refer to the 'Example usage' page on the [Read the Docs](https://py-predpurchase.readthedocs.io/en/latest/index.html) package documentation for a step by step, demonstration of each function in this package.*

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

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`py_predpurchase` was created by Nour Abdelfattah, Sana Shams, Calvin Choi, Sai Pusuluri. It is licensed under the terms of the MIT license.

## Other packages

`pandas`: While Pandas is an extensive tool for data manipulation, py_predpurchase specializes in e-commerce analytics, offering tailored functionalities that go beyond general data handling. It includes advanced features for importing e-commerce datasets, detecting unique shopping-related variables etc.py_predpurchase is for refined insights that are specifically geared towards optimizing online shopping platforms and driving sales.

`scikit-learn`: Scikit-learn excels in model building, but py_predpurchase extends its offerings by providing advanced tools for interpreting model outcomes. Unlike scikit-learn's broader approach, our package includes specific methods for detailing the impact of each predictor on the purchasing decision, allowing for a deeper understanding of model dynamics and more accurate validation scores. py_predpurchase benefits from these specialized insights and improve your model's predictive performance in the context of online shopping.

## Credits

`py_predpurchase` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
