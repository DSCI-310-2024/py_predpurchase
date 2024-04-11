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

*Please refer to the 'Example usage' page to see an interactive, step by step, demonstration of each function in this package.*

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`py_predpurchase` was created by Nour Abdelfattah, Sana Shams, Calvin Choi, Sai Pusuluri. It is licensed under the terms of the MIT license.

## Credits

`py_predpurchase` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
