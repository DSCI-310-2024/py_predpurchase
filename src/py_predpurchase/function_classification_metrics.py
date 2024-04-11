from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculates classification metrics for model predictions including precision, 
    recall, accuracy, and F1 scores. 
    
    Parameters:
    ----------
    y_true : array-like or pd.Series
        True target values in a dataset.
    y_pred : array-like or pd.Series
        Predicted target values by the model.
    
    Returns:
    ----------
    dict
        Contains precision, recall, accuracy, and F1 score.
    
    Examples:
    --------

    Assume `y_true` and `y_pred` are as follows:
    >>> y_true = [0, 1, 2, 0, 1]
    >>> y_pred = [0, 2, 1, 0, 0]
    >>> calculate_classification_metrics(y_true, y_pred)

    """

    if not all(isinstance(y, (int, float, np.number)) for y in np.concatenate([y_true, y_pred])):
        raise TypeError("y_true and y_pred must contain numeric values only.")

    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    
    metrics = {
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics