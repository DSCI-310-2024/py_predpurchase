import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

def model_cross_validation(preprocessed_training_data, preprocessed_testing_data, target, k, gamma):
	"""
	Calculates the cross validation results for a four common off-the-shelf models (Dummy, KNN, SVM and RandomForests)
	using preprocessed and cleaned training and testing datasets. Random forests and Dummy hyperparameters are fixed for simplicity sake.
	
	Parameters:
	- preprocessed_training_data: csv, cleaned and preprocessed training data 
	- preprocessed_testing_data: csv, cleaned and preprocessed testing data
	- target: str target column name
	- k: k value hyperparameter for KNearestNeighbours Int
	- gamma: gamma value hyperparameter for SVM
	
	Returns:
	- dictionary, containing cross validation results (mean and std of scores) from specified model
	"""

	train_data = pd.read_csv(preprocessed_training_data)
	test_data = pd.read_csv(preprocessed_testing_data)
	X_train = train_data.drop(target, axis=1)
	y_train = train_data[target]
	X_test = test_data.drop(target, axis=1)
	y_test = test_data[target]

	# Function to perform cross-validation and return mean/std scores
	def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
		scores = cross_validate(model, X_train, y_train, **kwargs)
		mean_scores = pd.DataFrame(scores).mean()
		std_scores = pd.DataFrame(scores).std()
		return pd.Series({col: f"{mean:.3f} (+/- {std:.3f})" for col, mean, std in zip(mean_scores.index, mean_scores, std_scores)})

	results_dict = {}

	#Dummy Classifier
	dummy = DummyClassifier(strategy='most_frequent')
	results_dict["dummy"] = mean_std_cross_val_scores(dummy, X_train, y_train, cv=5, return_train_score=True)
	
	# kNN Classifier
	best_k = k
	knn = KNeighborsClassifier(n_neighbors=best_k)
	results_dict["knn"] = mean_std_cross_val_scores(knn, X_train, y_train, cv=5, return_train_score=True)

	# SVM Classifier
	svm = SVC(gamma= gamma)
	results_dict["SVM"] = mean_std_cross_val_scores(svm, X_train, y_train, cv=5, return_train_score=True)

	# Random Forest Classifier
	random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=123)
	results_dict["random_forest"] = mean_std_cross_val_scores(random_forest, X_train, y_train, cv=5, return_train_score=True)

	return results_dict