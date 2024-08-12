import os
import warnings
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from piml.models import GAMClassifier
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the Generalized Additive Model Classifier in Pygam."""

    model_name = "Generalized Additive Model Classifier in Pygam"

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        n_splines: int = 5,
        spline_order: int = 3,
        lam: float = 0.6,
        max_iter: int = 100,
        **kwargs,
    ):
        """
        Initializes the model with specified configurations.

        Parameters:
            feature_names (Optional[List[str]]): The list of feature names. 
                                                 Default is None.
            feature_types (Optional[List[str]]): The list of feature types. 
                                                 Available types include “numerical” 
                                                 and “categorical”. Default is None.
            n_splines (int): Number of splines to use for the feature function. 
                             Must be non-negative.
            spline_order (int): Order of spline to use for the feature function.
                                Must be non-negative.
            lam (float): Strength of smoothing penalty. Must be a positive float. 
                                Larger values enforce stronger smoothing. 
                                If single value is passed, it will be repeated for 
                                every penalty. If iterable is passed, the length of lam 
                                must be equal to the length of penalties.
            max_iter (int): Maximum number of iterations allowed for the solver to converge.
            **kwargs: Additional keyword arguments for model configuration.

        Returns:
            None.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.n_splines = int(n_splines)
        self.spline_order = int(spline_order)
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> GAMClassifier:
        """Build a new binary classifier."""
        model = GAMClassifier(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            n_splines=self.n_splines,
            spline_order=self.spline_order,
            lam=self.lam,
            max_iter=self.max_iter,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the binary classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        return (
            f"Model name: {self.model_name} ("
            f"feature_types: {self.feature_types}, "
            f"n_splines: {self.n_splines}, "
            f"spline_order: {self.spline_order}, "
            f"lam: {self.lam}, "
            f"max_iter: {self.max_iter})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
