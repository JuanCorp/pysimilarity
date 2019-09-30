import numpy as np
from .base import BaseSimilarity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class ClassifierSimilarity(BaseSimilarity):
    """Compute Similarity between two matrices by using a classifier
       and computing probability of  being in the space of interest. 
       Useful for LookAlike problems, where potential prospects are 
       searched. The members of a sample of interest of a population 
       are marked with 1, while the rest are marked with 0. A classifier
       is trained to distinguish the 1s from the 0s. The 0s that are 
       marked as 1s by the classifier, because they have similar properties
       to the ones marked with 1, are tagged as similar.
    """

    def __init__(self, estimator=None):
        """Initialize the  estimator to use.

        Keyword Arguments:
            estimator {sklearn.BaseEstimator or Estimator that implements the fit method} -- Classifier to use.
            If None, will use a Random Forest Classifier.
        """
        if estimator is None:
            estimator = RandomForestClassifier()
        if "fit" not in dir(estimator):
            raise AttributeError("Estimator must include a fit method.")

        self.estimator = estimator
        self.is_fit = False

    def fit(self, X, y):
        """
        Train the model with the training data X and training targets y.
        In this case, y represents a 1D vector with only binary values, 
        where the 1s represent records of intereset.
        Arguments:
            X {numpy.ndarray or pandas.Dataframe} -- Base data to be fitted.
            y {numpy.ndarray or pandas.Dataframe} -- Target output to be fitted.
        """
        self.X = X.values if isinstance(X, pd.DataFrame) else X

        self.y = y.values if isinstance(y, pd.Series) else y

        _, count = np.unique(y, return_counts=True)
        if len(count) > 2:
            raise ValueError("Target must only contain two unique values.")

        self.estimator.fit(self.X, self.y)
        self.is_fit = True
        return self.estimator

    def transform(self, X=None):
        """

        Keyword Arguments:
            X {np.ndarray or pandas.Dataframe} -- Data to be passed through the model
            to compute similarity. If None will use the input X. (default: {None})

        Returns:
            [np.ndarray] -- A 1D of size m, m being the number of rows in X, 
                containing the probability of being similar per data point.
        """
        if not self.is_fit:
            raise Exception("The fit method has not been called.")

        data = X if X is not None else self.X
        if "predict_proba" in dir(self.estimator):
            similarity = self.estimator.predict_proba(data)[:, 1]
        else:
            similarity = self.estimator.predict(data)
        return similarity

    def fit_transform(self, X, y, X_compare=None):
        """Call fit and transform in succession.

        Arguments:
            X {numpy.ndarray or pandas.Dataframe} -- Base data to be fitted.
            y {numpy.ndarray or pandas.Dataframe} -- Target output to be fitted.
            X_compare {numpy.ndarray or pandas.Dataframe}  --  Data to be passed through the model
            to compute similarity. If None will use the input X. (default: {None})
        """
        self.fit(X, y)
        return self.transform(X_compare)
