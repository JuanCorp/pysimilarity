import numpy as np
from .base import BaseSimilarity
import pandas as pd
from scipy.spatial.distance import cdist


class DistanceSimilarity(BaseSimilarity):
    """Compute Similarity between two matrices by using the average
       pairwise distances between their data points.
    """

    def __init__(self, distance_metric='auto', batch_size=1000):
        """Initialize the distance metric to use. Vlaues can be a string,
        based on the scipy distance metrics, or a custom function.

        Keyword Arguments:
            distance_metric {str or function} -- Distance metric to be used to determine similarity 
                between data points. If auto, will determine best metric depending on the content
                of the data.(default: {'auto'})
            batch_size {int} -- Number of batches to use per division of data, when calculating distances.
        """
        METRIC_MAPPING = set(['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
                              'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'])
        if isinstance(distance_metric, str) and distance_metric != 'auto' and distance_metric not in METRIC_MAPPING:
            raise ValueError(
                f"If distance metric is string, must be auto or one of:{str(METRIC_MAPPING)}")
        self.distance_metric = distance_metric
        self.batch_size = batch_size
        self.is_fit = False

    def fit(self, X):
        """Sets the data to be compared and the distance metric object to be used.
        If the distance metric is auto, determines the best distance metric to be used.

        Arguments:
            X {numpy.ndarray or pandas.Dataframe} -- Base data to be compared.
        """
        self.X = X.values if isinstance(X, pd.DataFrame) else X

        if self.distance_metric == 'auto':
            self.distance_metric = self._get_best_distance_metric(self.X)

        self.is_fit = True
        return self.distance_metric

    def transform(self, Y):
        """Determines the distance between each data point in Y and all the data points in X.
        Then, computes the average distance for each data point in Y to X. 
        The logic is, the lower the average distance, the more similar they are to X.

        Arguments:
            Y {numpy.ndarray or pandas.Dataframe} -- Data to compare.

        Returns:
            numpy.ndarray -- A 1D of size m, m being the number of rows in Y, 
                with the average distance of each data point to all of X.
        """
        import time
        if not self.is_fit:
            raise Exception("The fit method has not been called.")

        comparison_data = Y.values if isinstance(Y, pd.DataFrame) else Y
        average_distances = list()
        print(
            f"Diving data into {comparison_data.shape[0] // self.batch_size + 1} batches of {self.batch_size} samples each.")
        for i in range(0, comparison_data.shape[0], self.batch_size):
            distances = cdist(
                comparison_data[i: (i + self.batch_size)], self.X, self.distance_metric)
            average_distance = distances.mean(axis=1)
            average_distances.extend(average_distance.tolist())
        return average_distances

    def fit_transform(self, X, Y):
        """Call fit and transform in succession.

        Arguments:
            X {numpy.ndarray or pandas.Dataframe} -- Base data to be compared.
             Y {numpy.ndarray or pandas.Dataframe} -- Data to compare to X.
        """
        self.fit(X)
        return self.transform(Y)

    def _get_sparsity_level(self, X):
        """Calculate the sparsity level of a matrix. This is determined 
           by dividing the total count of non-zeros in the matrix by 
           the total size of the matrix.

        Arguments:
            X {numpy.ndarray} -- Matrix to which the sparsity level will be calculated.
            Must be 2D. 

        Returns:
            [float] -- Sparsity level of the matrix. Real number between 0 and 1.
        """
        if len(X.shape) != 2:
            raise Exception("The matrix must be 2D.")

        replaced_nan_with_zero = np.where(X == np.nan, 0, X)
        total_non_zeros = np.count_nonzero(replaced_nan_with_zero)
        flattened_data_size = X.shape[0] * X.shape[1]
        return 1 - (total_non_zeros / flattened_data_size)

    def _is_unit_interval(self, X):
        """Determine is the range of the values of a matrix is between
        0 and 1.

        Arguments:
            X {numpy.ndarray} -- Matrix to which an unit interval range will be calculated. 

        Returns:
            [bool] -- Whether the matrix range is between 0 and 1.
        """
        return X.min() >= 0 and X.max() <= 1

    def _get_best_distance_metric(self, X):
        """
        Get the best distance metric based on certain conditions. 
        Will be expanded on the future. The conditions right now are:

        - If the data range is between 0 and 1 and the only values are 0 and 1 then use jaccard distance.
        - If the data range is between 0 and 1, and is somewhat sparse, use cosine distance.
        - Else if the data is not very sparse and not between 0 and 1, use euclidean distance.
        - Else use minkowski distance.
        Arguments:
            X {numpy.ndarray} --  Matrix containing data that will be used to determine 
                the best distance metric to use.

        Returns:
            [str] -- The name of the best metric to use.
        """
        sparsity_level = self._get_sparsity_level(X)
        is_unit_interval = self._is_unit_interval(X)
        _, unique_count = np.unique(X, return_counts=True)
        best_metric = ''
        if is_unit_interval and len(unique_count) == 2:
            best_metric = 'jaccard'

        elif is_unit_interval and sparsity_level > 0.05:
            best_metric = 'cosine'

        elif sparsity_level < 0.05:
            best_metric = 'euclidean'
        else:
            best_metric = 'minkowski'

        return best_metric
