import numpy as np


class KMeans:
    """
    K-means clustering algorithm.
    """

    def __init__(self, k, random_state=None):
        """
        Initializes the KMeans object.

        Parameters:
            k (int): Number of clusters.
            random_state (int or None): Random seed used for initializing the centroids.
        """
        self.k = k
        self.centroids = None
        self.random_state = random_state

    def fit(self, X, max_iter=100, init_centroids=None):
        """
        Runs the K-means algorithm to cluster the input data.

        Parameters:
            X (numpy.ndarray): Input data, shape (n_samples, n_features).
            max_iter (int, optional): Maximum number of iterations to perform. Defaults to 100.
            init_centroids (numpy.ndarray, optional): Initial centroids, shape (k, n_features). If None,
                                                      initial centroids are chosen randomly from the input data.

        Returns:
            centroids (numpy.ndarray): Final centroids, shape (k, n_features).
            labels (numpy.ndarray): Labels of the input data points, shape (n_samples,).
        """
        if init_centroids is None:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            self.centroids = X[np.random.choice(
                X.shape[0], self.k, replace=False)]
        else:
            self.centroids = init_centroids

        for i in range(max_iter):
            # Assign each data point to the nearest centroid
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids to be the mean of the assigned data points
            new_centroids = np.array(
                [X[labels == j].mean(axis=0) for j in range(self.k)])

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            else:
                self.centroids = new_centroids

        return self.centroids, labels
