import numpy as np
from sklearn.manifold import TSNE


class TSNEDimensionReducer:
    """
    A class that uses t-SNE to reduce the dimensions of feature vectors.
    """

    def __init__(self, n_components=2):
        """
        Initializes a new instance of the TSNEDimensionReducer class.

        Args:
            n_components (int): The number of components (dimensions) to reduce to.
        """
        self.n_components = n_components
        self.tsne = TSNE(n_components=self.n_components)

    def reduce_dimensions(self, feature_vectors):
        """
        Reduces the dimensions of the given feature vectors using t-SNE.

        Args:
            feature_vectors (numpy.ndarray): A 2D array of feature vectors.

        Returns:
            numpy.ndarray: A 2D array of reduced feature vectors.
        """
        return self.tsne.fit_transform(feature_vectors)
