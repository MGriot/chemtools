import numpy as np
from collections import Counter
from chemtools.base.base_models import BaseModel
from chemtools.utils.data import initialize_names_and_counts

class KNearestNeighbors(BaseModel):
    """
    k-Nearest Neighbors (k-NN) classifier.

    This classifier predicts the class of a new data point based on the majority
    class of its 'k' nearest neighbors in the training data.

    Attributes:
        n_neighbors (int): The number of neighbors to use (the 'k' value).
    
    References:
        - https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    """

    def __init__(self, n_neighbors=5):
        super().__init__()
        self.model_name = "k-Nearest Neighbors"
        self.method = "k-NN"
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y, variables_names=None, objects_names=None):
        """
        Stores the training data. For k-NN, 'fitting' is just memorizing the dataset.

        Args:
            X (np.ndarray): The training input samples (n_samples, n_features).
            y (np.ndarray): The target values (class labels) (n_samples,).
            variables_names (list, optional): Names of the variables. Defaults to None.
            objects_names (list, optional): Names of the objects. Defaults to None.
        """
        self.X_train = X
        self.y_train = y
        self.variables, self.objects, self.n_variables, self.n_objects = (
            initialize_names_and_counts(X, variables_names, objects_names)
        )
        return self

    def predict(self, X_new):
        """
        Predicts the class for new data points.

        Args:
            X_new (np.ndarray): New data to predict (n_new_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels for each new data point.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("The model must be fitted before predicting.")

        predictions = [self._predict_single(x) for x in X_new]
        return np.array(predictions)

    def _predict_single(self, x):
        """Predicts the class for a single data point."""
        # 1. Calculate distances to all training points
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        # 2. Get the indices of the k nearest neighbors
        k_neighbor_indices = np.argsort(distances)[:self.n_neighbors]
        
        # 3. Get the labels of the k nearest neighbors
        k_neighbor_labels = [self.y_train[i] for i in k_neighbor_indices]
        
        # 4. Return the most common class label (majority vote)
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    def _get_summary_data(self):
        """Returns a dictionary containing summary data for the model."""
        if self.X_train is None:
             return {}
             
        summary = self._create_general_summary(
            self.n_variables,
            self.n_objects,
            k_Neighbors=f"{self.n_neighbors}"
        )
        return summary
