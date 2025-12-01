import numpy as np
from chemtools.base.base_models import BaseModel
from chemtools.utils.data import initialize_names_and_counts

class KMeans(BaseModel):
    """
    Performs k-Means Clustering.

    k-Means clustering aims to partition n observations into k clusters in which
    each observation belongs to the cluster with the nearest mean (cluster centroid).

    Attributes:
        n_clusters (int): The number of clusters to form.
        max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.
        random_state (int): Determines random number generation for centroid initialization.
        cluster_centers_ (np.ndarray): Coordinates of cluster centers.
        labels_ (np.ndarray): Labels of each point.
    
    References:
        - https://en.wikipedia.org/wiki/K-means_clustering
    """

    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        super().__init__()
        self.model_name = "k-Means Clustering"
        self.method = "k-Means"
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, variables_names=None, objects_names=None):
        """
        Computes k-means clustering.

        Args:
            X (np.ndarray): The input data (n_samples, n_features).
            variables_names (list, optional): Names of the variables. Defaults to None.
            objects_names (list, optional): Names of the objects. Defaults to None.
        """
        self.X = X
        self.variables, self.objects, self.n_variables, self.n_objects = (
            initialize_names_and_counts(X, variables_names, objects_names)
        )

        # Initialize centroids randomly
        rng = np.random.default_rng(self.random_state)
        random_indices = rng.choice(self.n_objects, self.n_clusters, replace=False)
        self.cluster_centers_ = self.X[random_indices]

        for _ in range(self.max_iter):
            # Assign labels based on the old centroids
            self.labels_ = self._assign_labels(self.X)
            
            # Store old centroids
            old_centroids = np.copy(self.cluster_centers_)
            
            # Update centroids
            self._update_centroids()
            
            # Check for convergence
            if np.all(old_centroids == self.cluster_centers_):
                break
        
        return self

    def _assign_labels(self, X):
        """Assigns data points to the closest centroids."""
        distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self):
        """Updates centroids to be the mean of the points within each cluster."""
        new_centroids = np.zeros((self.n_clusters, self.n_variables))
        for i in range(self.n_clusters):
            points_in_cluster = self.X[self.labels_ == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = points_in_cluster.mean(axis=0)
        self.cluster_centers_ = new_centroids

    def predict(self, X_new):
        """
        Predict the closest cluster each sample in X_new belongs to.

        Args:
            X_new (np.ndarray): New data to predict.

        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self._assign_labels(X_new)

    def _get_summary_data(self):
        """Returns a dictionary containing summary data for the model."""
        summary = self._create_general_summary(
            self.n_variables,
            self.n_objects,
            No_Clusters=f"{self.n_clusters}",
            Max_Iterations=f"{self.max_iter}"
        )
        
        if self.cluster_centers_ is not None:
            cluster_table = [["Cluster"] + [f"Center-{i+1}" for i in range(self.n_variables)]]
            for i, center in enumerate(self.cluster_centers_):
                row = [f"Cluster {i}"] + [f"{val:.3f}" for val in center]
                cluster_table.append(row)
            summary["tables"] = {"Cluster Centers": cluster_table}

        return summary
