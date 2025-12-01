import numpy as np
from chemtools.base.base_models import BaseModel
from chemtools.utils.data import initialize_names_and_counts

class LinearDiscriminantAnalysis(BaseModel):
    """
    Performs Linear Discriminant Analysis (LDA).

    LDA is a supervised classification method that finds linear combinations of features
    to separate two or more classes. It maximizes the ratio of between-class variance
    to within-class variance.

    Attributes:
        n_components (int): The number of discriminant components to retain.
        W (np.ndarray): The transformation matrix (eigenvectors).
        eigenvalues_ (np.ndarray): The eigenvalues corresponding to the discriminant components.
    
    References:
        - https://en.wikipedia.org/wiki/Linear_discriminant_analysis
    """

    def __init__(self, n_components=None):
        super().__init__()
        self.model_name = "Linear Discriminant Analysis"
        self.method = "LDA"
        self.n_components = n_components
        self.W = None
        self.eigenvalues_ = None
        self._class_means = None
        self._class_priors = None

    def fit(self, X, y, variables_names=None, objects_names=None):
        """
        Fits the LDA model to the provided data and class labels.

        Args:
            X (np.ndarray): The input data (n_samples, n_features).
            y (np.ndarray): The class labels (n_samples,).
            variables_names (list, optional): Names of the variables. Defaults to None.
            objects_names (list, optional): Names of the objects. Defaults to None.
        """
        self.X = X
        self.y = y
        self.variables, self.objects, self.n_variables, self.n_objects = (
            initialize_names_and_counts(X, variables_names, objects_names)
        )

        class_labels = np.unique(y)
        n_classes = len(class_labels)
        
        if self.n_components is None:
            self.n_components = min(self.n_variables, n_classes - 1)
        if self.n_components > min(self.n_variables, n_classes - 1):
            raise ValueError("n_components cannot be greater than min(n_features, n_classes - 1)")

        # 1. Compute within-class scatter matrix (S_W)
        S_W = np.zeros((self.n_variables, self.n_variables))
        self._class_means = []
        self._class_priors = []
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self._class_means.append(mean_c)
            self._class_priors.append(len(X_c) / self.n_objects)
            S_W += (X_c - mean_c).T @ (X_c - mean_c)

        # 2. Compute between-class scatter matrix (S_B)
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((self.n_variables, self.n_variables))
        for i, c in enumerate(class_labels):
            n_c = X[y == c].shape[0]
            mean_c = self._class_means[i]
            mean_diff = (mean_c - overall_mean).reshape(self.n_variables, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # 3. Solve the generalized eigenvalue problem
        # inv(S_W) @ S_B
        A = np.linalg.inv(S_W) @ S_B
        
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Sort eigenvectors by eigenvalues in descending order
        order = np.argsort(eigenvalues.real)[::-1]
        self.eigenvalues_ = eigenvalues[order].real
        eigenvectors = eigenvectors[:, order].real

        # 4. Store the transformation matrix W
        self.W = eigenvectors[:, :self.n_components]

        self._class_means = np.array(self._class_means)
        self._class_priors = np.array(self._class_priors)

        return self

    def transform(self, X):
        """
        Projects the data onto the LDA components.

        Args:
            X (np.ndarray): The data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        if self.W is None:
            raise RuntimeError("The model has not been fitted yet.")
        return X @ self.W

    def fit_transform(self, X, y, **kwargs):
        """
        Fit the model and transform the data in one step.
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)

    def predict(self, X_new):
        """
        Predict the class labels for new data.

        This implementation uses the class centroids in the transformed space.
        A new point is assigned to the class of the closest centroid.

        Args:
            X_new (np.ndarray): New data to predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.W is None:
            raise RuntimeError("The model must be fitted before predicting.")
        
        X_transformed = self.transform(X_new)
        centroids_transformed = self._class_means @ self.W
        
        predictions = []
        for sample in X_transformed:
            distances = np.linalg.norm(sample - centroids_transformed, axis=1)
            predicted_class_index = np.argmin(distances)
            predictions.append(np.unique(self.y)[predicted_class_index])
            
        return np.array(predictions)

    def _get_summary_data(self):
        """Returns a dictionary containing summary data for the model."""
        if self.eigenvalues_ is None:
            return {}

        total_variance = np.sum(self.eigenvalues_)
        explained_variance_ratio = self.eigenvalues_ / total_variance
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        summary = self._create_general_summary(
            self.n_variables,
            self.n_objects,
            No_Components=f"{self.n_components}",
            No_Classes=f"{len(np.unique(self.y))}"
        )

        eigenvalue_table = [
            ["Component", "Eigenvalue", "Explained Variance (%)", "Cumulative Variance (%)"]
        ]
        for i in range(len(self.eigenvalues_)):
            eigenvalue_table.append([
                f"LD{i+1}",
                f"{self.eigenvalues_[i]:.4f}",
                f"{explained_variance_ratio[i] * 100:.2f}",
                f"{cumulative_explained_variance[i] * 100:.2f}"
            ])
        
        summary["tables"] = {"LDA Eigenvalues": eigenvalue_table}
        return summary
