import numpy as np
from scipy.stats import norm
from .PrincipalComponentAnalysis import PrincipalComponentAnalysis


class ExtendedPrincipalComponentAnalysis(PrincipalComponentAnalysis):
    """
    Extended Principal Component Analysis (XPCA) class that extends the traditional PCA
    to handle both discrete and continuous variables using Gaussian copula and nonparametric marginals.

    Attributes:
        autoscaling (bool): Indicates whether to autoscale the data.
        mean (np.ndarray): Mean of the data.
        std (np.ndarray): Standard deviation of the data.
        X_autoscaled (np.ndarray): Autoscaled data.
        correlation_matrix (np.ndarray): Covariance matrix of the copula.
        V (np.ndarray): Eigenvalues of the covariance matrix.
        L (np.ndarray): Eigenvectors of the covariance matrix.
        order (np.ndarray): Order of the eigenvalues.
        V_ordered (np.ndarray): Ordered eigenvalues.
        L_ordered (np.ndarray): Ordered eigenvectors.
        PC_index (np.ndarray): Principal component indices.
        V_reduced (np.ndarray): Reduced eigenvalues.
        W (np.ndarray): Matrix of eigenvectors for the reduced components.
        T (np.ndarray): Transformed data in the reduced space.
    """

    def fit(self, autoscaling=False):
        """
        Fit the XPCA model to the data.

        Parameters:
            autoscaling (bool): If True, autoscale the data before applying XPCA.
        """
        try:
            self.autoscaling = autoscaling
            if autoscaling:
                # Calculate mean and standard deviation
                self.mean = np.mean(self.X, axis=0)
                self.std = np.std(self.X, axis=0)
                # Check for zero standard deviation and identify problematic columns
                zero_std_columns = np.where(self.std == 0)[0]
                if zero_std_columns.size > 0:
                    raise ValueError(
                        f"The standard deviation contains zero in the columns: {zero_std_columns}"
                    )
                # Autoscale the data
                self.X_autoscaled = (self.X - self.mean) / self.std
                X_to_use = self.X_autoscaled
            else:
                X_to_use = self.X

            # Calculate nonparametric marginals
            U = np.array([norm.cdf(X_to_use[:, i]) for i in range(X_to_use.shape[1])]).T

            # Calculate the covariance matrix of the copula
            self.correlation_matrix = np.cov(U, rowvar=False)

            # Calculate eigenvalues and eigenvectors
            self.V, self.L = np.linalg.eigh(self.correlation_matrix)
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]
            self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
        except np.linalg.LinAlgError as e:
            print(f"Error during the calculation of eigenvalues and eigenvectors: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")

    def reduction(self, n_components=2):
        """
        Reduce the dimensionality of the data using the specified number of components.

        Parameters:
            n_components (int): Number of principal components to retain.
        """
        self.n_component = n_components
        try:
            # Select the top n_components eigenvalues and eigenvectors
            self.V_reduced = self.V_ordered[:n_components]
            self.W = self.L_ordered[:, :n_components]
            # Transform the data into the reduced space
            if self.autoscaling:
                self.T = np.dot(self.X_autoscaled, self.W)
            else:
                self.T = np.dot(self.X, self.W)
        except AttributeError as e:
            print(f"Attribute error: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")
