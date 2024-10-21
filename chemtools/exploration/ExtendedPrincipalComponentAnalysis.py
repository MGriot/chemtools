"""
chemtools.exploration.ExtendedPrincipalComponentAnalysis
---------------------------------------------------------

This module provides the ExtendedPrincipalComponentAnalysis class for performing 
Extended Principal Component Analysis (XPCA) on mixed data (both discrete 
and continuous variables).

XPCA utilizes the Gaussian copula to model the dependence structure 
between variables while using nonparametric marginals to account for the 
different nature of the data. This allows for a more robust analysis 
compared to traditional PCA when dealing with mixed data types.

Example usage:
>>> from chemtools.exploration import ExtendedPrincipalComponentAnalysis
>>> # Assuming 'X' is your dataset (numpy array or pandas DataFrame)
>>> xpca = ExtendedPrincipalComponentAnalysis(X) 
>>> xpca.fit(autoscaling=True)  # Consider autoscaling for mixed data
>>> xpca.reduction(n_components=2) # Reduce to 2 dimensions
>>> # ... Access results (e.g., xpca.T, xpca.V_reduced, xpca.W) 
>>> # ... and use plotting methods
"""

import numpy as np
from scipy.stats import norm

from chemtools.utility import reorder_array
from chemtools.utility import random_colorHEX
from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.base import BaseModel


class ExtendedPrincipalComponentAnalysis(BaseModel):
    """
    Performs Extended Principal Component Analysis (XPCA) on mixed data.

    Attributes:
        model_name (str): Name of the model.
        X (np.ndarray): Input data.
        variables (np.ndarray): Names of the variables.
        objects (np.ndarray): Names of the objects.
        n_variables (int): Number of variables.
        n_objects (int): Number of objects.
        variables_colors (list): List of colors for the variables.
        objects_colors (list): List of colors for the objects.
        autoscaling (bool): Indicates whether to autoscale the data.
        mean (np.ndarray): Mean of the data (if autoscaling is True).
        std (np.ndarray): Standard deviation of the data (if autoscaling is True).
        X_autoscaled (np.ndarray): Autoscaled data (if autoscaling is True).
        U (np.ndarray): Matrix of nonparametric marginals.
        correlation_matrix (np.ndarray): Covariance matrix of the copula.
        V (np.ndarray): Eigenvalues of the covariance matrix.
        L (np.ndarray): Eigenvectors of the covariance matrix.
        order (np.ndarray): Order of the eigenvalues.
        V_ordered (np.ndarray): Ordered eigenvalues.
        L_ordered (np.ndarray): Ordered eigenvectors.
        index (np.ndarray): Principal component indices.
        n_components (int): Number of principal components to retain.
        V_reduced (np.ndarray): Reduced eigenvalues.
        W (np.ndarray): Matrix of eigenvectors for the reduced components.
        T (np.ndarray): Transformed data in the reduced space.

    """

    def __init__(self, X, variables_names=None, objects_names=None):
        """
        Initializes the XPCA model.

        Args:
            X (np.ndarray): Input data matrix.
            variables_names (list, optional): List of variable names. Defaults to None.
            objects_names (list, optional): List of object names. Defaults to None.
        """
        self.model_name = "Extended Principal Component Analysis"
        self.X = X
        self.variables = set_variables_names(variables_names, X)
        self.objects = set_objects_names(objects_names, X)
        self.n_variables = self.X.shape[1]
        self.n_objects = self.X.shape[0]
        self.variables_colors = self.change_variables_colors()
        self.objects_colors = self.change_objects_colors()
        self.autoscaling = False

    def fit(self, autoscaling=False):
        """
        Fit the XPCA model to the data.

        Args:
            autoscaling (bool, optional): If True, autoscale the data before applying XPCA. Defaults to False.
        """
        try:
            self.autoscaling = autoscaling
            if autoscaling:
                # Calculate mean and standard deviation
                self.mean = np.mean(self.X, axis=0)
                self.std = np.std(self.X, axis=0)
                # Check for zero standard deviation
                zero_std_columns = np.where(self.std == 0)[0]
                if zero_std_columns.size > 0:
                    raise ValueError(
                        f"The standard deviation contains zero in the columns: {zero_std_columns}. Consider removing the variables or setting autoscaling to False."
                    )
                # Autoscale the data
                self.X_autoscaled = (self.X - self.mean) / self.std
                X_to_use = self.X_autoscaled
            else:
                X_to_use = self.X

            # Calculate nonparametric marginals
            self.U = np.array(
                [norm.cdf(X_to_use[:, i]) for i in range(X_to_use.shape[1])]
            ).T

            # Calculate the covariance matrix of the copula
            self.correlation_matrix = np.cov(self.U, rowvar=False)

            # Calculate eigenvalues and eigenvectors
            self.V, self.L = np.linalg.eigh(self.correlation_matrix)
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]
            self.index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
        except np.linalg.LinAlgError as e:
            print(f"Error during the calculation of eigenvalues and eigenvectors: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")

    def change_variables_colors(self):
        """Generates random colors for the variables."""
        return random_colorHEX(self.n_variables)

    def change_objects_colors(self):
        """Generates random colors for the objects."""
        return random_colorHEX(self.n_objects)

    def reduction(self, n_components=2):
        """
        Reduce the dimensionality of the data using the specified number of components.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
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
