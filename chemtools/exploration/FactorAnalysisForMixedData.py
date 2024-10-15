"""
chemtools.exploration.FactorAnalysisForMixedData
--------------------------------------------------

This module provides the FactorAnalysisForMixedData class for performing 
Factor Analysis for Mixed Data (FAMD) on datasets containing both 
continuous and categorical variables. 

FAMD is an extension of Principal Component Analysis (PCA) specifically 
designed to handle mixed data types. It provides a way to explore the 
underlying structure in data where variables are of mixed nature.

Example usage:
>>> from chemtools.exploration import FactorAnalysisForMixedData
>>> # Assuming 'X' is your dataset with continuous and categorical variables
>>> famd = FactorAnalysisForMixedData(X) 
>>> famd.fit() 
>>> # ... Access results (e.g., famd.T, famd.V_ordered, famd.L_ordered) 
>>> # ... and use plotting methods
"""

import numpy as np

from chemtools.utility import reorder_array
from chemtools.utility import random_colorHEX
from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.base import BaseModel


class FactorAnalysisForMixedData(BaseModel):
    """
    Performs Factor Analysis for Mixed Data (FAMD) on a dataset.

    Attributes:
        model_name (str): Name of the model.
        X (np.ndarray): Input data (mixed data types).
        variables (np.ndarray): Names of the variables.
        objects (np.ndarray): Names of the objects.
        n_variables (int): Number of variables.
        n_objects (int): Number of objects.
        variables_colors (list): List of colors for the variables.
        objects_colors (list): List of colors for the objects.
        cov_mat (np.ndarray): Covariance matrix used for FAMD.
        V (np.ndarray): Eigenvalues of the covariance matrix.
        L (np.ndarray): Eigenvectors of the covariance matrix.
        order (np.ndarray): Order of the eigenvalues.
        V_ordered (np.ndarray): Ordered eigenvalues.
        L_ordered (np.ndarray): Ordered eigenvectors.
        PC_index (np.ndarray): Principal component indices.
        n_components (int): Number of principal components to retain.
        V_reduced (np.ndarray): Reduced eigenvalues.
        W (np.ndarray): Matrix of eigenvectors for the reduced components.
        T (np.ndarray): Transformed data in the reduced space.
    """

    def __init__(self, X, variables_names=None, objects_names=None):
        self.model_name = "Factor Analysis for Mixed Data"
        self.X = X
        self.variables = set_variables_names(variables_names, X)
        self.objects = set_objects_names(objects_names, X)
        self.n_variables = self.X.shape[1]
        self.n_objects = self.X.shape[0]
        self.variables_colors = self.change_variables_colors()
        self.objects_colors = self.change_objects_colors()

    def fit(self):
        """
        Fits the FAMD model to the data.
        """
        try:
            # Standardize continuous variables
            X_continuous_normalized = (self.X - np.mean(self.X, axis=0)) / np.std(
                self.X, axis=0
            )

            # Combine standardized continuous and categorical data
            X_combined = np.hstack((X_continuous_normalized, self.X))

            # Covariance matrix of the combined data
            self.cov_mat = np.cov(X_combined, rowvar=False)

            # Eigendecomposition
            self.V, self.L = np.linalg.eigh(self.cov_mat)

            # Ordering of eigenvalues and eigenvectors
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]
            self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
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
            self.T = np.dot(self.X, self.W)
        except AttributeError as e:
            print(f"Attribute error: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")
