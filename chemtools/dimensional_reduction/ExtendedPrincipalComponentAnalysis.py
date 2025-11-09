"""
chemtools.dimensional_reduction.ExtendedPrincipalComponentAnalysis
-----------------------------------------------------------------

This module provides the ExtendedPrincipalComponentAnalysis class for performing 
Extended Principal Component Analysis (XPCA) on mixed data (both discrete 
and continuous variables).

XPCA utilizes the Gaussian copula to model the dependence structure 
between variables while using nonparametric marginals to account for the 
different nature of the data. This allows for a more robust analysis 
compared to traditional PCA when dealing with mixed data types.

Example usage:
>>> from chemtools.dimensional_reduction import ExtendedPrincipalComponentAnalysis
>>> # Assuming 'X' is your dataset (numpy array or pandas DataFrame)
>>> xpca = ExtendedPrincipalComponentAnalysis(X) 
>>> xpca.fit(autoscaling=True)
>>> xpca.reduction(n_components=2)
>>> # Access results: xpca.T, xpca.W, etc.
"""

import numpy as np
from scipy.stats import norm

from chemtools.utils import HarmonizedPaletteGenerator
from chemtools.utils.data import set_objects_names, set_variables_names
from chemtools.base.base_models import BaseModel


class ExtendedPrincipalComponentAnalysis(BaseModel):
    """
    Performs Extended Principal Component Analysis (XPCA) on mixed data.

    XPCA is a variant of PCA that uses Gaussian copulas to model the 
    dependence structure between variables. This makes it suitable for
    datasets containing both continuous and discrete variables.

    Attributes:
        model_name (str): Name of the model.
        X (np.ndarray): Input data.
        T (np.ndarray): Transformed data in the reduced space (scores).
        W (np.ndarray): Matrix of eigenvectors for the reduced components (loadings).
        V_ordered (np.ndarray): Ordered eigenvalues.
        
    References:
        - https://en.wikipedia.org/wiki/Copula_(probability_theory)
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
            autoscaling (bool, optional): If True, autoscale the data. Defaults to False.
        """
        self.autoscaling = autoscaling
        X_to_use = self.X
        if autoscaling:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0)
            if np.any(self.std == 0):
                raise ValueError("Cannot autoscale data with zero standard deviation in one or more columns.")
            self.X_autoscaled = (self.X - self.mean) / self.std
            X_to_use = self.X_autoscaled

        # Calculate nonparametric marginals (U-matrix)
        self.U = np.array([norm.cdf(X_to_use[:, i]) for i in range(X_to_use.shape[1])]).T

        # Calculate the correlation matrix of the copula
        self.correlation_matrix = np.corrcoef(self.U, rowvar=False)

        # Eigenvalue decomposition
        self.V, self.L = np.linalg.eigh(self.correlation_matrix)
        self.order = np.argsort(self.V)[::-1]
        self.V_ordered = self.V[self.order]
        self.L_ordered = self.L[:, self.order]
        self.index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])

    def change_variables_colors(self):
        """Generates colors for the variables."""
        return HarmonizedPaletteGenerator(self.n_variables).generate()

    def change_objects_colors(self):
        """Generates colors for the objects."""
        return HarmonizedPaletteGenerator(self.n_objects).generate()

    def reduction(self, n_components=2):
        """
        Reduce the dimensionality of the data.

        Args:
            n_components (int): Number of principal components to retain.
        """
        if not hasattr(self, 'V_ordered'):
            raise RuntimeError("Fit method must be called before reduction.")
        
        self.n_components = n_components
        self.V_reduced = self.V_ordered[:n_components]
        self.W = self.L_ordered[:, :n_components]
        
        X_to_transform = self.X_autoscaled if self.autoscaling else self.X
        self.T = np.dot(X_to_transform, self.W)

    def _get_summary_data(self):
        """Returns a dictionary containing summary data for the model."""
        if not hasattr(self, 'V_ordered'):
            return {}

        explained_variance = self.V_ordered / np.sum(self.V_ordered)
        cumulative_variance = np.cumsum(explained_variance)

        summary = self._create_general_summary(
            self.n_variables,
            self.n_objects,
            No_Components=f"{self.n_components}"
        )

        eigenvalue_table = [
            ["Component", "Eigenvalue", "Explained Variance (%)", "Cumulative Variance (%)"]
        ]
        for i in range(len(self.V_ordered)):
            eigenvalue_table.append([
                f"PC{i+1}",
                f"{self.V_ordered[i]:.4f}",
                f"{explained_variance[i] * 100:.2f}",
                f"{cumulative_variance[i] * 100:.2f}"
            ])
        
        summary["tables"] = {"Eigenvalues": eigenvalue_table}
        return summary