"""
chemtools.exploration.GeneralizedCanonicalCorrelationAnalysis
-------------------------------------------------------------

This module provides the GeneralizedCanonicalCorrelationAnalysis class for 
performing Generalized Canonical Correlation Analysis (GCCA) on two or more 
datasets. 

GCCA is a statistical technique used to identify relationships between 
multiple datasets. It finds linear combinations of variables in each 
dataset that maximize the correlation between them.

Example usage:
>>> from chemtools.exploration import GeneralizedCanonicalCorrelationAnalysis
>>> # Assuming 'X1', 'X2', ... are your datasets
>>> gcca = GeneralizedCanonicalCorrelationAnalysis() 
>>> gcca.fit(X1, X2, ...)
>>> # ... Access results and use plotting methods
"""

import numpy as np

from chemtools.utility import reorder_array
from chemtools.utility import random_colorHEX
from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.base import BaseModel


class GeneralizedCanonicalCorrelationAnalysis(BaseModel):
    """
    Performs Generalized Canonical Correlation Analysis (GCCA) on two or
    more datasets.

    Attributes:
        model_name (str): Name of the model.
        n_datasets (int): Number of datasets.
        Xs (list): List of input datasets (np.ndarray).
        variables_names (list): List of lists of variable names for each dataset.
        n_variables (list): List of the number of variables in each dataset.
        variables_colors (list): List of lists of colors for variables in each dataset.
        V (np.ndarray): Eigenvalues of the decomposition.
        Ls (list): List of left singular vectors for each dataset (dataset coordinates in factor space).
        order (np.ndarray): Indices that order eigenvalues in descending order.
        V_ordered (np.ndarray): Ordered eigenvalues.
        Ls_ordered (list): List of dataset coordinates ordered according to eigenvalues.
        PC_index (np.ndarray): Names of the principal components.
    """

    def __init__(self):
        self.model_name = "Generalized Canonical Correlation Analysis"

    def fit(self, *Xs, variables_names=None):
        """
        Fits the GCCA model to the datasets.

        Args:
            *Xs (np.ndarray): Variable length argument list of datasets.
            variables_names (list, optional): List of lists of variable names for each dataset. Defaults to None.
        """
        # 1. Handle datasets and their names
        self.n_datasets = len(Xs)
        self.Xs = Xs
        self.variables_names = (
            [
                set_variables_names(var_names, X)
                for var_names, X in zip(variables_names, self.Xs)
            ]
            if variables_names
            else [set_variables_names(None, X) for X in self.Xs]
        )
        self.n_variables = [X.shape[1] for X in Xs]
        self.variables_colors = self.change_variables_colors()

        # 2. Calculate covariance matrices
        cov_matrices = self._calculate_covariance_matrices()

        # 3. Perform Generalized Eigenvalue Decomposition
        self.V, self.Ls = self._generalized_eigen_decomposition(cov_matrices)

        # 4. Ordering
        self.order = np.argsort(self.V)[::-1]
        self.V_ordered = self.V[self.order]
        self.Ls_ordered = [L[:, self.order] for L in self.Ls]
        self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])

    def _calculate_covariance_matrices(self):
        """Calculates the covariance matrices needed for GCCA."""
        # Mean-center the datasets
        Xs_meaned = [X - np.mean(X, axis=0) for X in self.Xs]

        # Calculate covariance matrices
        cov_matrices = [np.cov(X.T) for X in Xs_meaned]

        return cov_matrices

    def _generalized_eigen_decomposition(self, cov_matrices):
        """Performs the generalized eigenvalue decomposition for GCCA."""
        # Initialize matrices for the generalized eigenvalue problem
        A = np.zeros((sum(self.n_variables), sum(self.n_variables)))
        B = np.zeros_like(A)

        # Construct A and B matrices
        start_idx = 0
        for i in range(self.n_datasets):
            end_idx = start_idx + self.n_variables[i]
            B[start_idx:end_idx, start_idx:end_idx] = cov_matrices[i]
            for j in range(i + 1, self.n_datasets):
                cross_cov = np.cov(self.Xs[i].T, self.Xs[j].T)[
                    : self.n_variables[i], self.n_variables[i] :
                ]
                A[start_idx:end_idx, start_idx + self.n_variables[i] :] = cross_cov
                A[start_idx + self.n_variables[i] :, start_idx:end_idx] = cross_cov.T
            start_idx += self.n_variables[i]

        # Solve the generalized eigenvalue problem
        V, L = np.linalg.eigh(A, B)

        # Separate eigenvectors for each dataset
        Ls = []
        start_idx = 0
        for n_var in self.n_variables:
            Ls.append(L[start_idx : start_idx + n_var, :])
            start_idx += n_var

        return V, Ls

    def change_variables_colors(self):
        """Generates random colors for the variables."""
        return [random_colorHEX(n_var) for n_var in self.n_variables]
