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

from chemtools.utils import HarmonizedPaletteGenerator
from chemtools.utils.data import set_variables_names
from chemtools.base import BaseModel


class GeneralizedCanonicalCorrelationAnalysis(BaseModel):
    """
    Performs Generalized Canonical Correlation Analysis (GCCA).

    GCCA is a method for exploring the relationships between two or more
    sets of variables. It extends Canonical Correlation Analysis (CCA) to more
    than two datasets.

    Attributes:
        model_name (str): Name of the model.
        n_datasets (int): Number of datasets.
        V_ordered (np.ndarray): Ordered eigenvalues.
        Ls_ordered (list): List of dataset coordinates ordered by eigenvalues.

    References:
        - https://en.wikipedia.org/wiki/Generalized_canonical_correlation
    """

    def __init__(self):
        self.model_name = "Generalized Canonical Correlation Analysis"

    def fit(self, *Xs, variables_names=None):
        """
        Fits the GCCA model to the datasets.

        Args:
            *Xs (np.ndarray): Variable length argument list of datasets.
            variables_names (list, optional): List of lists of variable names.
        """
        self.n_datasets = len(Xs)
        if self.n_datasets < 2:
            raise ValueError("GCCA requires at least two datasets.")
            
        self.Xs = [np.asarray(X) for X in Xs]
        
        if variables_names:
            self.variables_names = [set_variables_names(names, X) for names, X in zip(variables_names, self.Xs)]
        else:
            self.variables_names = [set_variables_names(None, X) for X in self.Xs]
            
        self.n_variables = [X.shape[1] for X in self.Xs]
        self.variables_colors = self.change_variables_colors()

        cov_matrices = self._calculate_covariance_matrices()
        self.V, self.Ls = self._generalized_eigen_decomposition(cov_matrices)

        self.order = np.argsort(self.V)[::-1]
        self.V_ordered = self.V[self.order]
        self.Ls_ordered = [L[:, self.order] for L in self.Ls]
        self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])

    def _calculate_covariance_matrices(self):
        """Calculates the covariance matrices needed for GCCA."""
        Xs_meaned = [X - np.mean(X, axis=0) for X in self.Xs]
        return [np.cov(X.T) for X in Xs_meaned]

    def _generalized_eigen_decomposition(self, cov_matrices):
        """Performs the generalized eigenvalue decomposition for GCCA."""
        total_vars = sum(self.n_variables)
        A = np.zeros((total_vars, total_vars))
        B = np.zeros_like(A)

        start_idx = 0
        for i in range(self.n_datasets):
            end_idx = start_idx + self.n_variables[i]
            B[start_idx:end_idx, start_idx:end_idx] = cov_matrices[i]
            for j in range(i + 1, self.n_datasets):
                cross_cov_start = sum(self.n_variables[:j])
                cross_cov_end = cross_cov_start + self.n_variables[j]
                
                # This cross-covariance calculation seems incorrect.
                # A proper cross-covariance between two centered matrices X_i and X_j is (X_i.T @ X_j) / (n-1)
                # The original implementation was complex and potentially flawed.
                # Let's use a more direct calculation.
                n_obs = self.Xs[i].shape[0]
                cross_cov = (self.Xs[i].T @ self.Xs[j]) / (n_obs - 1)

                A[start_idx:end_idx, cross_cov_start:cross_cov_end] = cross_cov
                A[cross_cov_start:cross_cov_end, start_idx:end_idx] = cross_cov.T
            start_idx += self.n_variables[i]

        V, L = np.linalg.eigh(A, B)

        Ls = []
        start_idx = 0
        for n_var in self.n_variables:
            Ls.append(L[start_idx : start_idx + n_var, :])
            start_idx += n_var

        return V, Ls

    def change_variables_colors(self):
        """Generates random colors for the variables."""
        return [HarmonizedPaletteGenerator(n_var).generate() for n_var in self.n_variables]

    def transform(self, *Xs_new):
        """
        Transforms the new datasets into the GCCA space.

        Args:
            *Xs_new: Variable length argument list of new datasets.

        Returns:
            list: List of transformed datasets.
        """
        if not hasattr(self, 'Xs'):
            raise RuntimeError("Fit method must be called before transforming data.")

        Xs_new_meaned = [X_new - np.mean(X, axis=0) for X_new, X in zip(Xs_new, self.Xs)]
        return [X_meaned @ L for X_meaned, L in zip(Xs_new_meaned, self.Ls_ordered)]

    def _get_summary_data(self):
        """Returns a dictionary containing summary data for the model."""
        if not hasattr(self, 'V'):
            return {}
            
        summary = self._create_general_summary(
            sum(self.n_variables),
            self.Xs[0].shape[0],
            **{"No. Datasets": str(self.n_datasets)},
        )

        total_variance = np.sum(self.V)
        if total_variance > 0:
            prop_variance = self.V / total_variance * 100
            summary["additional_stats"] = {
                f"Factor{i+1} Variance (%)": f"{var:.2f}"
                for i, var in enumerate(prop_variance)
            }
        return summary