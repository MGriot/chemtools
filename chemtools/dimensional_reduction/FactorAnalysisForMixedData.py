"""
chemtools.dimensional_reduction.FactorAnalysisOfMixedData
--------------------------------------------------

This module provides the FactorAnalysisOfMixedData class for performing 
Factor Analysis of Mixed Data (FAMD) on datasets containing both 
quantitative and qualitative variables. 

FAMD is a principal component method dedicated to exploring data with both
quantitative and qualitative variables. It balances the influence of both
sets of variables in the analysis.

This implementation is inspired by the work of Jérôme Pagès and the
FactoMineR R package.

References:
- Pagès, J. (2004). Analyse factorielle de données mixtes. Revue de Statistique Appliquée, 52(4), 93-111.
- https://en.wikipedia.org/wiki/Factor_analysis_of_mixed_data

Example usage:
>>> from chemtools.dimensional_reduction import FactorAnalysisOfMixedData
>>> import pandas as pd
>>> # Assuming 'X' is your pandas DataFrame with mixed data
>>> # and 'qualitative_variables' is a list of column names for categorical variables
>>> famd = FactorAnalysisOfMixedData(n_components=2) 
>>> famd.fit(X, qualitative_variables=['cat_var1', 'cat_var2'])
>>> # ... Access results (e.g., famd.T, famd.eigenvalues) 
>>> # ... and use plotting methods from DimensionalityReductionPlot
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from .base import DimensionalityReduction
from chemtools.utils.data import initialize_names_and_counts


class FactorAnalysisOfMixedData(DimensionalityReduction):
    """
    Performs Factor Analysis of Mixed Data (FAMD).

    FAMD is a principal component method that allows the analysis of a data set 
    containing both quantitative and qualitative variables. It balances the
    influence of both types of variables by preprocessing them appropriately
    before performing a global Principal Component Analysis (PCA).

    Quantitative variables are standardized (mean-centered and scaled to unit variance).
    Qualitative variables are transformed using one-hot encoding and then scaled
    in a manner similar to Multiple Correspondence Analysis (MCA).

    Attributes:
        n_components (int): The number of components to keep.
        X (pd.DataFrame): The input data.
        qualitative_variables (list): List of qualitative variable names.
        quantitative_variables (list): List of quantitative variable names.
        eigenvalues (np.ndarray): The eigenvalues of the analysis.
        eigenvectors (np.ndarray): The eigenvectors of the analysis.
        T (np.ndarray): The scores (coordinates of individuals).
        W (np.ndarray): The loadings (coordinates of variables).
        
    References:
        - Pagès, J. (2004). Analyse factorielle de données mixtes. Revue de Statistique Appliquée, 52(4), 93-111.
        - https://en.wikipedia.org/wiki/Factor_analysis_of_mixed_data
    """

    def __init__(self, n_components: int = 2):
        """
        Initialize FactorAnalysisOfMixedData.

        Args:
            n_components (int, optional): Number of factors to retain. Defaults to 2.
        """
        super().__init__(n_components=n_components)
        self.model_name = "Factor Analysis of Mixed Data"
        self.method = "FAMD"
        self.qualitative_variables = None
        self.quantitative_variables = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.T = None
        self.W = None
        self._one_hot_encoder = None
        self._quantitative_scaler = None
        self._mca_scaler = None
        self._global_pca_mean = None

    def fit(self, X: pd.DataFrame, qualitative_variables: list):
        """
        Fits the FAMD model to the provided mixed data.

        Args:
            X (pd.DataFrame): The data to fit the model to.
            qualitative_variables (list): Names of the qualitative (categorical) columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        self.X = X
        self.qualitative_variables = qualitative_variables
        self.quantitative_variables = [col for col in X.columns if col not in qualitative_variables]

        self.variables, self.objects, self.n_variables, self.n_objects = initialize_names_and_counts(
            X, variables_names=list(X.columns), objects_names=list(X.index)
        )

        # 1. Preprocess quantitative variables (standardization)
        X_quant = self.X[self.quantitative_variables].values
        self._quantitative_scaler = {
            'mean': np.mean(X_quant, axis=0),
            'std': np.std(X_quant, axis=0)
        }
        X_quant_scaled = (X_quant - self._quantitative_scaler['mean']) / self._quantitative_scaler['std']

        # 2. Preprocess qualitative variables (MCA-like scaling)
        X_qual = self.X[self.qualitative_variables]
        self._one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        indicator_matrix = self._one_hot_encoder.fit_transform(X_qual)
        
        # Scale the indicator matrix
        p_k = np.mean(indicator_matrix, axis=0)
        self._mca_scaler = {'p_k': p_k}
        X_qual_scaled = indicator_matrix / np.sqrt(p_k)
        X_qual_scaled = X_qual_scaled - np.mean(X_qual_scaled, axis=0)


        # 3. Concatenate the processed data
        Z = np.concatenate([X_quant_scaled, X_qual_scaled], axis=1)
        
        # 4. Perform global PCA
        self._global_pca_mean = np.mean(Z, axis=0)
        Z_centered = Z - self._global_pca_mean
        
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(Z_centered, rowvar=False))
        
        # Sort eigenvalues and eigenvectors
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[order]
        self.eigenvectors = eigenvectors[:, order]

        # Set W and T attributes
        self.W = self.eigenvectors
        self.T = Z_centered @ self.W

        # Reduce dimensionality
        self.reduction(self.n_components)

        return self

    def reduction(self, n_components: int):
        """
        Reduce the dimensionality of the data.

        Args:
            n_components (int): The number of components to keep.
        """
        self.n_components = n_components
        self.W_reduced = self.W[:, :self.n_components]
        self.T_reduced = self.T[:, :self.n_components]
        # For compatibility with plotting tools
        self.V_ordered = self.eigenvalues
        self.L_ordered = self.eigenvectors
        self.T = self.T_reduced
        self.W = self.W_reduced


    def transform(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Transforms new data into the FAMD space.

        Args:
            X_new (pd.DataFrame): The new data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError("Input X_new must be a pandas DataFrame.")

        # Preprocess quantitative variables
        X_quant_new = X_new[self.quantitative_variables].values
        X_quant_new_scaled = (X_quant_new - self._quantitative_scaler['mean']) / self._quantitative_scaler['std']

        # Preprocess qualitative variables
        X_qual_new = X_new[self.qualitative_variables]
        indicator_matrix_new = self._one_hot_encoder.transform(X_qual_new)
        X_qual_new_scaled = indicator_matrix_new / np.sqrt(self._mca_scaler['p_k'])
        X_qual_new_scaled = X_qual_new_scaled - np.mean(X_qual_new_scaled, axis=0)

        # Concatenate
        Z_new = np.concatenate([X_quant_new_scaled, X_qual_new_scaled], axis=1)
        
        # Center and project
        Z_new_centered = Z_new - self._global_pca_mean
        
        return Z_new_centered @ self.W_reduced

    def _get_summary_data(self):
        """
        Returns a dictionary containing summary data for the model.
        """
        if self.eigenvalues is None:
            return {}

        explained_variance_ratio = self.eigenvalues / np.sum(self.eigenvalues)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        summary = self._create_general_summary(
            self.n_variables,
            self.n_objects,
            No_Components=f"{self.n_components}"
        )

        eigenvalue_table = [
            ["Component", "Eigenvalue", "Explained Variance (%)", "Cumulative Variance (%)"]
        ]
        for i in range(len(self.eigenvalues)):
            eigenvalue_table.append([
                f"PC{i+1}",
                f"{self.eigenvalues[i]:.4f}",
                f"{explained_variance_ratio[i] * 100:.2f}",
                f"{cumulative_explained_variance[i] * 100:.2f}"
            ])
        
        summary["tables"] = {"Eigenvalues": eigenvalue_table}
        return summary
