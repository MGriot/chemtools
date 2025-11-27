# ---------------------------------------------------------
# Author: MGriot
# Date: 21/08/2024
#
# This code is protected by copyright and cannot be
# copied, modified, or used without the explicit
# permission of the author. All rights reserved.
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import f, norm
from sklearn.cross_decomposition import PLSCanonical

from chemtools.preprocessing import autoscaling
from chemtools.preprocessing.matrix_standard_deviation import matrix_standard_deviation
from chemtools.preprocessing import correlation_matrix
from chemtools.preprocessing import diagonalized_matrix
from chemtools.utils import reorder_array, HarmonizedPaletteGenerator
from chemtools.utils.data import (
    initialize_names_and_counts,
    set_variables_names,
    set_objects_names,
)
from chemtools.base.base_models import BaseModel
from chemtools.exploration import PrincipalComponentAnalysis


class PrincipalComponentClassificationAnalysis(BaseModel):
    """
    Performs Principal Component-based Classification Analysis (PCCA).

    This method combines Principal Component Analysis (PCA) with an approach
    similar to Canonical Correlation Analysis (CCA) to perform classification.
    It is particularly useful for high-dimensional data where classes are
    expected to be separable.

    The process involves:
    1. Performing PCA on the input data `X` to get scores `T`.
    2. Creating a dummy-coded matrix `Y` from the class labels.
    3. Finding a transformation `Wx` that maximizes the correlation between
       the PCA scores `T` and the class information `Y`.
    4. The final scores `U` are the projection of `T` onto this new space.

    Attributes:
        model_name (str): The name of the model.
        pca_model (PrincipalComponentAnalysis): The fitted PCA model.
        n_components (int): Number of components retained.
        T (np.ndarray): Scores from the PCA step.
        U (np.ndarray): Final transformed scores in the classification space.
        Wx (np.ndarray): The transformation matrix applied to the PCA scores.

    References:
        - This method is a custom implementation combining PCA and CCA principles.
          For background, see:
          https://en.wikipedia.org/wiki/Principal_component_analysis
          https://en.wikipedia.org/wiki/Canonical_correlation
    """

    def __init__(self):
        self.model_name = "Principal Component-based Classification Analysis"

    def fit(
        self,
        X,
        y,
        n_components=2,
        variables_names=None,
        objects_names=None,
    ):
        """
        Fits the PCCA model to the provided data and class labels.

        Args:
            X (np.ndarray): The input data (n_samples, n_features).
            y (np.ndarray): The class labels (n_samples,).
            n_components (int, optional): The number of components to retain. Defaults to 2.
            variables_names (list, optional): Names of the variables. Defaults to None.
            objects_names (list, optional): Names of the objects. Defaults to None.
        """
        self.X = X
        self.y = y
        self.variables, self.objects, self.n_variables, self.n_objects = (
            initialize_names_and_counts(X, variables_names, objects_names)
        )
        self.variables_colors = self.change_variables_colors()
        self.objects_colors = self.change_objects_colors()

        # Step 1: Perform PCA
        self.pca_model = PrincipalComponentAnalysis()
        self.pca_model.fit(self.X, variables_names, objects_names)
        self.pca_model.reduction(n_components)
        self.n_components = n_components
        self.T = self.pca_model.T

        # Step 2: Prepare class indicator matrix and correlation matrices
        Y_dummy = pd.get_dummies(self.y).to_numpy()
        Rxx = np.corrcoef(self.T, rowvar=False)
        Ryy = np.corrcoef(Y_dummy, rowvar=False)
        
        # Ensure Rxy is calculated correctly
        n_dummy_vars = Y_dummy.shape[1]
        full_corr = np.corrcoef(self.T, Y_dummy, rowvar=False)
        Rxy = full_corr[:self.n_components, self.n_components:]

        # Step 3: Solve the generalized eigenvalue problem
        # (Rxx^-1 * Rxy * Ryy^-1 * Ryx) * Wx = lambda * Wx
        try:
            Rxx_inv = np.linalg.inv(Rxx)
            Ryy_inv = np.linalg.inv(Ryy)
        except np.linalg.LinAlgError:
            raise RuntimeError("Could not invert correlation matrices. Check for collinearity.")

        A = Rxx_inv @ Rxy @ Ryy_inv @ Rxy.T
        eigenvalues, eigenvectors = np.linalg.eig(A)

        order = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues[order].real
        eigenvectors = eigenvectors[:, order].real

        self.Wx = eigenvectors
        self.U = self.T @ self.Wx

    def transform(self, X_test):
        """
        Transforms new data into the reduced PCCA space.

        Args:
            X_test (np.ndarray): The new data to transform.

        Returns:
            np.ndarray: The transformed data in the PCCA space.
        """
        if not hasattr(self, 'pca_model'):
            raise RuntimeError("The model must be fitted before transforming data.")
            
        T_test = self.pca_model.transform(X_test)
        U_test = T_test @ self.Wx
        return U_test

    def change_variables_colors(self):
        return HarmonizedPaletteGenerator(self.n_variables).generate()

    def change_objects_colors(self):
        return HarmonizedPaletteGenerator(self.n_objects).generate()

    def _get_summary_data(self):
        """Returns a dictionary containing summary data for the model."""
        summary = self._create_general_summary(
            self.n_variables,
            self.n_objects,
            Components=f"{self.n_components}",
        )

        if hasattr(self, "U"):
            variance_explained = np.var(self.U, axis=0)
            total_variance = np.sum(variance_explained)
            if total_variance > 0:
                prop_variance = variance_explained / total_variance * 100
                summary["additional_stats"] = {
                    f"Component {i+1} Variance (%)": f"{var:.2f}"
                    for i, var in enumerate(prop_variance)
                }
        return summary