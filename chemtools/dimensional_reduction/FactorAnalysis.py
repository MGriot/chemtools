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
from scipy.linalg import eig, eigh

from chemtools.preprocessing import autoscaling
from chemtools.preprocessing.matrix_standard_deviation import (
    matrix_standard_deviation,
)
from chemtools.preprocessing import correlation_matrix
from chemtools.preprocessing import diagonalized_matrix
from chemtools.utils import reorder_array, HarmonizedPaletteGenerator
from chemtools.utils.data import (
    initialize_names_and_counts,
    set_objects_names,
    set_variables_names,
)
from .base import DimensionalityReduction


class FactorAnalysis(DimensionalityReduction):
    """
    Performs Factor Analysis (FA) on a dataset.

    Factor analysis is a statistical method used to describe variability among
    observed, correlated variables in terms of a potentially lower number of
    unobserved variables called factors.

    This class provides methods to fit the FA model, reduce dimensionality,
    and compute statistical metrics. It supports rotation methods like
    Varimax and Quartimax to improve interpretability.

    Attributes:
        model_name (str): The name of the model, "Factor Analysis".
        method (str): The method name, "FA".
        n_components (int): Number of factors to retain.
        rotation (str): Type of rotation applied ('varimax', 'quartimax', or None).
        W (np.ndarray): The factor loadings matrix.
        T (np.ndarray): The factor scores matrix.
        communalities (np.ndarray): The proportion of each variable's variance explained by the factors.
        res_var (np.ndarray): The residual variances (uniqueness) of the variables.

    References:
        - https://en.wikipedia.org/wiki/Factor_analysis
    """

    def __init__(self, n_components: int = 2, rotation: str = None):
        """
        Initialize FactorAnalysis.

        Args:
            n_components (int, optional): Number of factors to retain. Defaults to 2.
            rotation (str, optional): Type of rotation to apply. Options are None, 'varimax', 'quartimax'. Defaults to None.
        """
        super().__init__()
        self.n_components = n_components
        self.rotation = rotation
        self.model_name = "Factor Analysis"
        self.method = "FA"

    def fit(self, X, variables_names=None, objects_names=None):
        """
        Fits the FA model to the provided data.

        Args:
            X (np.ndarray): The data to fit the model to.
            variables_names (list, optional): Names of the variables. Defaults to None.
            objects_names (list, optional): Names of the objects. Defaults to None.
        """
        self.X = X
        self.variables, self.objects, self.n_variables, self.n_objects = (
            initialize_names_and_counts(X, variables_names, objects_names)
        )
        self.variables_colors = self.change_variables_colors()
        self.objects_colors = self.change_objects_colors()

        try:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0)
            zero_std_columns = np.where(self.std == 0)[0]
            if zero_std_columns.size > 0:
                raise ValueError(
                    f"The standard deviation contains zero in the columns: {zero_std_columns}"
                )
            self.X_autoscaled = (self.X - self.mean) / self.std
            self.correlation_matrix = np.corrcoef(self.X_autoscaled, rowvar=False)

            self.V, self.L = eigh(self.correlation_matrix)

            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]

            self.V_reduced = self.V_ordered[: self.n_components]
            self.PC_index = [f"F{i+1}" for i in range(self.n_components)]
            self.index = np.array([f"F{i+1}" for i in range(self.V.shape[0])])
            self.W = self.L_ordered[:, : self.n_components]

            self.T = self.X_autoscaled @ self.W

            self.communalities = np.sum(self.W**2, axis=1)
            self.res_var = 1 - self.communalities

            if self.rotation:
                self.rotate()

        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Linear algebra error during fit: {e}")
        except ValueError as e:
            raise ValueError(f"Input data error during fit: {e}")

    def transform(self, X_new):
        """
        Projects new data onto the factor space.

        Args:
            X_new (np.ndarray): The new data to transform.

        Returns:
            np.ndarray: The transformed data in the factor space.
        """
        X_new_autoscaled = (X_new - self.mean) / self.std
        return np.dot(X_new_autoscaled, self.W)

    def fit_transform(self, X, **kwargs):
        """
        Fit FactorAnalysis to X and transform X.
        """
        self.fit(X, **kwargs)
        return self.T

    def rotate(self):
        """
        Performs rotation on the factor loadings.
        """
        if self.rotation == "varimax":
            self.W = self._varimax_rotation(self.W)
        elif self.rotation == "quartimax":
            self.W = self._quartimax_rotation(self.W)
        else:
            if self.rotation is not None:
                raise ValueError(f"Unknown rotation: {self.rotation}")

    def _varimax_rotation(self, loadings, max_iter=100, tol=1e-6):
        """Performs varimax rotation on a matrix."""
        p, k = loadings.shape
        rotated_loadings = loadings.copy()
        for _ in range(max_iter):
            lambda_rot = rotated_loadings
            h2 = np.sum(lambda_rot**2, axis=1)
            U = lambda_rot**3 - (lambda_rot * h2[:, np.newaxis]) / p
            svd = np.linalg.svd(loadings.T @ U)
            T = svd[0] @ svd[2]
            rotated_loadings_new = loadings @ T
            if np.sum((rotated_loadings_new - rotated_loadings)**2) < tol:
                break
            rotated_loadings = rotated_loadings_new
        return rotated_loadings

    def _quartimax_rotation(self, loadings, max_iter=100, tol=1e-6):
        """Performs quartimax rotation on a matrix."""
        p, k = loadings.shape
        rotated_loadings = loadings.copy()
        for _ in range(max_iter):
            lambda_rot = rotated_loadings
            U = lambda_rot**3
            svd = np.linalg.svd(loadings.T @ U)
            T = svd[0] @ svd[2]
            rotated_loadings_new = loadings @ T
            if np.sum((rotated_loadings_new - rotated_loadings)**2) < tol:
                break
            rotated_loadings = rotated_loadings_new
        return rotated_loadings

    def change_variables_colors(self):
        """Generates random color codes for the variables."""
        return HarmonizedPaletteGenerator(self.n_variables).generate()

    def change_objects_colors(self):
        """Generates random color codes for the objects."""
        return HarmonizedPaletteGenerator(self.n_objects).generate()

    def _get_summary_data(self):
        """Returns a dictionary of data for the summary."""
        loadings_table = [
            ["Variable"] + [f"Factor {j+1}" for j in range(self.n_components)]
        ]
        for i in range(self.n_variables):
            row = [self.variables[i]] + [f"{self.W[i, j]:.2f}" for j in range(self.n_components)]
            loadings_table.append(row)

        summary = self._create_general_summary(
            self.n_variables, self.n_objects, No_Components=f"{self.n_components}"
        )
        summary["tables"] = {"Loadings": loadings_table}
        
        uniquenesses = {self.variables[i]: f"{self.res_var[i]:.3f}" for i in range(self.n_variables)}
        summary["additional_stats"] = {"Uniquenesses": uniquenesses}
        
        return summary