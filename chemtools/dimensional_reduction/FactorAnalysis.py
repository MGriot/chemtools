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
from chemtools.utility import reorder_array

from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.utility import random_colorHEX
from chemtools.base import BaseModel
from chemtools.dimensional_reduction import DimensionalityReduction


class FactorAnalysis(DimensionalityReduction):
    """
    A class to perform Factor Analysis (FA) on a dataset.

    This class provides methods to fit the FA model to the data, reduce its dimensionality,
    and compute statistical metrics related to the analysis. It also manages the internal
    state of the FA, including the mean, standard deviation, and eigenvalues of the data.

    Attributes:
        model_name (str): The name of the model.
        X (ndarray): The input data.
        variables (list): Names of the variables.
        objects (list): Names of the objects.
        n_variables (int): Number of variables in the dataset.
        n_objects (int): Number of objects in the dataset.
        variables_colors (list): Colors assigned to variables for visualization.
        objects_colors (list): Colors assigned to objects for visualization.
        mean (ndarray): Mean of the input data.
        std (ndarray): Standard deviation of the input data.
        X_autoscaled (ndarray): Autoscaled version of the input data.
        correlation_matrix (ndarray): Correlation matrix of the autoscaled data.
        V (ndarray): Eigenvalues of the correlation matrix.
        L (ndarray): Eigenvectors of the correlation matrix.
        order (ndarray): Order of eigenvalues.
        V_ordered (ndarray): Ordered eigenvalues.
        L_ordered (ndarray): Ordered eigenvectors.
        PC_index (ndarray): Index labels for principal components.
        n_components (int): Number of components retained after reduction.
        V_reduced (ndarray): Reduced eigenvalues.
        W (ndarray): Reduced eigenvectors (loadings).
        T (ndarray): Transformed data in the FA space (scores).
        communalities (ndarray): Communalities of the variables.
        res_var (ndarray): Residual variances of the variables.

    Methods:
        __init__: Initializes the FA model.
        fit: Fits the FA model to the provided data.
        transform: Projects new data onto the factor space.
        fit_transform: Fits FA to data and transforms it.
        change_variables_colors: Generates colors for the variables.
        change_objects_colors: Generates colors for the objects.
        _get_summary_data: Returns a dictionary of data for the summary.
        rotate: Performs rotation on the factor loadings.
        _varimax_rotation: Performs varimax rotation on a matrix.
        _quartimax_rotation: Performs quartimax rotation on a matrix.
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
            X (ndarray): The data to fit the model to.
            variables_names (list, optional): Names of the variables. Defaults to None.
            objects_names (list, optional): Names of the objects. Defaults to None.
        """
        self.X = X
        self.variables = set_variables_names(variables_names, X)
        self.objects = set_objects_names(objects_names, X)
        self.n_variables = self.X.shape[1]
        self.n_objects = self.X.shape[0]
        self.variables_colors = self.change_variables_colors()
        self.objects_colors = self.change_objects_colors()

        try:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0)
            # Check to avoid division by zero and identify problematic columns
            zero_std_columns = np.where(self.std == 0)[0]
            if zero_std_columns.size > 0:
                raise ValueError(
                    f"The standard deviation contains zero in the columns: {zero_std_columns}"
                )
            self.X_autoscaled = (self.X - self.mean) / self.std
            self.correlation_matrix = np.corrcoef(self.X_autoscaled, rowvar=False)

            # Calculate eigenvalues and eigenvectors
            self.V, self.L = eigh(self.correlation_matrix)

            # Sort eigenvalues and eigenvectors in descending order
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]

            # Select the top n_components eigenvectors
            self.V_reduced = self.V_ordered[: self.n_components]
            self.PC_index = [f"F{i+1}" for i in range(self.n_components)]
            self.index = np.array([f"F{i+1}" for i in range(self.V.shape[0])])
            self.W = self.L_ordered[:, : self.n_components]

            # Calculate factor scores
            self.T = self.X_autoscaled @ self.W

            # Calculate communalities and residual variances
            self.communalities = np.sum(self.W**2, axis=1)
            self.res_var = 1 - self.communalities

            # Apply rotation if specified
            if self.rotation == "varimax":
                self.W = self._varimax_rotation(self.W)
            elif self.rotation == "quartimax":
                self.W = self._quartimax_rotation(self.W)

        except np.linalg.LinAlgError as e:
            print(f"Error during the calculation of eigenvalues and eigenvectors: {e}")
        except ValueError as e:
            print(f"Error in input data: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")

    def transform(self, X_new):
        """
        Projects new data onto the factor space.

        Args:
            X_new (ndarray): The new data to transform (shape: n_samples x n_variables).

        Returns:
            ndarray: The transformed data in the factor space (shape: n_samples x n_components).
        """
        X_new_autoscaled = (X_new - self.mean) / self.std
        return np.dot(X_new_autoscaled, self.W)

    def fit_transform(self, X):
        """
        Fit FactorAnalysis to X and transform X.

        Args:
            X (ndarray): The data to fit the model to and then transform.

        Returns:
            ndarray: The transformed data in the FA space.
        """
        self.fit(X)
        return self.transform(X)

    def change_variables_colors(self):
        """Generates random color codes for the variables."""
        return random_colorHEX(self.n_variables)

    def change_objects_colors(self):
        """Generates random color codes for the objects."""
        return random_colorHEX(self.n_objects)

    def _get_summary_data(self):
        """Returns a dictionary of data for the summary."""

        summary = {
            "general": {
                "No. Variables": f"{self.n_variables}",
                "No. Objects": f"{self.n_objects}",
                "No. Components": f"{self.n_components}",
            },
            "communalities": {
                var: f"{comm:.2f}"
                for var, comm in zip(self.variables, self.communalities)
            },
        }
        return summary

    def _varimax_rotation(self, A):
        """Performs Varimax rotation on a matrix.

        Args:
            A (ndarray): The matrix to rotate.

        Returns:
            ndarray: The rotated matrix.
        """
        p, k = A.shape
        R = np.eye(k)
        d = 0
        for _ in range(200):
            A_rot = np.dot(A, R)
            u, _, vt = np.linalg.svd(
                np.dot(
                    A_rot.T,
                    np.power(A_rot, 3)
                    - np.dot(A_rot, np.diag(np.sum(A_rot**2, axis=0))),
                )
            )
            R = np.dot(u, vt)
            d_old = d
            d = np.sum(np.diag(np.dot(A_rot.T, A_rot)))
            if np.abs(d - d_old) < 1e-6:
                break
        return np.dot(A, R)

    def _quartimax_rotation(self, A):
        """Performs Quartimax rotation on a matrix.

        Args:
            A (ndarray): The matrix to rotate.

        Returns:
            ndarray: The rotated matrix.
        """
        p, k = A.shape
        R = np.eye(k)
        d = 0
        for _ in range(200):
            A_rot = np.dot(A, R)
            u, _, vt = np.linalg.svd(np.dot(A_rot**3, A_rot.T))
            R = np.dot(u, vt)
            d_old = d
            d = np.sum(np.diag(np.dot(A_rot, A_rot.T)))
            if np.abs(d - d_old) < 1e-6:
                break
        return np.dot(A, R)

    def _get_summary_data(self):
        """Returns a dictionary of data for the summary."""

        # Prepare loadings for display
        loadings_output = {f"Factor{i+1}": [] for i in range(self.n_components)}
        for i in range(self.n_variables):
            for j in range(self.n_components):
                if abs(self.W[i, j]) > 0.3:  # Consider loadings above a threshold
                    loadings_output[f"Factor{j+1}"].append(
                        f"{self.variables[i]}: {self.W[i, j]:.2f}"
                    )

        # --- Prepare loadings for table display ---
        loadings_table = [
            ["Variable"] + [f"Factor {j+1}" for j in range(self.n_components)]
        ]
        for i in range(self.n_variables):
            row = [self.variables[i]]
            for j in range(self.n_components):
                row.append(f"{self.W[i, j]:.2f}")
            loadings_table.append(row)

        # Prepare coefficients for display
        coefficients_table = []
        coefficients_table.append(["Uniquenesses"])  # Section Header
        coefficients_table.extend(
            [var, f"{uniq:.2f}"] for var, uniq in zip(self.variables, self.res_var)
        )

        summary = {
            "general": {
                "No. Variables": f"{self.n_variables}",
                "No. Objects": f"{self.n_objects}",
                "No. Components": f"{self.n_components}",
            },
            "tables": {"Loadings": loadings_table},
        }
        return summary
