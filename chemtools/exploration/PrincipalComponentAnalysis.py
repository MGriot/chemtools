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

from chemtools.preprocessing import autoscaling
from chemtools.preprocessing.matrix_standard_deviation import matrix_standard_deviation
from chemtools.preprocessing import correlation_matrix
from chemtools.preprocessing import diagonalized_matrix
from chemtools.utility import reorder_array

from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.utility import random_colorHEX
from chemtools.base import BaseModel


class PrincipalComponentAnalysis(BaseModel):
    """
    A class to perform Principal Component Analysis (PCA) on a dataset.

    This class provides methods to fit the PCA model to the data, reduce its dimensionality,
    and compute statistical metrics related to the analysis. It also manages the internal
    state of the PCA, including the mean, standard deviation, and eigenvalues of the data.

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
        n_component (int): Number of components retained after reduction.
        V_reduced (ndarray): Reduced eigenvalues.
        W (ndarray): Reduced eigenvectors.
        T (ndarray): Transformed data in the PCA space.

    Methods:
        __init__: Initializes the PCA model.
        fit: Fits the PCA model to the provided data.
        reduction: Reduces the dimensionality of the dataset.
        statistics: Computes statistical metrics for the PCA.
        hotellings_t2_critical_value: Calculates the critical value for Hotelling's T-squared.
        change_variables_colors: Generates colors for the variables.
        change_objects_colors: Generates colors for the objects.
    """

    def __init__(self):
        self.model_name = "Principal Component Analysis"
        self.method = "PCA"  # Set the method name for the summary
        self.notes = []  # Add this line to initialize notes

    def fit(self, X, variables_names=None, objects_names=None):
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
            # Controllo per evitare divisione per zero e identificazione delle colonne problematiche
            zero_std_columns = np.where(self.std == 0)[0]
            if zero_std_columns.size > 0:
                raise ValueError(
                    f"The standard deviation contains zero in the columns: {zero_std_columns}"
                )
            self.X_autoscaled = (self.X - self.mean) / self.std
            self.correlation_matrix = np.corrcoef(self.X_autoscaled, rowvar=False)
            self.V, self.L = np.linalg.eigh(self.correlation_matrix)
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]
            self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
        except np.linalg.LinAlgError as e:
            print(f"Error during the calculation of eigenvalues and eigenvectors: {e}")
        except ValueError as e:
            print(f"Error in input data: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")

    def reduction(self, n_components):
        """
        Reduce the dimensionality of the dataset using principal component analysis.

        This method selects a specified number of principal components to represent the
        original data, effectively reducing its dimensionality while preserving as much
        variance as possible. It updates the internal attributes to reflect the reduced
        components and their corresponding values.

        Args:
            n_components (int): The number of principal components to retain.

        Returns:
            None
        """
        self.n_component = n_components
        self.V_reduced = self.V_ordered[:n_components]
        self.W = self.L_ordered[:, :n_components]
        self.T = np.dot(self.X_autoscaled, self.W)

    def statistics(self, alpha=0.05):
        """
        Calculate statistical metrics for the principal component analysis.

        This method computes the reconstructed data, error metrics, and critical values
        based on the principal components. It provides insights into the quality of the
        PCA model and helps in assessing the significance of the components.

        Args:
            alpha (float): Significance level for the critical value calculation.
                        Default is 0.05.

        Returns:
            None
        """
        self.X_reconstructed = np.dot(self.T, self.W.T)
        self.E = self.X_autoscaled - self.X_reconstructed
        self.T2 = np.diag(
            self.T @ np.diag(self.V_ordered[: self.n_component] ** (-1)) @ self.T.T
        )
        self.T2con = (
            self.T @ np.diag(self.V_ordered[: self.n_component] ** (-1 / 2)) @ self.W.T
        )
        self.Q = np.sum(self.E**2, axis=1)
        self.Qcon = self.E
        self.T2_critical_value = self.hotellings_t2_critical_value(alpha=alpha)

    def transform(self, X_new):
        """
        Projects new data onto the principal component space.

        Args:
            X_new (ndarray): The new data to transform (shape: n_samples x n_variables).

        Returns:
            ndarray: The transformed data in the PCA space (shape: n_samples x n_components).
        """
        X_new_autoscaled = (X_new - self.mean) / self.std
        return np.dot(X_new_autoscaled, self.W)

    def hotellings_t2_critical_value(self, alpha=0.05):
        p = self.n_variables
        n = self.n_objects
        f_critical_value = f.ppf(1 - alpha, p, n - p)
        return (p * (n - 1)) / (n - p) * f_critical_value

    def change_variables_colors(self):
        return random_colorHEX(self.n_variables)

    def change_objects_colors(self):
        return random_colorHEX(self.n_objects)

    def _get_summary_data(self):
        """Returns a dictionary of data for the summary."""
        summary = {
            "general": {
                "No. Variables": f"{self.n_variables}",
                "No. Objects": f"{self.n_objects}",
            },
            "eigenvalues": {
                self.PC_index[i]: [
                    f"{val:.2f}",
                    f"{self.V_ordered[i] / np.sum(self.V_ordered) * 100:.2f}%",
                    f"{np.cumsum(self.V_ordered)[i] / np.sum(self.V_ordered) * 100:.2f}%",
                ]
                for i, val in enumerate(self.V_ordered)
            },
        }
        return summary
