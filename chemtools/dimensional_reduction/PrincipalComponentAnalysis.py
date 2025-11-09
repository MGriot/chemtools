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

from chemtools.utils import HarmonizedPaletteGenerator
from chemtools.utils.data import set_objects_names, set_variables_names
from chemtools.base import BaseModel


class PrincipalComponentAnalysis(BaseModel):
    """
    Performs Principal Component Analysis (PCA) on a dataset.

    PCA is a statistical procedure that uses an orthogonal transformation to
    convert a set of observations of possibly correlated variables into a set of
    linearly uncorrelated variables called principal components.

    This class provides methods to fit the PCA model, reduce dimensionality,
    and compute various statistics for analysis.

    Attributes:
        model_name (str): The name of the model, "Principal Component Analysis".
        method (str): The method name, "PCA".
        X (np.ndarray): The input data.
        n_variables (int): Number of variables in the dataset.
        n_objects (int): Number of objects (samples) in the dataset.
        eigenvalues (np.ndarray): Ordered eigenvalues (descending).
        eigenvectors (np.ndarray): Ordered eigenvectors.
        T (np.ndarray): Scores matrix (transformed data in the PCA space).
        W (np.ndarray): Loadings matrix (reduced eigenvectors).
        explained_variance_ratio (np.ndarray): Percentage of variance explained by each component.
        T2 (np.ndarray): Hotelling's T-squared values for each object.
        Q (np.ndarray): Squared Prediction Error (SPE) or Q-residuals for each object.

    References:
        - https://en.wikipedia.org/wiki/Principal_component_analysis
    """

    def __init__(self):
        self.model_name = "Principal Component Analysis"
        self.method = "PCA"
        self.notes = []
        self.X = None
        self.variables = None
        self.objects = None
        self.n_variables = None
        self.n_objects = None
        self.variables_colors = None
        self.objects_colors = None
        self.mean = None
        self.std = None
        self.X_autoscaled = None
        self.correlation_matrix = None
        self.V = None
        self.L = None
        self.order = None
        self.V_ordered = None
        self.L_ordered = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.PC_index = None
        self.index = None
        self.explained_variance_ratio = None
        self.n_component = None
        self.V_reduced = None
        self.W = None
        self.T = None
        self.X_reconstructed = None
        self.E = None
        self.T2 = None
        self.T2con = None
        self.Q = None
        self.Qcon = None
        self.T2_critical_value = None

    def fit(self, X, variables_names=None, objects_names=None):
        """
        Fits the PCA model to the provided data.

        Performs autoscaling, calculates the correlation matrix, computes eigenvalues
        and eigenvectors, and orders them. Initializes W and T using all components.

        Args:
            X (np.ndarray): The input data (shape: n_objects x n_variables).
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

            zero_std_columns = np.where(self.std == 0)[0]
            if zero_std_columns.size > 0:
                raise ValueError(
                    f"The standard deviation contains zero in the columns: {zero_std_columns}. "
                    "PCA cannot be performed on variables with zero variance."
                )

            self.X_autoscaled = (self.X - self.mean) / self.std
            self.correlation_matrix = np.corrcoef(self.X_autoscaled, rowvar=False)
            self.V, self.L = np.linalg.eigh(self.correlation_matrix)
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]
            self.eigenvalues = self.V_ordered
            self.eigenvectors = self.L_ordered
            self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
            self.index = self.PC_index

            total_variance = np.sum(self.V_ordered)
            if total_variance > 0:
                self.explained_variance_ratio = self.V_ordered / total_variance
            else:
                self.explained_variance_ratio = np.zeros_like(self.V_ordered)

            self.n_component = self.n_variables
            self.V_reduced = self.V_ordered[: self.n_component]
            self.W = self.L_ordered[:, : self.n_component]
            self.T = np.dot(self.X_autoscaled, self.W)

        except np.linalg.LinAlgError as e:
            self.notes.append(f"Fit Error: Linear algebra error - {e}")
            raise
        except ValueError as e:
            self.notes.append(f"Fit Error: Input data issue - {e}")
            raise
        except Exception as e:
            self.notes.append(f"Fit Error: Unknown error - {e}")
            raise

    def reduction(self, n_components):
        """
        Reduces the dimensionality of the dataset.

        Args:
            n_components (int): The number of principal components to retain.
        """
        if not hasattr(self, "V_ordered") or self.V_ordered is None:
            raise RuntimeError("Fit method must be called before reduction.")

        if not 0 <= n_components <= self.n_variables:
            raise ValueError(f"n_components must be between 0 and {self.n_variables}.")

        self.n_component = n_components
        self.V_reduced = self.V_ordered[:n_components]
        self.W = self.L_ordered[:, :n_components]
        self.T = np.dot(self.X_autoscaled, self.W)

        if hasattr(self, "T2") and self.T2 is not None:
            self.notes.append("Reduction Warning: Statistics recalculated after reduction.")
            self.X_reconstructed = None
            self.E = None
            self.T2 = None
            self.T2con = None
            self.Q = None
            self.Qcon = None
            self.T2_critical_value = None

    def statistics(self, alpha=0.05):
        """
        Calculates statistical metrics for the PCA model.

        Args:
            alpha (float): Significance level for critical value calculation. Default is 0.05.
        """
        if not hasattr(self, "W") or self.W is None:
            raise RuntimeError("Fit and Reduction must be called before statistics.")

        if self.n_component == 0:
            self.notes.append("Statistics Warning: 0 components selected.")
            self.X_reconstructed = np.zeros_like(self.X_autoscaled)
            self.E = self.X_autoscaled
            self.T2 = np.zeros(self.n_objects)
            self.T2con = np.zeros_like(self.X_autoscaled)
            self.Q = np.sum(self.E**2, axis=1)
            self.Qcon = self.E
            self.T2_critical_value = np.nan
            return

        self.X_reconstructed = np.dot(self.T, self.W.T)
        self.E = self.X_autoscaled - self.X_reconstructed

        V_reduced_stable = np.where(self.V_reduced <= 1e-9, 1e-9, self.V_reduced)
        if np.any(self.V_reduced <= 1e-9):
            self.notes.append("Statistics Warning: Near-zero reduced eigenvalues.")
        
        self.T2 = np.diag(self.T @ np.diag(V_reduced_stable ** -1) @ self.T.T)
        self.T2con = self.T @ np.diag(V_reduced_stable ** -0.5) @ self.W.T
        self.Q = np.sum(self.E**2, axis=1)
        self.Qcon = self.E

        p = self.n_component
        n = self.n_objects
        if n > p:
            self.T2_critical_value = self.hotellings_t2_critical_value(alpha=alpha, p=p, n=n)
        else:
            self.notes.append("Statistics Warning: Cannot calculate T2 critical value (n <= p).")
            self.T2_critical_value = np.nan

    def transform(self, X_new):
        """
        Projects new data onto the principal component space.

        Args:
            X_new (np.ndarray): New data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        if not hasattr(self, "mean") or self.W is None:
            raise RuntimeError("Fit method must be called before transforming data.")
        if X_new.shape[1] != self.n_variables:
            raise ValueError("New data has incorrect number of variables.")

        std_stable = np.where(self.std == 0, 1e-9, self.std)
        X_new_autoscaled = (X_new - self.mean) / std_stable
        return np.dot(X_new_autoscaled, self.W)

    def hotellings_t2_critical_value(self, alpha=0.05, p=None, n=None):
        """
        Calculates the critical value for Hotelling's T-squared statistic.
        """
        p = p if p is not None else self.n_component
        n = n if n is not None else self.n_objects

        if n is None or p is None or n <= p:
            return np.nan

        try:
            f_critical_value = f.ppf(1 - alpha, p, n - p)
            return (p * (n - 1)) / (n - p) * f_critical_value
        except (ValueError, Exception):
            return np.nan

    def change_variables_colors(self):
        if self.n_variables is None: return []
        return HarmonizedPaletteGenerator(self.n_variables).generate()

    def change_objects_colors(self):
        if self.n_objects is None: return []
        return HarmonizedPaletteGenerator(self.n_objects).generate()

    def _get_summary_data(self):
        if not hasattr(self, "V_ordered") or self.V_ordered is None:
            return {}

        cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)

        summary = {
            "general": {
                "No. Variables": f"{self.n_variables}",
                "No. Objects": f"{self.n_objects}",
                "No. Components (Reduced)": f"{self.n_component if self.n_component is not None else 'N/A'}",
            },
            "eigenvalues": {
                self.PC_index[i]: [
                    f"{self.V_ordered[i]:.4f}",
                    f"{self.explained_variance_ratio[i] * 100:.2f}%",
                    f"{cumulative_explained_variance[i] * 100:.2f}%",
                ]
                for i in range(len(self.V_ordered))
            },
            "statistics": (
                {
                    "Hotelling's T2 Critical Value (alpha=0.05)": (
                        f"{self.T2_critical_value:.4f}"
                        if self.T2_critical_value is not None and not np.isnan(self.T2_critical_value)
                        else "N/A"
                    ),
                }
                if hasattr(self, "T2_critical_value")
                else {}
            ),
        }
        return summary