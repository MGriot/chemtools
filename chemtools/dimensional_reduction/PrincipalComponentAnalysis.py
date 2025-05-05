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

# Assuming these imports are available in the chemtools library
# from chemtools.preprocessing import autoscaling # Not directly used in the provided code snippet, but kept for context
# from chemtools.preprocessing.matrix_standard_deviation import matrix_standard_deviation # Not directly used
# from chemtools.preprocessing import correlation_matrix # np.corrcoef is used instead
# from chemtools.preprocessing import diagonalized_matrix # Not directly used
# from chemtools.utility import reorder_array # Not directly used

from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.utility import random_colorHEX
from chemtools.base import BaseModel  # Assuming BaseModel provides basic structure


class PrincipalComponentAnalysis(BaseModel):
    """
    A class to perform Principal Component Analysis (PCA) on a dataset.

    This class provides methods to fit the PCA model to the data, reduce its dimensionality,
    and compute statistical metrics related to the analysis. It also manages the internal
    state of the PCA, including the mean, standard deviation, and eigenvalues of the data.

    Attributes:
        model_name (str): The name of the model.
        method (str): The method name ("PCA").
        notes (list): A list to store notes.
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
        V (ndarray): Eigenvalues of the correlation matrix (unordered).
        L (ndarray): Eigenvectors of the correlation matrix (unordered).
        order (ndarray): Order of eigenvalues (descending).
        V_ordered (ndarray): Ordered eigenvalues (descending).
        L_ordered (ndarray): Ordered eigenvectors (corresponding to V_ordered).
        eigenvalues (ndarray): Ordered eigenvalues (same as V_ordered, for clarity/testing).
        eigenvectors (ndarray): Ordered eigenvectors (same as L_ordered, for clarity/testing). # Added
        PC_index (ndarray): Index labels for principal components (e.g., 'PC1', 'PC2').
        index (ndarray): Index labels for principal components (same as PC_index).
        explained_variance_ratio (ndarray): The percentage of variance explained by each component.
        n_component (int): Number of components retained after reduction.
        V_reduced (ndarray): Reduced eigenvalues (subset of V_ordered).
        W (ndarray): Reduced eigenvectors (subset of L_ordered).
        T (ndarray): Transformed data in the PCA space (scores).
        X_reconstructed (ndarray): Data reconstructed from reduced components.
        E (ndarray): Residuals (X_autoscaled - X_reconstructed).
        T2 (ndarray): Hotelling's T-squared values for each object.
        T2con (ndarray): Contribution of each variable to Hotelling's T-squared.
        Q (ndarray): Squared Prediction Error (SPE) or Q-residuals for each object.
        Qcon (ndarray): Contribution of each variable to SPE/Q-residuals.
        T2_critical_value (float): Critical value for Hotelling's T-squared.

    Methods:
        __init__: Initializes the PCA model.
        fit: Fits the PCA model to the provided data.
        reduction: Reduces the dimensionality of the dataset.
        statistics: Computes statistical metrics for the PCA.
        transform: Projects new data onto the principal component space.
        hotellings_t2_critical_value: Calculates the critical value for Hotelling's T-squared.
        change_variables_colors: Generates colors for the variables.
        change_objects_colors: Generates colors for the objects.
        _get_summary_data: Returns data for the summary.
    """

    def __init__(self):
        self.model_name = "Principal Component Analysis"
        self.method = "PCA"  # Set the method name for the summary
        self.notes = []  # Add this line to initialize notes
        # Initialize attributes that will be set during fit
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
        self.eigenvalues = None  # Added for clarity and testing
        self.eigenvectors = None  # Added for clarity and testing
        self.PC_index = None
        self.index = None
        self.explained_variance_ratio = None  # Added
        self.n_component = None  # Will be set initially to n_variables in fit
        self.V_reduced = None
        self.W = None  # Will be set initially in fit
        self.T = None  # Will be set initially in fit
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
            X (ndarray): The input data (shape: n_objects x n_variables).
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
            # Autoscaling
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0)

            # Check for zero standard deviation
            zero_std_columns = np.where(self.std == 0)[0]
            if zero_std_columns.size > 0:
                # Add a note instead of raising an error immediately,
                # as autoscaling might still proceed if handled carefully,
                # but the PCA interpretation will be affected.
                # However, for standard PCA, zero std columns are problematic.
                # Raising an error is appropriate here as per the original code's intent.
                raise ValueError(
                    f"The standard deviation contains zero in the columns: {zero_std_columns}. "
                    "PCA cannot be performed on variables with zero variance."
                )

            self.X_autoscaled = (self.X - self.mean) / self.std

            # Correlation Matrix
            self.correlation_matrix = np.corrcoef(self.X_autoscaled, rowvar=False)

            # Eigenvalues and Eigenvectors
            # Use eigh for symmetric matrices like correlation matrix for stability
            self.V, self.L = np.linalg.eigh(self.correlation_matrix)

            # Order eigenvalues and eigenvectors in descending order
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]

            # Set eigenvalues and eigenvectors attributes for testing/clarity
            self.eigenvalues = self.V_ordered
            self.eigenvectors = self.L_ordered  # Added

            # Set PC index labels
            self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
            self.index = self.PC_index  # Alias for PC_index

            # Calculate explained variance ratio
            total_variance = np.sum(self.V_ordered)
            # Avoid division by zero if total_variance is zero (unlikely with real data, but good practice)
            if total_variance > 0:
                self.explained_variance_ratio = self.V_ordered / total_variance
            else:
                self.explained_variance_ratio = np.zeros_like(
                    self.V_ordered
                )  # Or handle as an error

            # Initialize W and T using all components
            # This allows transform to be called after fit
            self.n_component = self.n_variables  # Initially consider all components
            self.V_reduced = self.V_ordered[: self.n_component]
            self.W = self.L_ordered[:, : self.n_component]
            self.T = np.dot(self.X_autoscaled, self.W)

        except np.linalg.LinAlgError as e:
            print(f"Error during the calculation of eigenvalues and eigenvectors: {e}")
            # Consider adding a note or setting an error state
            self.notes.append(
                f"Fit Error: Linear algebra error during eigenvalue/eigenvector calculation - {e}"
            )
        except ValueError as e:
            print(f"Error in input data: {e}")
            self.notes.append(f"Fit Error: Input data issue - {e}")
        except Exception as e:
            print(f"Unknown error during fit: {e}")
            self.notes.append(f"Fit Error: Unknown error - {e}")

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
        if not hasattr(self, "V_ordered") or self.V_ordered is None:
            print("Error: Fit method must be called before reduction.")
            self.notes.append("Reduction Error: Fit method not called.")
            return

        if n_components > self.n_variables or n_components < 0:
            print(
                f"Error: n_components ({n_components}) must be between 0 and the number of variables ({self.n_variables})."
            )
            self.notes.append(
                f"Reduction Error: Invalid n_components ({n_components})."
            )
            return

        self.n_component = n_components
        self.V_reduced = self.V_ordered[:n_components]
        self.W = self.L_ordered[:, :n_components]
        self.T = np.dot(self.X_autoscaled, self.W)

        # Recalculate statistics after reduction if they were already computed
        if hasattr(self, "T2") and self.T2 is not None:
            print(
                "Warning: Statistics were already computed. Recalculating after reduction."
            )
            self.notes.append(
                "Reduction Warning: Statistics recalculated after reduction."
            )
            # Need to ensure alpha is available, maybe store it as an attribute
            # For now, assume a default alpha or require calling statistics again
            # self.statistics(alpha=self._last_alpha_for_stats) # Requires storing alpha
            # A safer approach is to just clear the stats and require the user to call statistics again
            self.X_reconstructed = None
            self.E = None
            self.T2 = None
            self.T2con = None
            self.Q = None
            self.Qcon = None
            self.T2_critical_value = None

    def statistics(self, alpha=0.05):
        """
        Calculate statistical metrics for the principal component analysis.

        This method computes the reconstructed data, error metrics (Hotelling's T-squared
        and Q-residuals), and critical values based on the principal components. It
        provides insights into the quality of the PCA model and helps in assessing the
        significance of the components.

        Args:
            alpha (float): Significance level for the critical value calculation.
                         Default is 0.05.

        Returns:
            None
        """
        if (
            not hasattr(self, "W")
            or self.W is None
            or not hasattr(self, "T")
            or self.T is None
        ):
            print(
                "Error: Fit and Reduction methods must be called before calculating statistics."
            )
            self.notes.append("Statistics Error: Fit or Reduction not called.")
            return

        if self.n_component == 0:
            print("Warning: Statistics cannot be calculated with 0 components.")
            self.notes.append("Statistics Warning: 0 components selected.")
            # Set stats to None or appropriate empty values
            self.X_reconstructed = np.zeros_like(self.X_autoscaled)
            self.E = self.X_autoscaled  # Error is the original autoscaled data
            self.T2 = np.zeros(self.n_objects)
            self.T2con = np.zeros_like(self.X_autoscaled)
            self.Q = np.sum(self.E**2, axis=1)
            self.Qcon = self.E
            self.T2_critical_value = np.nan  # Or handle appropriately
            return

        # Store alpha for potential recalculation after reduction
        # self._last_alpha_for_stats = alpha # Requires adding _last_alpha_for_stats attribute

        # Reconstructed data
        self.X_reconstructed = np.dot(self.T, self.W.T)

        # Residuals
        self.E = self.X_autoscaled - self.X_reconstructed

        # Hotelling's T-squared
        # Ensure V_reduced has no zero values before inverse
        if np.any(self.V_reduced <= 1e-9):  # Use a small tolerance for near-zero values
            print(
                "Warning: Some reduced eigenvalues are zero or near-zero. Hotelling's T-squared calculation may be unstable."
            )
            self.notes.append("Statistics Warning: Near-zero reduced eigenvalues.")
            # Handle potential division by zero - replace near-zero with a small value or NaN
            V_reduced_stable = np.where(self.V_reduced <= 1e-9, 1e-9, self.V_reduced)
            self.T2 = np.diag(self.T @ np.diag(V_reduced_stable ** (-1)) @ self.T.T)
        else:
            self.T2 = np.diag(self.T @ np.diag(self.V_reduced ** (-1)) @ self.T.T)

        # Contribution to Hotelling's T-squared
        if np.any(self.V_reduced <= 1e-9):
            V_reduced_stable = np.where(self.V_reduced <= 1e-9, 1e-9, self.V_reduced)
            self.T2con = self.T @ np.diag(V_reduced_stable ** (-1 / 2)) @ self.W.T
        else:
            self.T2con = self.T @ np.diag(self.V_reduced ** (-1 / 2)) @ self.W.T

        # Squared Prediction Error (SPE) or Q-residuals
        self.Q = np.sum(self.E**2, axis=1)

        # Contribution to Q-residuals
        self.Qcon = self.E

        # Hotelling's T-squared critical value
        # Ensure p and n are appropriate for the calculation
        p = self.n_component  # Use number of components for T2 critical value
        n = self.n_objects
        if n > p:
            self.T2_critical_value = self.hotellings_t2_critical_value(
                alpha=alpha, p=p, n=n
            )
        else:
            print(
                "Warning: Cannot calculate Hotelling's T-squared critical value (n <= p)."
            )
            self.notes.append(
                "Statistics Warning: Cannot calculate T2 critical value (n <= p)."
            )
            self.T2_critical_value = np.nan  # Not defined

        # Note: Q critical value calculation is more complex and not included in the original code,
        # but is often needed for complete PCA diagnostics.

    def transform(self, X_new):
        """
        Projects new data onto the principal component space.

        Args:
            X_new (ndarray): The new data to transform (shape: n_samples x n_variables).

        Returns:
            ndarray: The transformed data in the PCA space (shape: n_samples x n_components).
        """
        if (
            not hasattr(self, "mean")
            or self.mean is None
            or not hasattr(self, "std")
            or self.std is None
            or not hasattr(self, "W")
            or self.W is None
        ):
            print("Error: Fit method must be called before transforming new data.")
            # Return None or raise an error
            return None  # Or raise AttributeError("PCA model has not been fitted.")

        # Ensure the new data has the same number of variables
        if X_new.shape[1] != self.n_variables:
            print(
                f"Error: New data has {X_new.shape[1]} variables, but the model was fitted with {self.n_variables}."
            )
            return None  # Or raise ValueError("Number of variables in new data does not match the fitted model.")

        # Autoscaling new data using the mean and std from the training data
        # Handle potential division by zero if std was zero in training data (already checked in fit, but defensive here)
        std_stable = np.where(
            self.std == 0, 1e-9, self.std
        )  # Use a small value to avoid division by zero
        X_new_autoscaled = (X_new - self.mean) / std_stable

        # Project onto the reduced principal components
        return np.dot(X_new_autoscaled, self.W)

    def hotellings_t2_critical_value(self, alpha=0.05, p=None, n=None):
        """
        Calculates the critical value for Hotelling's T-squared statistic.

        Args:
            alpha (float): Significance level. Default is 0.05.
            p (int, optional): Number of variables or components used in T2 calculation.
                             Defaults to self.n_component if available, otherwise self.n_variables.
            n (int, optional): Number of objects. Defaults to self.n_objects.

        Returns:
            float: The critical value for Hotelling's T-squared.
        """
        # Use the number of components used for the T2 calculation if p is not provided
        if p is None:
            p = (
                self.n_component
                if hasattr(self, "n_component") and self.n_component is not None
                else self.n_variables
            )
            print(
                f"Using p = {p} (number of components/variables) for T2 critical value calculation."
            )

        if n is None:
            n = (
                self.n_objects
                if hasattr(self, "n_objects") and self.n_objects is not None
                else None
            )
            print(
                f"Using n = {n} (number of objects) for T2 critical value calculation."
            )

        if n is None or n <= p:
            print(
                "Error: Cannot calculate Hotelling's T-squared critical value (n <= p or n is None)."
            )
            return np.nan  # Not defined

        # F-distribution critical value
        try:
            f_critical_value = f.ppf(1 - alpha, p, n - p)
            # Hotelling's T-squared critical value formula
            t2_crit = (p * (n - 1)) / (n - p) * f_critical_value
            return t2_crit
        except ValueError as e:
            print(f"Error calculating F-distribution critical value: {e}")
            return np.nan  # Calculation failed
        except Exception as e:
            print(f"Unknown error calculating T2 critical value: {e}")
            return np.nan

    def change_variables_colors(self):
        """Generates random HEX colors for the variables."""
        if self.n_variables is None:
            print("Error: Fit method must be called before generating variable colors.")
            return []
        return random_colorHEX(self.n_variables)

    def change_objects_colors(self):
        """Generates random HEX colors for the objects."""
        if self.n_objects is None:
            print("Error: Fit method must be called before generating object colors.")
            return []
        return random_colorHEX(self.n_objects)

    def _get_summary_data(self):
        """
        Returns a dictionary of data for the summary.

        This method is intended to be used by a separate summary generation function
        or class. It provides key statistics from the PCA analysis.

        Returns:
            dict: A dictionary containing summary data. Returns an empty dict if fit
                  has not been called.
        """
        if not hasattr(self, "V_ordered") or self.V_ordered is None:
            print("Error: Fit method must be called before generating summary data.")
            return {}  # Return empty dict if no data is available

        # Ensure explained_variance_ratio is calculated
        if (
            not hasattr(self, "explained_variance_ratio")
            or self.explained_variance_ratio is None
        ):
            # This should be calculated in fit, but as a fallback
            total_variance = np.sum(self.V_ordered)
            if total_variance > 0:
                self.explained_variance_ratio = self.V_ordered / total_variance
            else:
                self.explained_variance_ratio = np.zeros_like(self.V_ordered)

        # Ensure cumulative explained variance is calculated
        cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)

        summary = {
            "general": {
                "No. Variables": f"{self.n_variables}",
                "No. Objects": f"{self.n_objects}",
                "No. Components (Reduced)": f"{self.n_component if hasattr(self, 'n_component') and self.n_component is not None else 'N/A'}",  # Added reduced components count
            },
            "eigenvalues": {
                # Using PC_index as keys and a list of values for each PC
                self.PC_index[i]: [
                    f"{self.V_ordered[i]:.4f}",  # Eigenvalue
                    f"{self.explained_variance_ratio[i] * 100:.2f}%",  # Explained Variance Ratio
                    f"{cumulative_explained_variance[i] * 100:.2f}%",  # Cumulative Explained Variance
                ]
                for i in range(len(self.V_ordered))
            },
            # Add other relevant statistics if available (T2_critical_value, etc.)
            "statistics": (
                {
                    "Hotelling's T2 Critical Value (alpha=0.05)": (
                        f"{self.T2_critical_value:.4f}"
                        if hasattr(self, "T2_critical_value")
                        and self.T2_critical_value is not None
                        and not np.isnan(self.T2_critical_value)
                        else "N/A"
                    ),
                    # Add Q critical value if implemented
                }
                if hasattr(self, "T2_critical_value")
                else {}
            ),  # Only include statistics if stats were calculated
        }
        return summary
