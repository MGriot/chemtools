"""
chemtools.exploration.MultipleCorrespondenceAnalysis
-----------------------------------------------------

This module provides the MultipleCorrespondenceAnalysis class for performing 
Multiple Correspondence Analysis (MCA) on categorical data. 

MCA is a statistical technique used to analyze the relationships between 
multiple categorical variables. It represents the data in a 
low-dimensional space, revealing patterns and associations between the 
categories of the variables.

Example usage:
>>> from chemtools.exploration import MultipleCorrespondenceAnalysis
>>> # Assuming 'data' is your contingency table
>>> mca = MultipleCorrespondenceAnalysis() 
>>> mca.fit(data)
>>> # ... Access results and use plotting methods
"""

import numpy as np
import pandas as pd

from chemtools.utility import reorder_array
from chemtools.utility import random_colorHEX
from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.base import BaseModel


class MultipleCorrespondenceAnalysis(BaseModel):
    """
    Performs Multiple Correspondence Analysis (MCA) on a contingency table.

    Attributes:
        model_name (str): Name of the model.
        X (np.ndarray): Contingency table (input data).
        variables (np.ndarray): Names of the variables.
        objects (np.ndarray): Names of the objects.
        n_variables (int): Number of variables.
        n_objects (int): Number of objects.
        variables_colors (list): List of colors for the variables.
        objects_colors (list): List of colors for the objects.
        correspondence_matrix (np.ndarray): Normalized contingency table.
        row_profiles (np.ndarray): Probability of observing each variable given an object.
        col_profiles (np.ndarray): Probability of observing each object given a variable.
        V (np.ndarray): Eigenvalues of the decomposition.
        L (np.ndarray): Left singular vectors (object coordinates in factor space).
        G (np.ndarray): Right singular vectors (variable coordinates in factor space).
        order (np.ndarray): Indices that order eigenvalues in descending order.
        V_ordered (np.ndarray): Ordered eigenvalues.
        L_ordered (np.ndarray): Object coordinates ordered according to eigenvalues.
        PC_index (np.ndarray): Names of the principal components.
    """

    def __init__(self):
        self.model_name = "Multiple Correspondence Analysis"

    def fit(self, X, variables_names=None, objects_names=None):
        """
        Fits the MCA model to the data.

        Args:
            X (np.ndarray): Contingency table.
            variables_names (list, optional): List of variable names. Defaults to None.
            objects_names (list, optional): List of object names. Defaults to None.
        """
        # 1. Contingency Table
        self.X = X
        self.variables = set_variables_names(variables_names, X)
        self.objects = set_objects_names(objects_names, X)
        self.n_variables = self.X.shape[1]
        self.n_objects = self.X.shape[0]
        self.variables_colors = self.change_variables_colors()
        self.objects_colors = self.change_objects_colors()

        # 2. Correspondence Matrix
        grand_total = np.sum(self.X)
        self.correspondence_matrix = self.X / grand_total

        # 3. Row and Column Profiles
        self.row_profiles = self.correspondence_matrix / np.sum(
            self.correspondence_matrix, axis=1, keepdims=True
        )
        self.col_profiles = self.correspondence_matrix / np.sum(
            self.correspondence_matrix, axis=0, keepdims=True
        )

        # 4. Centering the matrix
        row_sums = np.sum(self.correspondence_matrix, axis=1)
        col_sums = np.sum(self.correspondence_matrix, axis=0)
        expected_freqs = np.outer(row_sums, col_sums) / grand_total
        centered_matrix = (self.correspondence_matrix - expected_freqs) / np.sqrt(
            expected_freqs
        )

        # 5. Apply SVD
        U, s, V = np.linalg.svd(centered_matrix)

        # 6.  Eigenvalues and Eigenvectors
        self.V = s**2
        self.L = U
        self.G = V * s[:, None]  # Calculate variable coordinates
        self.order = np.argsort(self.V)[::-1]
        self.V_ordered = self.V[self.order]
        self.L_ordered = self.L[:, self.order]
        self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])

    def change_variables_colors(self):
        """Generates random colors for the variables."""
        return random_colorHEX(self.n_variables)

    def change_objects_colors(self):
        """Generates random colors for the objects."""
        return random_colorHEX(self.n_objects)

    def _get_summary_data(self, X, explained_variance):
        """
        Calculates summary data for the MCA model.

        Args:
            X (np.ndarray): The data matrix.
            explained_variance (np.ndarray): Explained variance for each component.

        Returns:
            pd.DataFrame: A DataFrame containing the summary data.
        """
        n_components = X.shape[1]
        columns = ["Variable", "PC"] + [f"PC{i+1}" for i in range(n_components)]
        summary_data = []

        for i, var in enumerate(self.variables):
            row = [var, ""] + list(X[i, :])
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data, columns=columns)
        summary_df["Explained Variance"] = explained_variance

        return summary_df
