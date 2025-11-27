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

from chemtools.utils import HarmonizedPaletteGenerator
from chemtools.utils.data import set_objects_names, set_variables_names
from chemtools.base import BaseModel


class MultipleCorrespondenceAnalysis(BaseModel):
    """
    Performs Multiple Correspondence Analysis (MCA) on a contingency table.

    MCA is an extension of correspondence analysis (CA) which allows one to
    analyze the pattern of relationships of several categorical dependent
    variables.

    Attributes:
        model_name (str): Name of the model.
        X (np.ndarray): Contingency table (input data).
        V_ordered (np.ndarray): Ordered eigenvalues.
        L_ordered (np.ndarray): Object coordinates ordered according to eigenvalues.
        G (np.ndarray): Variable coordinates in factor space.

    References:
        - https://en.wikipedia.org/wiki/Multiple_correspondence_analysis
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
        self.X = X
        self.variables = set_variables_names(variables_names, X)
        self.objects = set_objects_names(objects_names, X)
        self.n_variables = self.X.shape[1]
        self.n_objects = self.X.shape[0]
        self.variables_colors = self.change_variables_colors()
        self.objects_colors = self.change_objects_colors()

        grand_total = np.sum(self.X)
        if grand_total == 0:
            raise ValueError("The input contingency table cannot be all zeros.")
            
        self.correspondence_matrix = self.X / grand_total

        row_sums = np.sum(self.correspondence_matrix, axis=1, keepdims=True)
        col_sums = np.sum(self.correspondence_matrix, axis=0, keepdims=True)
        
        if np.any(row_sums == 0) or np.any(col_sums == 0):
            raise ValueError("The input table cannot have rows or columns that sum to zero.")

        self.row_profiles = self.correspondence_matrix / row_sums
        self.col_profiles = self.correspondence_matrix / col_sums

        expected_freqs = np.outer(row_sums.flatten(), col_sums.flatten())
        centered_matrix = (self.correspondence_matrix - expected_freqs) / np.sqrt(expected_freqs)

        U, s, Vt = np.linalg.svd(centered_matrix, full_matrices=False)

        self.V = s**2
        self.L = U
        self.G = Vt.T * s
        
        self.order = np.argsort(self.V)[::-1]
        self.V_ordered = self.V[self.order]
        self.L_ordered = self.L[:, self.order]
        self.G_ordered = self.G[:, self.order]
        
        self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])

    def change_variables_colors(self):
        """Generates random colors for the variables."""
        return HarmonizedPaletteGenerator(self.n_variables).generate()

    def change_objects_colors(self):
        """Generates random colors for the objects."""
        return HarmonizedPaletteGenerator(self.n_objects).generate()

    def _get_summary_data(self):
        """
        Calculates summary data for the MCA model.
        """
        if not hasattr(self, 'V_ordered'):
            return {}

        explained_variance = self.V_ordered / np.sum(self.V_ordered)
        cumulative_variance = np.cumsum(explained_variance)

        summary = self._create_general_summary(
            self.n_variables,
            self.n_objects
        )

        eigenvalue_table = [
            ["Component", "Eigenvalue", "Explained Variance (%)", "Cumulative Variance (%)"]
        ]
        for i in range(len(self.V_ordered)):
            eigenvalue_table.append([
                f"PC{i+1}",
                f"{self.V_ordered[i]:.4f}",
                f"{explained_variance[i] * 100:.2f}",
                f"{cumulative_variance[i] * 100:.2f}"
            ])
        
        summary["tables"] = {"Eigenvalues": eigenvalue_table}
        return summary