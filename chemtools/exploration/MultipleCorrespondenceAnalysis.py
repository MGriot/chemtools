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

from chemtools.utility import reorder_array
from chemtools.utility import random_colorHEX
from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.base import BaseModel


class MultipleCorrespondenceAnalysis(BaseModel):
    def __init__(self):
        self.model_name = "Multiple Correspondence Analysis"

    def fit(self, X, variables_names=None, objects_names=None):
        # 1. Contingency Table (Assuming X is already a contingency table)
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

        # 4. Apply SVD (Singular Value Decomposition) on the centered matrix
        row_sums = np.sum(self.correspondence_matrix, axis=1)
        col_sums = np.sum(self.correspondence_matrix, axis=0)
        expected_freqs = np.outer(row_sums, col_sums) / grand_total
        centered_matrix = (self.correspondence_matrix - expected_freqs) / np.sqrt(
            expected_freqs
        )

        # 5. Get Eigenvalues and Eigenvectors
        U, s, V = np.linalg.svd(centered_matrix)
        self.V = s**2  # Eigenvalues are squared singular values
        self.L = U  # Left singular vectors correspond to row (object) coordinates
        self.order = np.argsort(self.V)[::-1]
        self.V_ordered = self.V[self.order]
        self.L_ordered = self.L[:, self.order]
        self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])

    def change_variables_colors(self):
        return random_colorHEX(self.n_variables)

    def change_objects_colors(self):
        return random_colorHEX(self.n_objects)
