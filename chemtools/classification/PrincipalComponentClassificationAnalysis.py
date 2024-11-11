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
from chemtools.utility import reorder_array

from chemtools.utility.set_names import initialize_names_and_counts, set_variables_names, set_objects_names
from chemtools.utility import random_colorHEX
from chemtools.base.base_models import BaseModel
from chemtools.exploration import PrincipalComponentAnalysis


class PrincipalComponentClassificationAnalysis(BaseModel):
    """
    A class to perform Principal Component-based Classification Analysis (PCCCA).

    This method combines Principal Component Analysis (PCA) with Canonical Correlation
    Analysis (CCA) to achieve classification based on dimensionality reduction. It's
    particularly useful for datasets with a high number of variables and a clear
    separation between classes.

    Attributes:
        model_name (str): The name of the model.
        X (ndarray): The input data.
        y (ndarray): The class labels.
        variables (list): Names of the variables.
        objects (list): Names of the objects.
        n_variables (int): Number of variables in the dataset.
        n_objects (int): Number of objects in the dataset.
        variables_colors (list): Colors assigned to variables for visualization.
        objects_colors (list): Colors assigned to objects for visualization.
        pca_model (PrincipalComponentAnalysis): Fitted PCA model.
        cca_model (PLSCanonical): Fitted CCA model.
        n_components (int): Number of components retained after reduction.
        T (ndarray): Transformed data in the PCA space.
        U (ndarray): Transformed class indicator matrix in the CCA space.

    Methods:
        __init__: Initializes the PCCCA model.
        fit: Fits the PCCCA model to the provided data and class labels.
        transform: Transforms the data into the reduced CCA space.
        change_variables_colors: Generates colors for the variables.
        change_objects_colors: Generates colors for the objects.
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
        Fits the PCCCA model to the provided data and class labels.

        This method first performs PCA on the input data to reduce its dimensionality.
        Then, it applies CCA between the principal components and a class indicator
        matrix derived from the class labels. This results in a low-dimensional
        representation that maximizes the correlation between the data and class
        information.

        Args:
            X (ndarray): The input data.
            y (ndarray): The class labels.
            n_components (int, optional): The number of components to retain. Defaults to 2.
            variables_names (list, optional): Names of the variables. Defaults to None.
            objects_names (list, optional): Names of the objects. Defaults to None.

        Returns:
            None
        """
        self.X = X
        self.y = y
        self.variables, self.objects, self.n_variables, self.n_objects = initialize_names_and_counts(
            X, variables_names, objects_names
        )
        self.variables_colors = self.change_variables_colors()
        self.objects_colors = self.change_objects_colors()

        self.pca_model = PrincipalComponentAnalysis()
        self.pca_model.fit(self.X, variables_names, objects_names)
        self.pca_model.reduction(n_components)
        self.n_components = n_components
        self.T = self.pca_model.T

        Y_dummy = pd.get_dummies(self.y).to_numpy()
        Rxx = np.corrcoef(self.T, rowvar=False)
        Ryy = np.corrcoef(Y_dummy, rowvar=False)
        Rxy = np.corrcoef(self.T, Y_dummy, rowvar=False)[
            : self.n_components, self.n_components :
        ]

        A = np.linalg.solve(Rxx, Rxy @ np.linalg.solve(Ryy, Rxy.T))
        eigenvalues, eigenvectors = np.linalg.eig(A)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.Wx = eigenvectors[:, : self.n_components]
        self.U = self.T @ self.Wx

    def transform(self, X_test):
        """
        Transforms the data into the reduced CCA space.

        This method applies the fitted PCA and CCA transformations to new data,
        projecting it onto the lower-dimensional space that captures class-relevant
        information.

        Args:
            X_test (ndarray): The new data to transform.

        Returns:
            ndarray: The transformed data in the CCA space.
        """
        T_test = self.pca_model.transform(
            X_test
        )  # Assuming your PCA model has a transform method
        U_test = self.pca_model.transform(T_test)
        return U_test

    def change_variables_colors(self):
        return random_colorHEX(self.n_variables)

    def change_objects_colors(self):
        return random_colorHEX(self.n_objects)
