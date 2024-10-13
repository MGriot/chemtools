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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from chemtools.preprocessing import autoscaling
from chemtools.preprocessing.matrix_standard_deviation import matrix_standard_deviation
from chemtools.preprocessing import correlation_matrix
from chemtools.preprocessing import diagonalized_matrix
from chemtools.utility import reorder_array

from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.utility import random_colorHEX
from chemtools.base import BaseModel


class PrincipalComponentAnalysis(BaseModel):
    def __init__(self):
        self.model_name = "Principal Component Analysis"

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
        self.n_component = n_components
        self.V_reduced = self.V_ordered[:n_components]
        self.W = self.L_ordered[:, :n_components]
        self.T = np.dot(self.X_autoscaled, self.W)

    def statistics(self, alpha=0.05):
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

    def hotellings_t2_critical_value(self, alpha=0.05):
        p = self.n_variables
        n = self.n_objects
        f_critical_value = f.ppf(1 - alpha, p, n - p)
        return (p * (n - 1)) / (n - p) * f_critical_value

    def change_variables_colors(self):
        return random_colorHEX(self.n_variables)

    def change_objects_colors(self):
        return random_colorHEX(self.n_objects)
