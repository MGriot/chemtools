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
from chemtools.utility import heatmap
from chemtools.utility import annotate_heatmap
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

    ### PLots ----------------------------------------------------------------
    def plot_correlation_matrix(self, cmap="coolwarm", threshold=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.correlation_matrix, cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap, label="Correlation value")
        ax.set_xticks(np.arange(len(self.variables)))
        ax.set_yticks(np.arange(len(self.variables)))
        ax.set_xticklabels(self.variables)
        ax.set_yticklabels(self.variables)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Calcola il valore medio della colormap se threshold è None
        if threshold is None:
            norm = plt.Normalize(
                vmin=self.correlation_matrix.min(), vmax=self.correlation_matrix.max()
            )
            midpoint = (
                self.correlation_matrix.max() + self.correlation_matrix.min()
            ) / 2
            threshold = norm(midpoint)

        for i in range(len(self.variables)):
            for j in range(len(self.variables)):
                color = (
                    "black"
                    if abs(self.correlation_matrix[i, j]) < threshold
                    else "white"
                )
                text = ax.text(
                    j,
                    i,
                    f"{self.correlation_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                )

        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def plot_eigenvalues_greater_than_one(self, show=True):
        self.num_eigenvalues_greater_than_one = np.argmax(self.V_ordered < 1)
        plt.axvline(
            x=self.num_eigenvalues_greater_than_one - 0.5,
            color="brown",
            linestyle="-",
            label="Autovalori maggiori di 1",
        )
        plt.xticks(range(len(self.PC_index)), self.PC_index)
        plt.xlabel(r"$PC_i$")
        plt.legend(loc="best")
        plt.title("Eigenvalue")
        if show:
            plt.tight_layout()
            plt.show()

    def plot_eigenvalues_variance(self, show=True):
        plt.bar(
            x=self.PC_index,
            height=(self.V_ordered / self.V_ordered.sum()) * 100,
            fill=False,
            edgecolor="darkorange",
            label="Varianza %",
        )
        plt.xticks(range(len(self.PC_index)), self.PC_index)
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")
        if show:
            plt.tight_layout()
            plt.show()

    def plot_cumulative_variance(self, show=True):
        c = np.cumsum(self.V_ordered / self.V_ordered.sum()) * 100
        plt.bar(
            x=self.PC_index,
            height=c,
            fill=False,
            edgecolor="black",
            linestyle="--",
            width=0.6,
            label="Varianza cumulata %",
        )
        plt.xticks(range(len(self.PC_index)), self.PC_index)
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")
        if show:
            plt.tight_layout()
            plt.show()

    def plot_average_eigenvalue_criterion(self, show=True):
        plt.axvline(
            x=np.argmax(self.V_ordered < self.V_ordered.mean()) - 0.5,
            color="red",
            alpha=0.5,
            linestyle="-",
            label="AEC",
        )
        plt.xticks(range(len(self.PC_index)), self.PC_index)
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")

        if show:
            plt.tight_layout()
            plt.show()

    def plot_KP_criterion(self, show=True):
        rank = np.linalg.matrix_rank(self.correlation_matrix)
        sum_term = sum(self.V[m] / self.V.sum() - 1 / self.V.size for m in range(rank))
        x = (
            round(
                1
                + (self.V.size - 1)
                * (
                    1
                    - (
                        (sum_term + (self.V.size - rank) ** (1 / self.V.size))
                        / (2 * (self.V.size - 1) / self.V.size)
                    )
                )
            )
            - 1
        )
        plt.axvline(
            x=x,
            color="purple",
            alpha=0.5,
            linestyle="--",
            label="KP",
        )
        plt.xticks(range(len(self.PC_index)), self.PC_index)
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")
        if show:
            plt.tight_layout()
            plt.show()

    def plot_KL_criterion(self, show=True):
        rank = np.linalg.matrix_rank(self.correlation_matrix)
        sum_term = sum(self.V[m] / self.V.sum() - 1 / self.V.size for m in range(rank))
        x = (
            round(
                self.V.size
                ** (
                    1
                    - (sum_term + (self.V.size - rank) ** (1 / self.V.size))
                    / (2 * (self.V.size - 1) / self.V.size)
                )
            )
            - 1
        )
        plt.axvline(
            x=x,
            color="cyan",
            alpha=0.5,
            linestyle="-",
            label="KL",
        )
        plt.xticks(range(len(self.PC_index)), self.PC_index)
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")
        if show:
            plt.tight_layout()
            plt.show()

    def plot_CAEC_criterion(self, show=True):
        plt.axvline(
            x=np.argmax(self.V_ordered < 0.7 * self.V_ordered.mean()) - 0.5,
            color="blue",
            alpha=0.5,
            linestyle="--",
            label="CAEC",
        )
        plt.xticks(range(len(self.PC_index)), self.PC_index)
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")
        if show:
            plt.tight_layout()
            plt.show()

    def plot_broken_stick(self, show=True):
        n = self.V_ordered.shape[0]
        dm = (100 / n) * np.cumsum(1 / np.arange(1, n + 1)[::-1])
        plt.plot(self.PC_index, dm, color="lightgreen", label="Broken stick")
        plt.xticks(range(len(self.PC_index)), self.PC_index)
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")
        if show:
            plt.tight_layout()
            plt.show()

    def plot_eigenvalue(self):
        self.plot_eigenvalues_greater_than_one(show=False)
        self.plot_eigenvalues_variance(show=False)
        self.plot_cumulative_variance(show=False)
        self.plot_average_eigenvalue_criterion(show=False)
        self.plot_KP_criterion(show=False)
        self.plot_KL_criterion(show=False)
        self.plot_CAEC_criterion(show=False)
        self.plot_broken_stick(show=False)
        plt.tight_layout()
        plt.show()

    def plot_hotteling_t2_vs_q(self):
        # Calcola Q critico
        # Q_critico =f_value/(self.T.shape[0]-self.T.shape[1])
        # Crea il grafico
        for i in range(len(self.Q)):
            plt.plot(self.Q[i], self.T2[i], "o", label=self.objects[i])
        # Aggiungi il valore di T2 critico
        # plt.axhline(y=T2_critico, color='r', linestyle='-', label=r"$T^2_{crit}$")
        # Aggiungi il valore di Q critico
        # plt.axvline(x=Q_critico, color='r', linestyle='-', label=r"$Q_{crit}$")
        plt.xlabel(r"$Q$")
        plt.ylabel(r"$Hotteling\'s T^2$")
        plt.legend(loc="best")

    def plot_pci_contribution(self, text_color="black"):
        for i in range(self.W.shape[1]):
            plt.plot(
                np.arange(self.n_variables),
                self.W[:, i],
                marker="o",
                markerfacecolor="none",
                label=f"PC$_{i+1}$",
            )
        plt.title(f"Contributions of the PC$_i$")
        plt.xticks(np.arange(self.n_variables), self.variables)
        plt.legend(labelcolor=text_color)
        plt.xlabel("Variable")
        plt.ylabel("Value of loading")
        plt.tight_layout()
        plt.show()

    def plot_loadings(self, arrows=True, text_color="black", components=None):
        if components is not None:
            i, j = components
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.suptitle(
                f"Loadings plot PC{i+1} vs PC{j+1}", fontsize=24, y=1, color=text_color
            )

            x_data = self.W[:, i]
            y_data = self.W[:, j]
            colors = self.variables_colors
            ax.scatter(x_data, y_data, c=colors, label=self.variables)

            if arrows:
                for d in range(self.n_variables):
                    ax.arrow(
                        0,
                        0,
                        self.W[d, i],
                        self.W[d, j],
                        length_includes_head=True,
                        width=0.01,
                        color=text_color,
                        alpha=0.3,
                    )

            for d in range(self.n_variables):
                position = (
                    ["left", "bottom"]
                    if self.W[d, i] > self.W[:, i].mean()
                    else ["right", "top"]
                )
                ax.annotate(
                    text=self.variables[d],
                    xy=(self.W[d, i], self.W[d, j]),
                    ha=position[0],
                    va=position[1],
                    color=text_color,
                )

            ax.set_xlabel(rf"PC$_{i+1}$")
            ax.set_ylabel(rf"PC$_{j+1}$")
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                labelcolor=text_color,
            )
            plt.tight_layout()
            plt.show()
        else:
            height_ratios = [1] * self.n_component
            width_ratios = [1] * self.n_component

            fig, axs = plt.subplots(
                self.n_component,
                self.n_component,
                figsize=(5 * self.n_component, 5 * self.n_component),
                gridspec_kw={
                    "height_ratios": height_ratios,
                    "width_ratios": width_ratios,
                },
            )
            fig.suptitle("Loadings plot", fontsize=24, y=1, color=text_color)

            for i in range(self.n_component):
                for j in range(self.n_component):
                    ax = axs[i, j] if self.n_component > 1 else axs
                    if i != j:

                        x_data = self.W[:, i]
                        y_data = self.W[:, j]
                        colors = self.variables_colors
                        ax.scatter(x_data, y_data, c=colors, label=self.variables)

                        if arrows:
                            for d in range(self.n_variables):
                                ax.arrow(
                                    0,
                                    0,
                                    self.W[d, i],
                                    self.W[d, j],
                                    length_includes_head=True,
                                    width=0.01,
                                    color=text_color,
                                    alpha=0.3,
                                )

                        for d in range(self.n_variables):
                            position = (
                                ["left", "bottom"]
                                if self.W[d, i] > self.W[:, i].mean()
                                else ["right", "top"]
                            )
                            ax.annotate(
                                text=self.variables[d],
                                xy=(self.W[d, i], self.W[d, j]),
                                ha=position[0],
                                va=position[1],
                                color=text_color,
                            )
                        ax.set_xlabel(rf"PC$_{i+1}$")
                        ax.set_ylabel(rf"PC$_{j+1}$")
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            rf"PC$_{i+1}$",
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=20,
                            color=text_color,
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.add_patch(
                            patches.Rectangle(
                                (0, 0),
                                1,
                                1,
                                fill=False,
                                transform=ax.transAxes,
                                clip_on=False,
                            )
                        )

            handles, labels = ax.get_legend_handles_labels()

            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                labelcolor=text_color,
            )
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            plt.show()

    def plot_scores(self, label_point=False, text_color="black", components=None):
        if components is not None:
            i, j = components
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.suptitle(
                f"Scores plot PC{i+1} vs PC{j+1}", fontsize=24, y=1, color=text_color
            )

            x_data = self.T[:, i]
            y_data = self.T[:, j]
            colors = self.objects_colors
            ax.scatter(x_data, y_data, c=colors, label=self.objects)

            if label_point:
                for d in range(self.n_objects):
                    ax.annotate(
                        text=self.objects[d][0],
                        xy=(self.T[d, i], self.T[d, j]),
                        color=text_color,
                    )

            ax.set_xlabel(f"PC{i+1}")
            ax.set_ylabel(f"PC{j+1}")
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                labelcolor=text_color,
            )
            plt.tight_layout()
            plt.show()
        else:
            height_ratios = [1] * self.n_component
            width_ratios = [1] * self.n_component

            fig, axs = plt.subplots(
                self.n_component,
                self.n_component,
                figsize=(5 * self.n_component, 5 * self.n_component),
                gridspec_kw={
                    "height_ratios": height_ratios,
                    "width_ratios": width_ratios,
                },
            )
            fig.suptitle("Scores plot", fontsize=24, y=1, color=text_color)

            for i in range(self.n_component):
                for j in range(self.n_component):
                    ax = axs[i, j] if self.n_component > 1 else axs
                    if i != j:
                        x_data = self.T[:, i]
                        y_data = self.T[:, j]
                        colors = self.objects_colors
                        ax.scatter(x_data, y_data, c=colors, label=self.objects)

                        if label_point:
                            for d in range(self.n_objects):
                                ax.annotate(
                                    text=self.objects[d][0],
                                    xy=(self.T[d, i], self.T[d, j]),
                                    color=text_color,
                                )
                        ax.set_xlabel(f"PC{i+1}")
                        ax.set_ylabel(f"PC{j+1}")
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            f"PC{i+1}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=20,
                            color=text_color,
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.add_patch(
                            patches.Rectangle(
                                (0, 0),
                                1,
                                1,
                                fill=False,
                                transform=ax.transAxes,
                                clip_on=False,
                            )
                        )

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                labelcolor=text_color,
            )
            plt.tight_layout()
            plt.show()

    def plot_biplot(
        self,
        label_point=False,
        subplot_dimensions=[5, 5],
        text_color="black",
        components=None,
        show_legend=True,  # Aggiunto parametro per mostrare la legenda
    ):
        if components is not None:
            i, j = components
            fig, ax = plt.subplots(
                figsize=(subplot_dimensions[0], subplot_dimensions[1])
            )
            fig.suptitle(
                f"Biplot PC{i+1} vs PC{j+1}", fontsize=24, y=1, color=text_color
            )

            x_data = self.T[:, i]
            y_data = self.T[:, j]
            colors = self.objects_colors
            scatter = ax.scatter(x_data, y_data, c=colors, label=self.objects)

            if label_point:
                for e in range(self.n_objects):
                    ax.annotate(
                        text=self.objects[e],
                        xy=(self.T[e, i], self.T[e, j]),
                        color=text_color,
                    )

            for d in range(self.n_variables):
                position = (
                    ["left", "bottom"]
                    if self.W[d, i] > self.W[:, i].mean()
                    else ["right", "top"]
                )
                ax.arrow(
                    0,
                    0,
                    self.W[d, i],
                    self.W[d, j],
                    length_includes_head=True,
                    width=0.01,
                    color=text_color,
                    alpha=0.3,
                )
                ax.annotate(
                    text=self.variables[d],
                    xy=(self.W[d, i], self.W[d, j]),
                    ha=position[0],
                    va=position[1],
                    color=text_color,
                )
            ax.set_xlabel(f"PC{i+1}")
            ax.set_ylabel(f"PC{j+1}")

            if show_legend:
                handles, labels = scatter.legend_elements()
                fig.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    labelcolor=text_color,
                )
            plt.tight_layout()
            plt.show()
        else:
            height_ratios = [1] * self.n_component
            width_ratios = [1] * self.n_component

            fig, axs = plt.subplots(
                self.n_component,
                self.n_component,
                figsize=(
                    subplot_dimensions[0] * self.n_component,
                    subplot_dimensions[1] * self.n_component,
                ),
                gridspec_kw={
                    "height_ratios": height_ratios,
                    "width_ratios": width_ratios,
                },
            )
            fig.suptitle("Biplots plot", fontsize=24, y=1, color=text_color)

            all_handles = []
            all_labels = []

            for i in range(self.n_component):
                for j in range(self.n_component):
                    ax = axs[i, j] if self.n_component > 1 else axs
                    if i != j:

                        x_data = self.T[:, i]
                        y_data = self.T[:, j]
                        colors = self.objects_colors
                        scatter = ax.scatter(
                            x_data, y_data, c=colors, label=self.objects
                        )

                        if label_point:
                            for e in range(self.n_objects):
                                ax.annotate(
                                    text=self.objects[e],
                                    xy=(self.T[e, i], self.T[e, j]),
                                    color=text_color,
                                )

                        for d in range(self.n_variables):
                            position = (
                                ["left", "bottom"]
                                if self.W[d, i] > self.W[:, i].mean()
                                else ["right", "top"]
                            )
                            ax.arrow(
                                0,
                                0,
                                self.W[d, i],
                                self.W[d, j],
                                length_includes_head=True,
                                width=0.01,
                                color=text_color,
                                alpha=0.3,
                            )
                            ax.annotate(
                                text=self.variables[d],
                                xy=(
                                    self.W[d, i],
                                    self.W[d, j],
                                ),
                                ha=position[0],
                                va=position[1],
                                color=text_color,
                            )
                        ax.set_xlabel(f"PC{i+1}")
                        ax.set_ylabel(f"PC{j+1}")

                        handles, labels = scatter.legend_elements()
                        all_handles.extend(handles)
                        all_labels.extend(labels)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            f"PC{i+1}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=20,
                            color=text_color,
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.add_patch(
                            patches.Rectangle(
                                (0, 0),
                                1,
                                1,
                                fill=False,
                                transform=ax.transAxes,
                                clip_on=False,
                            )
                        )

            if show_legend:
                fig.legend(
                    all_handles,
                    all_labels,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    labelcolor=text_color,
                )
            plt.tight_layout()
            plt.show()
