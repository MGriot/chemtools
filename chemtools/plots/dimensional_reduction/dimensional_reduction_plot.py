import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import numpy as np

from chemtools.plots.Plotter import Plotter
from chemtools.dimensional_reduction import FactorAnalysis
from chemtools.exploration import PrincipalComponentAnalysis


class DimensionalityReductionPlot(Plotter):
    """Class to generate plots for dimensionality reduction models."""

    def __init__(self, dim_reduction_model, **kwargs):
        super().__init__(**kwargs)
        self.dim_reduction_model = dim_reduction_model

    def plot_correlation_matrix(self, cmap="coolwarm", threshold=None):
        """Plots the correlation matrix of the data used in the PCA.

        Args:
            cmap (str, optional): The colormap for the heatmap. Defaults to "coolwarm".
            threshold (float, optional): The threshold for displaying text. If None,
                the midpoint of the colormap is used. Defaults to None.
        """
        fig, ax = self._create_figure(figsize=(10, 10))
        im = ax.imshow(self.dim_reduction_model.correlation_matrix, cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap, label="Correlation value")
        ax.set_xticks(np.arange(len(self.dim_reduction_model.variables)))
        ax.set_yticks(np.arange(len(self.dim_reduction_model.variables)))
        ax.set_xticklabels(self.dim_reduction_model.variables)
        ax.set_yticklabels(self.dim_reduction_model.variables)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Calculate the midpoint of the colormap if threshold is None
        if threshold is None:
            norm = plt.Normalize(
                vmin=self.dim_reduction_model.correlation_matrix.min(),
                vmax=self.dim_reduction_model.correlation_matrix.max(),
            )
            midpoint = (
                self.dim_reduction_model.correlation_matrix.max()
                + self.dim_reduction_model.correlation_matrix.min()
            ) / 2
            threshold = norm(midpoint)

        # Add correlation values as text
        for i in range(len(self.dim_reduction_model.variables)):
            for j in range(len(self.dim_reduction_model.variables)):
                color = (
                    "black"
                    if abs(self.dim_reduction_model.correlation_matrix[i, j])
                    < threshold
                    else "white"
                )
                text = ax.text(
                    j,
                    i,
                    f"{self.dim_reduction_model.correlation_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                )

        self._set_labels(ax, title="Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def plot_eigenvalues(self, criteria=None):
        """Plots the eigenvalues and highlights them based on the chosen criteria.

        Args:
            criteria (list, optional): A list of criteria to use for highlighting
                                       eigenvalues. Options are: 'greater_than_one',
                                       'variance', 'cumulative_variance',
                                       'average_eigenvalue', 'kp', 'kl', 'caec',
                                       'broken_stick'. If None, no highlighting is
                                       applied. Defaults to None.
        """
        fig, ax = self._create_figure(figsize=(8, 6))
        ax.plot(
            self.dim_reduction_model.V_ordered,
            marker="o",
            linestyle="-",
            color="b",
            label="Eigenvalues",
        )  # Plot eigenvalues
        ax.set_xticks(range(len(self.dim_reduction_model.index)))
        ax.set_xticklabels(self.dim_reduction_model.index)
        ax.grid(True)

        all_criteria = [
            "greater_than_one",
            "variance",
            "cumulative_variance",
            "average_eigenvalue",
            "kp",
            "kl",
            "caec",
            "broken_stick",
        ]

        if criteria is None:
            criteria = all_criteria
        if criteria is not None:
            for criterion in criteria:
                if criterion == "greater_than_one":
                    self._plot_eigenvalues_greater_than_one(ax)
                elif criterion == "variance":
                    self._plot_eigenvalues_variance(ax)
                elif criterion == "cumulative_variance":
                    self._plot_cumulative_variance(ax)
                elif criterion == "average_eigenvalue":
                    self._plot_average_eigenvalue_criterion(ax)
                elif criterion == "kp":
                    self._plot_KP_criterion(ax)
                elif criterion == "kl":
                    self._plot_KL_criterion(ax)
                elif criterion == "caec":
                    self._plot_CAEC_criterion(ax)
                elif criterion == "broken_stick":
                    self._plot_broken_stick(ax)
                else:
                    raise ValueError(f"Invalid criterion: {criterion}")
                # Set x-axis label based on model type
        if isinstance(self.dim_reduction_model, PrincipalComponentAnalysis):
            xlabel = r"$PC_i$"
        elif isinstance(self.dim_reduction_model, FactorAnalysis):
            xlabel = r"$F_i$"
        else:
            xlabel = r"Component $i$"
        self._set_labels(
            ax,
            xlabel=xlabel,
            ylabel="Eigenvalue",
            title="Eigenvalues with Selected Criteria",
        )
        ax.legend(loc="best")
        plt.show()

    def _plot_eigenvalues_greater_than_one(self, ax):
        """Highlights eigenvalues greater than one on the plot."""
        num_eigenvalues_greater_than_one = np.argmax(
            self.dim_reduction_model.V_ordered < 1
        )
        ax.axvline(
            x=num_eigenvalues_greater_than_one - 0.5,
            color="brown",
            linestyle="-",
            label="Eigenvalues > 1",
        )

    def _plot_eigenvalues_variance(self, ax):
        """Plots the percentage of variance explained by each principal component."""
        variance_explained = (
            self.dim_reduction_model.V_ordered
            / self.dim_reduction_model.V_ordered.sum()
        ) * 100
        ax.bar(
            x=self.dim_reduction_model.index,
            height=variance_explained,
            fill=False,
            edgecolor="darkorange",
            label="Variance Explained (%)",
        )

    def _plot_cumulative_variance(self, ax):
        """Plots the cumulative percentage of variance explained by the principal components."""
        cumulative_variance = (
            np.cumsum(
                self.dim_reduction_model.V_ordered
                / self.dim_reduction_model.V_ordered.sum()
            )
            * 100
        )
        ax.bar(
            x=self.dim_reduction_model.index,
            height=cumulative_variance,
            fill=False,
            edgecolor="black",
            linestyle="--",
            width=0.6,
            label="Cumulative Variance Explained (%)",
        )

    def _plot_average_eigenvalue_criterion(self, ax):
        """Highlights eigenvalues greater than the average eigenvalue."""
        ax.axvline(
            x=np.argmax(
                self.dim_reduction_model.V_ordered
                < self.dim_reduction_model.V_ordered.mean()
            )
            - 0.5,
            color="red",
            alpha=0.5,
            linestyle="-",
            label="AEC (Average Eigenvalue)",
        )

    def _plot_KP_criterion(self, ax):
        """Indicates the Kaiser-Piggott (KP) criterion on the plot."""
        rank = np.linalg.matrix_rank(self.dim_reduction_model.correlation_matrix)
        sum_term = sum(
            self.dim_reduction_model.V[m] / self.dim_reduction_model.V.sum()
            - 1 / self.dim_reduction_model.V.size
            for m in range(rank)
        )
        x = (
            round(
                1
                + (self.dim_reduction_model.V.size - 1)
                * (
                    1
                    - (
                        (
                            sum_term
                            + (self.dim_reduction_model.V.size - rank)
                            ** (1 / self.dim_reduction_model.V.size)
                        )
                        / (
                            2
                            * (self.dim_reduction_model.V.size - 1)
                            / self.dim_reduction_model.V.size
                        )
                    )
                )
            )
            - 1
        )
        ax.axvline(
            x=x,
            color="purple",
            alpha=0.5,
            linestyle="--",
            label="KP Criterion",
        )

    def _plot_KL_criterion(self, ax):
        """Marks the KL criterion for component selection on the plot."""
        rank = np.linalg.matrix_rank(self.dim_reduction_model.correlation_matrix)
        sum_term = sum(
            self.dim_reduction_model.V[m] / self.dim_reduction_model.V.sum()
            - 1 / self.dim_reduction_model.V.size
            for m in range(rank)
        )
        x = (
            round(
                self.dim_reduction_model.V.size
                ** (
                    1
                    - (
                        sum_term
                        + (self.dim_reduction_model.V.size - rank)
                        ** (1 / self.dim_reduction_model.V.size)
                    )
                    / (
                        2
                        * (self.dim_reduction_model.V.size - 1)
                        / self.dim_reduction_model.V.size
                    )
                )
            )
            - 1
        )
        ax.axvline(
            x=x,
            color="cyan",
            alpha=0.5,
            linestyle="-",
            label="KL Criterion",
        )

    def _plot_CAEC_criterion(self, ax):
        """Marks the CAEC (Cumulative Average Eigenvalue Criterion) on the plot."""
        ax.axvline(
            x=np.argmax(
                self.dim_reduction_model.V_ordered
                < 0.7 * self.dim_reduction_model.V_ordered.mean()
            )
            - 0.5,
            color="blue",
            alpha=0.5,
            linestyle="--",
            label="CAEC (70% of Avg. Eigenvalue)",
        )

    def _plot_broken_stick(self, ax):
        """Plots the broken stick criterion for selecting the number of PCs."""
        n = self.dim_reduction_model.V_ordered.shape[0]
        dm = (100 / n) * np.cumsum(1 / np.arange(1, n + 1)[::-1])
        ax.plot(
            self.dim_reduction_model.index, dm, color="lightgreen", label="Broken Stick"
        )

    def plot_hotteling_t2_vs_q(self, show_legend=True):
        """Plots the Hotelling's T2 statistic versus the Q statistic (Squared Prediction Error).

        Args:
            show_legend (bool, optional): Whether to display the legend.
                                        Defaults to True.
        """
        fig, ax = self._create_figure(figsize=(8, 6))

        for i in range(len(self.dim_reduction_model.Q)):
            ax.plot(
                self.dim_reduction_model.Q[i],
                self.dim_reduction_model.T2[i],
                "o",
                label=self.dim_reduction_model.objects[i],
            )

        self._set_labels(
            ax,
            xlabel=r"$Q$ (Squared Prediction Error)",
            ylabel=r"$Hotelling's T^2$",
            title="Hotelling's T2 vs. Q",
        )
        ax.grid(True)

        if show_legend:
            ax.legend(loc="best")

        plt.tight_layout()
        plt.show()

    def plot_pci_contribution(self):
        """Plots the contribution of each variable to each principal component."""
        fig, ax = self._create_figure(figsize=(10, 6))
        for i in range(self.dim_reduction_model.W.shape[1]):
            ax.plot(
                np.arange(self.dim_reduction_model.n_variables),
                self.dim_reduction_model.W[:, i],
                marker="o",
                markerfacecolor="none",
                label=f"PC$_{i+1}$",
            )

        self._set_labels(
            ax,
            xlabel="Variable",
            ylabel="Value of Loading",
            title="Contributions of Variables to Each PC",
        )

        plt.xticks(
            np.arange(self.dim_reduction_model.n_variables),
            self.dim_reduction_model.variables,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.legend(labelcolor=self.theme_color)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_loadings(self, components=None, show_arrows=True):
        """Plots the loadings of the principal components.

        Args:
            components (tuple, optional): The components to plot.
                                        Defaults to None (all components).
            show_arrows (bool, optional): Whether to display arrows for the loadings.
                                        Defaults to True.
        """
        if components is not None:
            i, j = components
            self._plot_loadings_single(i, j, show_arrows)
        else:
            self._plot_loadings_matrix(show_arrows)

    def _plot_loadings_single(self, i, j, show_arrows=True):
        """Plots loadings for a single pair of components."""
        fig, ax = self._create_figure(figsize=(5, 5))
        fig.suptitle(
            f"Loadings plot PC{i+1} vs PC{j+1}",
            fontsize=24,
            y=1,
            color=self.theme_color,
        )

        x_data = self.dim_reduction_model.W[:, i]
        y_data = self.dim_reduction_model.W[:, j]
        colors = self.dim_reduction_model.variables_colors
        scatter = ax.scatter(
            x_data, y_data, c=colors, label=self.dim_reduction_model.variables
        )

        if show_arrows:
            for d in range(self.dim_reduction_model.n_variables):
                ax.arrow(
                    0,
                    0,
                    self.dim_reduction_model.W[d, i],
                    self.dim_reduction_model.W[d, j],
                    length_includes_head=True,
                    width=0.01,
                    color=self.theme_color,
                    alpha=0.3,
                )

        for d in range(self.dim_reduction_model.n_variables):
            position = (
                ["left", "bottom"]
                if self.dim_reduction_model.W[d, i]
                > self.dim_reduction_model.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=self.dim_reduction_model.variables[d],
                xy=(self.dim_reduction_model.W[d, i], self.dim_reduction_model.W[d, j]),
                ha=position[0],
                va=position[1],
                color=self.theme_color,
            )

        self._set_labels(ax, xlabel=rf"PC$_{i+1}$", ylabel=rf"PC$_{j+1}$")

        handles, labels = scatter.legend_elements()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.theme_color,
        )
        plt.tight_layout()
        plt.show()

    def _plot_loadings_matrix(self, show_arrows=True):
        """Plots loadings for all pairs of components in a matrix."""
        height_ratios = [1] * self.dim_reduction_model.n_component
        width_ratios = [1] * self.dim_reduction_model.n_component

        fig, axs = plt.subplots(
            self.dim_reduction_model.n_component,
            self.dim_reduction_model.n_component,
            figsize=(
                5 * self.dim_reduction_model.n_component,
                5 * self.dim_reduction_model.n_component,
            ),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Loadings plot", fontsize=24, y=1, color=self.theme_color)

        for i in range(self.dim_reduction_model.n_component):
            for j in range(self.dim_reduction_model.n_component):
                ax = axs[i, j] if self.dim_reduction_model.n_component > 1 else axs
                if i != j:
                    self._plot_loadings_on_axis(ax, i, j, show_arrows)
                else:
                    self._plot_empty_loadings_on_axis(ax, i)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.theme_color,
        )
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()

    def _plot_loadings_on_axis(self, ax, i, j, show_arrows=True):
        """Plots loadings for a specific pair of components on a given axis."""
        x_data = self.dim_reduction_model.W[:, i]
        y_data = self.dim_reduction_model.W[:, j]
        colors = self.dim_reduction_model.variables_colors
        ax.scatter(x_data, y_data, c=colors, label=self.dim_reduction_model.variables)

        if show_arrows:
            for d in range(self.dim_reduction_model.n_variables):
                ax.arrow(
                    0,
                    0,
                    self.dim_reduction_model.W[d, i],
                    self.dim_reduction_model.W[d, j],
                    length_includes_head=True,
                    width=0.01,
                    color=self.theme_color,
                    alpha=0.3,
                )

        for d in range(self.dim_reduction_model.n_variables):
            position = (
                ["left", "bottom"]
                if self.dim_reduction_model.W[d, i]
                > self.dim_reduction_model.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=self.dim_reduction_model.variables[d],
                xy=(self.dim_reduction_model.W[d, i], self.dim_reduction_model.W[d, j]),
                ha=position[0],
                va=position[1],
                color=self.theme_color,
            )

        self._set_labels(ax, xlabel=rf"PC$_{i+1}$", ylabel=rf"PC$_{j+1}$")

    def _plot_empty_loadings_on_axis(self, ax, i):
        """Plots an empty space for diagonal cells in the loadings matrix."""
        ax.text(
            0.5,
            0.5,
            rf"PC$_{i+1}$",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
            color=self.theme_color,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(
            patches.Rectangle(
                (0, 0), 1, 1, fill=False, transform=ax.transAxes, clip_on=False
            )
        )

    def plot_scores(self, components=None, label_points=False):
        """Plots the scores of the principal components.

        Args:
            components (tuple, optional): The components to plot.
                                        Defaults to None (all components).
            label_points (bool, optional): Whether to label the points.
                                        Defaults to False.
        """
        if components is not None:
            i, j = components
            self._plot_scores_single(i, j, label_points)
        else:
            self._plot_scores_matrix(label_points)

    def _plot_scores_single(self, i, j, label_points=False):
        """Plots scores for a single pair of components."""
        fig, ax = self._create_figure(figsize=(5, 5))
        fig.suptitle(
            f"Scores plot PC{i+1} vs PC{j+1}",
            fontsize=24,
            y=1,
            color=self.theme_color,
        )

        x_data = self.dim_reduction_model.T[:, i]
        y_data = self.dim_reduction_model.T[:, j]
        colors = self.dim_reduction_model.objects_colors
        scatter = ax.scatter(
            x_data, y_data, c=colors, label=self.dim_reduction_model.objects
        )

        if label_points:
            for d in range(self.dim_reduction_model.n_objects):
                ax.annotate(
                    text=self.dim_reduction_model.objects[d][0],
                    xy=(
                        self.dim_reduction_model.T[d, i],
                        self.dim_reduction_model.T[d, j],
                    ),
                    color=self.theme_color,
                )

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")

        handles, labels = scatter.legend_elements()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.theme_color,
        )
        plt.tight_layout()
        plt.show()

    def _plot_scores_matrix(self, label_points=False):
        """Plots scores for all pairs of components in a matrix."""
        height_ratios = [1] * self.dim_reduction_model.n_component
        width_ratios = [1] * self.dim_reduction_model.n_component

        fig, axs = plt.subplots(
            self.dim_reduction_model.n_component,
            self.dim_reduction_model.n_component,
            figsize=(
                5 * self.dim_reduction_model.n_component,
                5 * self.dim_reduction_model.n_component,
            ),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Scores plot", fontsize=24, y=1, color=self.theme_color)

        for i in range(self.dim_reduction_model.n_component):
            for j in range(self.dim_reduction_model.n_component):
                ax = axs[i, j] if self.dim_reduction_model.n_component > 1 else axs
                if i != j:
                    self._plot_scores_on_axis(ax, i, j, label_points)
                else:
                    self._plot_empty_scores_on_axis(ax, i)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.theme_color,
        )
        plt.tight_layout()
        plt.show()

    def _plot_scores_on_axis(self, ax, i, j, label_points=False):
        """Plots scores for a specific pair of components on a given axis."""
        x_data = self.dim_reduction_model.T[:, i]
        y_data = self.dim_reduction_model.T[:, j]
        colors = self.dim_reduction_model.objects_colors
        ax.scatter(x_data, y_data, c=colors, label=self.dim_reduction_model.objects)

        if label_points:
            for d in range(self.dim_reduction_model.n_objects):
                ax.annotate(
                    text=self.dim_reduction_model.objects[d][0],
                    xy=(
                        self.dim_reduction_model.T[d, i],
                        self.dim_reduction_model.T[d, j],
                    ),
                    color=self.theme_color,
                )
        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")

    def _plot_empty_scores_on_axis(self, ax, i):
        """Plots an empty space for diagonal cells in the scores matrix."""
        ax.text(
            0.5,
            0.5,
            f"PC{i+1}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
            color=self.theme_color,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(
            patches.Rectangle(
                (0, 0), 1, 1, fill=False, transform=ax.transAxes, clip_on=False
            )
        )

    def plot_biplot(
        self,
        components=None,
        label_points=False,
        subplot_dimensions=(5, 5),
        show_legend=True,
    ):
        """Plots a biplot, combining scores and loadings in one plot.

        Args:
            components (tuple, optional): The principal components to display.
                                        Defaults to None (all components).
            label_points (bool, optional): Whether to label the score points.
                                        Defaults to False.
            subplot_dimensions (tuple, optional): Dimensions of subplots.
                                                Defaults to (5, 5).
            show_legend (bool, optional): Whether to display the legend.
                                        Defaults to True.
        """
        if components is not None:
            i, j = components
            self._plot_biplot_single(
                i, j, label_points, subplot_dimensions, show_legend
            )
        else:
            self._plot_biplot_matrix(label_points, subplot_dimensions, show_legend)

    def _plot_biplot_single(
        self, i, j, label_points=False, subplot_dimensions=(5, 5), show_legend=True
    ):
        """Plots a biplot for a single pair of components."""
        fig, ax = self._create_figure(figsize=subplot_dimensions)
        fig.suptitle(
            f"Biplot PC{i+1} vs PC{j+1}",
            fontsize=24,
            y=1,
            color=self.theme_color,
        )

        x_data = self.dim_reduction_model.T[:, i]
        y_data = self.dim_reduction_model.T[:, j]
        colors = self.dim_reduction_model.objects_colors
        scatter = ax.scatter(
            x_data, y_data, c=colors, label=self.dim_reduction_model.objects
        )

        if label_points:
            for e in range(self.dim_reduction_model.n_objects):
                ax.annotate(
                    text=self.dim_reduction_model.objects[e],
                    xy=(
                        self.dim_reduction_model.T[e, i],
                        self.dim_reduction_model.T[e, j],
                    ),
                    color=self.theme_color,
                )

        for d in range(self.dim_reduction_model.n_variables):
            ax.arrow(
                0,
                0,
                self.dim_reduction_model.W[d, i],
                self.dim_reduction_model.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=self.theme_color,
                alpha=0.3,
            )
            position = (
                ["left", "bottom"]
                if self.dim_reduction_model.W[d, i]
                > self.dim_reduction_model.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=self.dim_reduction_model.variables[d],
                xy=(self.dim_reduction_model.W[d, i], self.dim_reduction_model.W[d, j]),
                ha=position[0],
                va=position[1],
                color=self.theme_color,
            )

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")

        if show_legend:
            handles, labels = scatter.legend_elements()
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                labelcolor=self.theme_color,
            )

        plt.tight_layout()
        plt.show()

    def _plot_biplot_matrix(
        self, label_points=False, subplot_dimensions=(5, 5), show_legend=True
    ):
        """Plots biplots for all pairs of components in a matrix."""
        height_ratios = [1] * self.dim_reduction_model.n_component
        width_ratios = [1] * self.dim_reduction_model.n_component

        fig, axs = plt.subplots(
            self.dim_reduction_model.n_component,
            self.dim_reduction_model.n_component,
            figsize=(
                subplot_dimensions[0] * self.dim_reduction_model.n_component,
                subplot_dimensions[1] * self.dim_reduction_model.n_component,
            ),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Biplots plot", fontsize=24, y=1, color=self.theme_color)

        all_handles = []
        all_labels = []

        for i in range(self.dim_reduction_model.n_component):
            for j in range(self.dim_reduction_model.n_component):
                ax = axs[i, j] if self.dim_reduction_model.n_component > 1 else axs
                if i != j:
                    (
                        handles,
                        labels,
                    ) = self._plot_biplot_on_axis(
                        ax, i, j, label_points
                    )  # Get handles and labels
                    all_handles.extend(handles)
                    all_labels.extend(labels)
                else:
                    self._plot_empty_biplot_on_axis(ax, i)

        if show_legend:
            fig.legend(
                all_handles,
                all_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                labelcolor=self.theme_color,
            )

        plt.tight_layout()
        plt.show()

    def _plot_biplot_on_axis(self, ax, i, j, label_points=False):
        """Plots a biplot for a specific pair of components on a given axis."""
        x_data = self.dim_reduction_model.T[:, i]
        y_data = self.dim_reduction_model.T[:, j]
        colors = self.dim_reduction_model.objects_colors
        scatter = ax.scatter(
            x_data, y_data, c=colors, label=self.dim_reduction_model.objects
        )  # Assign scatter to a variable

        if label_points:
            for e in range(self.dim_reduction_model.n_objects):
                ax.annotate(
                    text=self.dim_reduction_model.objects[e],
                    xy=(
                        self.dim_reduction_model.T[e, i],
                        self.dim_reduction_model.T[e, j],
                    ),
                    color=self.theme_color,
                )
        for d in range(self.dim_reduction_model.n_variables):
            ax.arrow(
                0,
                0,
                self.dim_reduction_model.W[d, i],
                self.dim_reduction_model.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=self.theme_color,
                alpha=0.3,
            )
            position = (
                ["left", "bottom"]
                if self.dim_reduction_model.W[d, i]
                > self.dim_reduction_model.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=self.dim_reduction_model.variables[d],
                xy=(self.dim_reduction_model.W[d, i], self.dim_reduction_model.W[d, j]),
                ha=position[0],
                va=position[1],
                color=self.theme_color,
            )
        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")
        return scatter.legend_elements()  # Return handles and labels for the legend

    def _plot_empty_biplot_on_axis(self, ax, i):
        """Plots an empty space for diagonal cells in the biplot matrix."""
        ax.text(
            0.5,
            0.5,
            f"PC{i+1}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
            color=self.theme_color,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(
            patches.Rectangle(
                (0, 0), 1, 1, fill=False, transform=ax.transAxes, clip_on=False
            )
        )

    def plot_explained_variance_ellipse(self, components=(0, 1), confidence_level=0.95):
        """Plots the explained variance ellipse on the scores plot of specified components.

        Args:
            components (tuple, optional): The components to plot. Defaults to (0, 1).
            confidence_level (float, optional): The confidence level for the ellipse.
                Defaults to 0.95.
        """
        i, j = components
        fig, ax = plt.subplots(figsize=(5, 5))

        # Plot scores
        x_data = self.dim_reduction_model.T[:, i]
        y_data = self.dim_reduction_model.T[:, j]
        colors = self.dim_reduction_model.objects_colors
        scatter = ax.scatter(
            x_data, y_data, c=colors, label=self.dim_reduction_model.objects
        )

        # Calculate ellipse parameters
        covariance_matrix = np.cov(x_data, y_data)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * np.sqrt(eigenvalues[0] * 2 * np.log(1 / (1 - confidence_level)))
        height = 2 * np.sqrt(eigenvalues[1] * 2 * np.log(1 / (1 - confidence_level)))

        # Plot ellipse
        ellipse = Ellipse(
            (np.mean(x_data), np.mean(y_data)),
            width,
            height,
            angle,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            label=f"{confidence_level * 100:.1f}% Explained Variance",
        )
        ax.add_patch(ellipse)

        # Set labels and legend
        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")
        handles, labels = scatter.legend_elements()
        handles.append(ellipse)
        labels.append(f"{confidence_level * 100:.1f}% Explained Variance")
        plt.legend(handles, labels, loc="best")

        plt.tight_layout()
        plt.show()
