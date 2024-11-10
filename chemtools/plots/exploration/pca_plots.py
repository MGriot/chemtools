import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import numpy as np

from chemtools.plots.Plotter import Plotter


class PCAplots(Plotter):
    """Class to generate various plots related to Principal Component Analysis (PCA).

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        **kwargs: Keyword arguments passed to the Plotter class.
    """

    def __init__(self, pca_object, **kwargs):
        super().__init__(**kwargs)
        self.pca_object = pca_object

    def plot_correlation_matrix(self, cmap="coolwarm", threshold=None):
        """Plots the correlation matrix of the data used in the PCA.

        Args:
            cmap (str, optional): The colormap for the heatmap. Defaults to "coolwarm".
            threshold (float, optional): The threshold for displaying text. If None,
                the midpoint of the colormap is used. Defaults to None.
        """
        fig, ax = self._create_figure(figsize=(10, 10))
        im = ax.imshow(self.pca_object.correlation_matrix, cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap, label="Correlation value")
        ax.set_xticks(np.arange(len(self.pca_object.variables)))
        ax.set_yticks(np.arange(len(self.pca_object.variables)))
        ax.set_xticklabels(self.pca_object.variables)
        ax.set_yticklabels(self.pca_object.variables)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Calculate the midpoint of the colormap if threshold is None
        if threshold is None:
            norm = plt.Normalize(
                vmin=self.pca_object.correlation_matrix.min(),
                vmax=self.pca_object.correlation_matrix.max(),
            )
            midpoint = (
                self.pca_object.correlation_matrix.max()
                + self.pca_object.correlation_matrix.min()
            ) / 2
            threshold = norm(midpoint)

        # Add correlation values as text
        for i in range(len(self.pca_object.variables)):
            for j in range(len(self.pca_object.variables)):
                color = (
                    "black"
                    if abs(self.pca_object.correlation_matrix[i, j]) < threshold
                    else "white"
                )
                text = ax.text(
                    j,
                    i,
                    f"{self.pca_object.correlation_matrix[i, j]:.2f}",
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
            self.pca_object.V_ordered,
            marker="o",
            linestyle="-",
            color="b",
            label="Eigenvalues",
        )  # Plot eigenvalues
        ax.set_xticks(range(len(self.pca_object.PC_index)))
        ax.set_xticklabels(self.pca_object.PC_index)
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
        self._set_labels(
            ax,
            xlabel=r"$PC_i$",
            ylabel="Eigenvalue",
            title="Eigenvalues with Selected Criteria",
        )
        ax.legend(loc="best")
        plt.show()

    def _plot_eigenvalues_greater_than_one(self, ax):
        """Highlights eigenvalues greater than one on the plot."""
        num_eigenvalues_greater_than_one = np.argmax(self.pca_object.V_ordered < 1)
        ax.axvline(
            x=num_eigenvalues_greater_than_one - 0.5,
            color="brown",
            linestyle="-",
            label="Eigenvalues > 1",
        )

    def _plot_eigenvalues_variance(self, ax):
        """Plots the percentage of variance explained by each principal component."""
        variance_explained = (
            self.pca_object.V_ordered / self.pca_object.V_ordered.sum()
        ) * 100
        ax.bar(
            x=self.pca_object.PC_index,
            height=variance_explained,
            fill=False,
            edgecolor="darkorange",
            label="Variance Explained (%)",
        )

    def _plot_cumulative_variance(self, ax):
        """Plots the cumulative percentage of variance explained by the principal components."""
        cumulative_variance = (
            np.cumsum(self.pca_object.V_ordered / self.pca_object.V_ordered.sum()) * 100
        )
        ax.bar(
            x=self.pca_object.PC_index,
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
            x=np.argmax(self.pca_object.V_ordered < self.pca_object.V_ordered.mean())
            - 0.5,
            color="red",
            alpha=0.5,
            linestyle="-",
            label="AEC (Average Eigenvalue)",
        )

    def _plot_KP_criterion(self, ax):
        """Indicates the Kaiser-Piggott (KP) criterion on the plot."""
        rank = np.linalg.matrix_rank(self.pca_object.correlation_matrix)
        sum_term = sum(
            self.pca_object.V[m] / self.pca_object.V.sum() - 1 / self.pca_object.V.size
            for m in range(rank)
        )
        x = (
            round(
                1
                + (self.pca_object.V.size - 1)
                * (
                    1
                    - (
                        (
                            sum_term
                            + (self.pca_object.V.size - rank)
                            ** (1 / self.pca_object.V.size)
                        )
                        / (2 * (self.pca_object.V.size - 1) / self.pca_object.V.size)
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
        rank = np.linalg.matrix_rank(self.pca_object.correlation_matrix)
        sum_term = sum(
            self.pca_object.V[m] / self.pca_object.V.sum() - 1 / self.pca_object.V.size
            for m in range(rank)
        )
        x = (
            round(
                self.pca_object.V.size
                ** (
                    1
                    - (
                        sum_term
                        + (self.pca_object.V.size - rank)
                        ** (1 / self.pca_object.V.size)
                    )
                    / (2 * (self.pca_object.V.size - 1) / self.pca_object.V.size)
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
                self.pca_object.V_ordered < 0.7 * self.pca_object.V_ordered.mean()
            )
            - 0.5,
            color="blue",
            alpha=0.5,
            linestyle="--",
            label="CAEC (70% of Avg. Eigenvalue)",
        )

    def _plot_broken_stick(self, ax):
        """Plots the broken stick criterion for selecting the number of PCs."""
        n = self.pca_object.V_ordered.shape[0]
        dm = (100 / n) * np.cumsum(1 / np.arange(1, n + 1)[::-1])
        ax.plot(self.pca_object.PC_index, dm, color="lightgreen", label="Broken Stick")

    def plot_hotteling_t2_vs_q(self, show_legend=True):
        """Plots the Hotelling's T2 statistic versus the Q statistic (Squared Prediction Error).

        Args:
            show_legend (bool, optional): Whether to display the legend.
                                        Defaults to True.
        """
        fig, ax = self._create_figure(figsize=(8, 6))

        for i in range(len(self.pca_object.Q)):
            ax.plot(
                self.pca_object.Q[i],
                self.pca_object.T2[i],
                "o",
                label=self.pca_object.objects[i],
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
        for i in range(self.pca_object.W.shape[1]):
            ax.plot(
                np.arange(self.pca_object.n_variables),
                self.pca_object.W[:, i],
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
            np.arange(self.pca_object.n_variables),
            self.pca_object.variables,
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

        x_data = self.pca_object.W[:, i]
        y_data = self.pca_object.W[:, j]
        colors = self.pca_object.variables_colors
        scatter = ax.scatter(x_data, y_data, c=colors, label=self.pca_object.variables)

        if show_arrows:
            for d in range(self.pca_object.n_variables):
                ax.arrow(
                    0,
                    0,
                    self.pca_object.W[d, i],
                    self.pca_object.W[d, j],
                    length_includes_head=True,
                    width=0.01,
                    color=self.theme_color,
                    alpha=0.3,
                )

        for d in range(self.pca_object.n_variables):
            position = (
                ["left", "bottom"]
                if self.pca_object.W[d, i] > self.pca_object.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=self.pca_object.variables[d],
                xy=(self.pca_object.W[d, i], self.pca_object.W[d, j]),
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
        height_ratios = [1] * self.pca_object.n_component
        width_ratios = [1] * self.pca_object.n_component

        fig, axs = plt.subplots(
            self.pca_object.n_component,
            self.pca_object.n_component,
            figsize=(
                5 * self.pca_object.n_component,
                5 * self.pca_object.n_component,
            ),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Loadings plot", fontsize=24, y=1, color=self.theme_color)

        for i in range(self.pca_object.n_component):
            for j in range(self.pca_object.n_component):
                ax = axs[i, j] if self.pca_object.n_component > 1 else axs
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
        x_data = self.pca_object.W[:, i]
        y_data = self.pca_object.W[:, j]
        colors = self.pca_object.variables_colors
        ax.scatter(x_data, y_data, c=colors, label=self.pca_object.variables)

        if show_arrows:
            for d in range(self.pca_object.n_variables):
                ax.arrow(
                    0,
                    0,
                    self.pca_object.W[d, i],
                    self.pca_object.W[d, j],
                    length_includes_head=True,
                    width=0.01,
                    color=self.theme_color,
                    alpha=0.3,
                )

        for d in range(self.pca_object.n_variables):
            position = (
                ["left", "bottom"]
                if self.pca_object.W[d, i] > self.pca_object.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=self.pca_object.variables[d],
                xy=(self.pca_object.W[d, i], self.pca_object.W[d, j]),
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

        x_data = self.pca_object.T[:, i]
        y_data = self.pca_object.T[:, j]
        colors = self.pca_object.objects_colors
        scatter = ax.scatter(x_data, y_data, c=colors, label=self.pca_object.objects)

        if label_points:
            for d in range(self.pca_object.n_objects):
                ax.annotate(
                    text=self.pca_object.objects[d][0],
                    xy=(self.pca_object.T[d, i], self.pca_object.T[d, j]),
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
        height_ratios = [1] * self.pca_object.n_component
        width_ratios = [1] * self.pca_object.n_component

        fig, axs = plt.subplots(
            self.pca_object.n_component,
            self.pca_object.n_component,
            figsize=(
                5 * self.pca_object.n_component,
                5 * self.pca_object.n_component,
            ),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Scores plot", fontsize=24, y=1, color=self.theme_color)

        for i in range(self.pca_object.n_component):
            for j in range(self.pca_object.n_component):
                ax = axs[i, j] if self.pca_object.n_component > 1 else axs
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
        x_data = self.pca_object.T[:, i]
        y_data = self.pca_object.T[:, j]
        colors = self.pca_object.objects_colors
        ax.scatter(x_data, y_data, c=colors, label=self.pca_object.objects)

        if label_points:
            for d in range(self.pca_object.n_objects):
                ax.annotate(
                    text=self.pca_object.objects[d][0],
                    xy=(self.pca_object.T[d, i], self.pca_object.T[d, j]),
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

        x_data = self.pca_object.T[:, i]
        y_data = self.pca_object.T[:, j]
        colors = self.pca_object.objects_colors
        scatter = ax.scatter(x_data, y_data, c=colors, label=self.pca_object.objects)

        if label_points:
            for e in range(self.pca_object.n_objects):
                ax.annotate(
                    text=self.pca_object.objects[e],
                    xy=(self.pca_object.T[e, i], self.pca_object.T[e, j]),
                    color=self.theme_color,
                )

        for d in range(self.pca_object.n_variables):
            ax.arrow(
                0,
                0,
                self.pca_object.W[d, i],
                self.pca_object.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=self.theme_color,
                alpha=0.3,
            )
            position = (
                ["left", "bottom"]
                if self.pca_object.W[d, i] > self.pca_object.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=self.pca_object.variables[d],
                xy=(self.pca_object.W[d, i], self.pca_object.W[d, j]),
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
        height_ratios = [1] * self.pca_object.n_component
        width_ratios = [1] * self.pca_object.n_component

        fig, axs = plt.subplots(
            self.pca_object.n_component,
            self.pca_object.n_component,
            figsize=(
                subplot_dimensions[0] * self.pca_object.n_component,
                subplot_dimensions[1] * self.pca_object.n_component,
            ),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Biplots plot", fontsize=24, y=1, color=self.theme_color)

        all_handles = []
        all_labels = []

        for i in range(self.pca_object.n_component):
            for j in range(self.pca_object.n_component):
                ax = axs[i, j] if self.pca_object.n_component > 1 else axs
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
        x_data = self.pca_object.T[:, i]
        y_data = self.pca_object.T[:, j]
        colors = self.pca_object.objects_colors
        scatter = ax.scatter(
            x_data, y_data, c=colors, label=self.pca_object.objects
        )  # Assign scatter to a variable

        if label_points:
            for e in range(self.pca_object.n_objects):
                ax.annotate(
                    text=self.pca_object.objects[e],
                    xy=(self.pca_object.T[e, i], self.pca_object.T[e, j]),
                    color=self.theme_color,
                )

        for d in range(self.pca_object.n_variables):
            ax.arrow(
                0,
                0,
                self.pca_object.W[d, i],
                self.pca_object.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=self.theme_color,
                alpha=0.3,
            )
            position = (
                ["left", "bottom"]
                if self.pca_object.W[d, i] > self.pca_object.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=self.pca_object.variables[d],
                xy=(self.pca_object.W[d, i], self.pca_object.W[d, j]),
                ha=position[0],
                va=position[1],
                color=self.theme_color,
            )

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")
        handles, labels = (
            scatter.legend_elements()
        )  # Get handles and labels from the scatter
        return handles, labels  # Return handles and labels

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

    def plot_classes_pca(
        self, class_labels, new_data_point=None, colors=None, ellipse_std=2
    ):
        """Plots multiple classes in PCA space with centroids,
        class boundaries, and an optional new data point.

        Args:
            class_labels (list): A list of class labels corresponding to the PCA models.
            new_data_point (ndarray, optional): The new data point to visualize
                                            (shape: 1 x n_variables). Defaults to None.
            colors (list, optional): A list of colors to use for each class. Defaults to None.
            ellipse_std (float, optional): The number of standard deviations to use for
                                        the ellipse radius. Defaults to 2.
        """
        pca_models = self.pca_object
        if colors is None:
            colors = plt.cm.get_cmap("viridis", len(pca_models)).colors

        fig, ax = self._create_figure()

        for i, pca in enumerate(pca_models):
            class_scores = pca.T
            ax.scatter(
                class_scores[:, 0],
                class_scores[:, 1],
                color=colors[i],
                label=class_labels[i],
            )

            centroid = np.mean(class_scores, axis=0)
            ax.plot(
                centroid[0],
                centroid[1],
                marker="*",
                markersize=10,
                color="black",
                markeredgecolor="white",
            )

            cov = np.cov(class_scores, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * ellipse_std * np.sqrt(eigenvalues)
            ellipse = Ellipse(
                xy=centroid,
                width=width,
                height=height,
                angle=angle,
                edgecolor=colors[i],
                facecolor="none",
                linewidth=2,
            )
            ax.add_patch(ellipse)

            if new_data_point is not None:
                new_data_projection = pca.transform(new_data_point)
                ax.scatter(
                    new_data_projection[:, 0],
                    new_data_projection[:, 1],
                    color=colors[i],
                    marker="o",
                    s=100,
                    label=f"New Data (Proj. on {class_labels[i]})",
                )

        self._set_labels(
            ax, xlabel="PC1", ylabel="PC2", title="Class Separation in PCA Space"
        )
        plt.legend()
        plt.show()
