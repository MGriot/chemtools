import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from chemtools.plots.Plotter import Plotter
from chemtools.dimensional_reduction import FactorAnalysis
from chemtools.exploration import PrincipalComponentAnalysis


class DimensionalityReductionPlot(Plotter):
    """Class for dimensional reduction visualization."""

    def __init__(self, dim_reduction_model, **kwargs):
        super().__init__(**kwargs)
        self.dim_reduction_model = dim_reduction_model
        # Default color scheme
        self.colors = {
            "line_color": "#1f77b4",  # Default blue color
            "scatter_color": "#2ca02c",  # Default green color
            "bar_color": "#ff7f0e",  # Default orange color
            "text_color": "#000000",  # Black
            "grid_color": "#cccccc",  # Light gray
        }

    def plot_correlation_matrix(self, cmap="coolwarm", threshold=None):
        fig, ax = self._create_figure(figsize=(10, 10))

        # Update to use theme colors
        im = ax.imshow(self.dim_reduction_model.correlation_matrix, cmap=cmap)

        # Add colorbar with themed text
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors=self.colors["text_color"])

        # Apply themed text colors
        [t.set_color(self.colors["text_color"]) for t in ax.get_xticklabels()]
        [t.set_color(self.colors["text_color"]) for t in ax.get_yticklabels()]

        fig = self.apply_style_preset(fig)
        return fig

    def plot_eigenvalues(self, criteria=None):
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(self.dim_reduction_model.V_ordered) + 1),
            self.dim_reduction_model.V_ordered,
            "bo-",
            color=self.colors["line_color"],
            linewidth=2,
        )
        plt.xlabel("Component Number")
        plt.ylabel("Eigenvalue")
        plt.title("Scree Plot")
        plt.grid(True, color=self.colors["grid_color"])
        plt.show()

    def _plot_eigenvalues_greater_than_one(self, ax):
        num_eigenvalues_greater_than_one = np.argmax(
            self.dim_reduction_model.V_ordered < 1
        )
        ax.axvline(
            x=num_eigenvalues_greater_than_one - 0.5,
            color=self.colors["highlight_color"],
            linestyle="-",
            label="Eigenvalues > 1",
        )

    def _plot_eigenvalues_variance(self, ax):
        variance_explained = (
            self.dim_reduction_model.V_ordered
            / self.dim_reduction_model.V_ordered.sum()
        ) * 100
        ax.bar(
            x=self.dim_reduction_model.index,
            height=variance_explained,
            fill=False,
            edgecolor=self.colors["bar_edge_color"],
            label="Variance Explained (%)",
        )

    def _plot_cumulative_variance(self, ax):
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
            edgecolor=self.colors["bar_edge_color"],
            linestyle="--",
            width=0.6,
            label="Cumulative Variance Explained (%)",
        )

    def _plot_average_eigenvalue_criterion(self, ax):
        ax.axvline(
            x=np.argmax(
                self.dim_reduction_model.V_ordered
                < self.dim_reduction_model.V_ordered.mean()
            )
            - 0.5,
            color=self.colors["highlight_color"],
            alpha=0.5,
            linestyle="-",
            label="AEC (Average Eigenvalue)",
        )

    def _plot_KP_criterion(self, ax):
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
            color=self.colors["highlight_color"],
            alpha=0.5,
            linestyle="--",
            label="KP Criterion",
        )

    def _plot_KL_criterion(self, ax):
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
            color=self.colors["highlight_color"],
            alpha=0.5,
            linestyle="-",
            label="KL Criterion",
        )

    def _plot_CAEC_criterion(self, ax):
        ax.axvline(
            x=np.argmax(
                self.dim_reduction_model.V_ordered
                < 0.7 * self.dim_reduction_model.V_ordered.mean()
            )
            - 0.5,
            color=self.colors["highlight_color"],
            alpha=0.5,
            linestyle="--",
            label="CAEC (70% of Avg. Eigenvalue)",
        )

    def _plot_broken_stick(self, ax):
        n = self.dim_reduction_model.V_ordered.shape[0]
        dm = (100 / n) * np.cumsum(1 / np.arange(1, n + 1)[::-1])
        ax.plot(
            self.dim_reduction_model.index,
            dm,
            color=self.colors["line_color"],
            label="Broken Stick",
        )

    def plot_hotteling_t2_vs_q(self, show_legend=True):
        fig, ax = self._create_figure(figsize=(8, 6))

        for i in range(len(self.dim_reduction_model.Q)):
            ax.plot(
                self.dim_reduction_model.Q[i],
                self.dim_reduction_model.T2[i],
                "o",
                label=self.dim_reduction_model.objects[i],
                color=self.colors["scatter_color"],
            )

        self._set_labels(
            ax,
            xlabel=r"$Q$ (Squared Prediction Error)",
            ylabel=r"$Hotelling's T^2$",
            title="Hotelling's T2 vs. Q",
        )
        ax.grid(True, color=self.colors["grid_color"])

        if show_legend:
            ax.legend(loc="best", labelcolor=self.colors["text_color"])

        fig = self.apply_style_preset(fig)
        return fig

    def plot_pci_contribution(self):
        fig, ax = self._create_figure(figsize=(10, 6))
        for i in range(self.dim_reduction_model.W.shape[1]):
            ax.plot(
                np.arange(self.dim_reduction_model.n_variables),
                self.dim_reduction_model.W[:, i],
                marker="o",
                markerfacecolor="none",
                label=f"PC$_{i+1}$",
                color=self.colors["line_color"],
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
            color=self.colors["text_color"],
        )
        ax.legend(labelcolor=self.colors["text_color"])
        ax.grid(True, color=self.colors["grid_color"])
        fig = self.apply_style_preset(fig)
        return fig

    def plot_loadings(self, components=None, show_arrows=True):
        if components is not None:
            i, j = components
            self._plot_loadings_single(i, j, show_arrows)
        else:
            self._plot_loadings_matrix(show_arrows)

    def _plot_loadings_single(self, i, j, show_arrows=True):
        fig, ax = self._create_figure(figsize=(5, 5))
        fig.suptitle(
            f"Loadings plot PC{i+1} vs PC{j+1}",
            fontsize=24,
            y=1,
            color=self.colors["text_color"],
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
                    color=self.colors["arrow_color"],
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
                color=self.colors["text_color"],
            )

        self._set_labels(ax, xlabel=rf"PC$_{i+1}$", ylabel=rf"PC$_{j+1}$")

        handles, labels = scatter.legend_elements()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.colors["text_color"],
        )
        fig = self.apply_style_preset(fig)
        return fig

    def _plot_loadings_matrix(self, show_arrows=True):
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
        fig.suptitle("Loadings plot", fontsize=24, y=1, color=self.colors["text_color"])

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
            labelcolor=self.colors["text_color"],
        )
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        fig = self.apply_style_preset(fig)
        return fig

    def _plot_loadings_on_axis(self, ax, i, j, show_arrows=True):
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
                    color=self.colors["arrow_color"],
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
                color=self.colors["text_color"],
            )

        self._set_labels(ax, xlabel=rf"PC$_{i+1}$", ylabel=rf"PC$_{j+1}$")

    def _plot_empty_loadings_on_axis(self, ax, i):
        ax.text(
            0.5,
            0.5,
            rf"PC$_{i+1}$",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
            color=self.colors["text_color"],
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
        if components is not None:
            i, j = components
            self._plot_scores_single(i, j, label_points)
        else:
            self._plot_scores_matrix(label_points)

    def _plot_scores_single(self, i, j, label_points=False):
        fig, ax = self._create_figure(figsize=(5, 5))
        fig.suptitle(
            f"Scores plot PC{i+1} vs PC{j+1}",
            fontsize=24,
            y=1,
            color=self.colors["text_color"],
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
                    color=self.colors["text_color"],
                )

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")

        handles, labels = scatter.legend_elements()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.colors["text_color"],
        )
        fig = self.apply_style_preset(fig)
        return fig

    def _plot_scores_matrix(self, label_points=False):
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
        fig.suptitle("Scores plot", fontsize=24, y=1, color=self.colors["text_color"])

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
            labelcolor=self.colors["text_color"],
        )
        fig = self.apply_style_preset(fig)
        return fig

    def _plot_scores_on_axis(self, ax, i, j, label_points=False):
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
                    color=self.colors["text_color"],
                )
        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")

    def _plot_empty_scores_on_axis(self, ax, i):
        ax.text(
            0.5,
            0.5,
            f"PC{i+1}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
            color=self.colors["text_color"],
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
        fig, ax = self._create_figure(figsize=subplot_dimensions)
        fig.suptitle(
            f"Biplot PC{i+1} vs PC{j+1}",
            fontsize=24,
            y=1,
            color=self.colors["text_color"],
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
                    color=self.colors["text_color"],
                )

        for d in range(self.dim_reduction_model.n_variables):
            ax.arrow(
                0,
                0,
                self.dim_reduction_model.W[d, i],
                self.dim_reduction_model.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=self.colors["arrow_color"],
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
                color=self.colors["text_color"],
            )

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")

        if show_legend:
            handles, labels = scatter.legend_elements()
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                labelcolor=self.colors["text_color"],
            )

        fig = self.apply_style_preset(fig)
        return fig

    def _plot_biplot_matrix(
        self, label_points=False, subplot_dimensions=(5, 5), show_legend=True
    ):
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
        fig.suptitle("Biplots plot", fontsize=24, y=1, color=self.colors["text_color"])

        all_handles = []
        all_labels = []

        for i in range(self.dim_reduction_model.n_component):
            for j in range(self.dim_reduction_model.n_component):
                ax = axs[i, j] if self.dim_reduction_model.n_component > 1 else axs
                if i != j:
                    (
                        handles,
                        labels,
                    ) = self._plot_biplot_on_axis(ax, i, j, label_points)
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
                labelcolor=self.colors["text_color"],
            )

        fig = self.apply_style_preset(fig)
        return fig

    def _plot_biplot_on_axis(self, ax, i, j, label_points=False):
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
                    color=self.colors["text_color"],
                )
        for d in range(self.dim_reduction_model.n_variables):
            ax.arrow(
                0,
                0,
                self.dim_reduction_model.W[d, i],
                self.dim_reduction_model.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=self.colors["arrow_color"],
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
                color=self.colors["text_color"],
            )
        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")
        return scatter.legend_elements()

    def _plot_empty_biplot_on_axis(self, ax, i):
        ax.text(
            0.5,
            0.5,
            f"PC{i+1}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
            color=self.colors["text_color"],
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
        i, j = components
        fig, ax = plt.subplots(figsize=(5, 5))

        x_data = self.dim_reduction_model.T[:, i]
        y_data = self.dim_reduction_model.T[:, j]
        colors = self.dim_reduction_model.objects_colors
        scatter = ax.scatter(
            x_data, y_data, c=colors, label=self.dim_reduction_model.objects
        )

        covariance_matrix = np.cov(x_data, y_data)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * np.sqrt(eigenvalues[0] * 2 * np.log(1 / (1 - confidence_level)))
        height = 2 * np.sqrt(eigenvalues[1] * 2 * np.log(1 / (1 - confidence_level)))

        ellipse = Ellipse(
            (np.mean(x_data), np.mean(y_data)),
            width,
            height,
            angle,
            facecolor="none",
            edgecolor=self.colors["highlight_color"],
            linestyle="--",
            label=f"{confidence_level * 100:.1f}% Explained Variance",
        )
        ax.add_patch(ellipse)

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")
        handles, labels = scatter.legend_elements()
        handles.append(ellipse)
        labels.append(f"{confidence_level * 100:.1f}% Explained Variance")
        plt.legend(handles, labels, loc="best", labelcolor=self.colors["text_color"])

        fig = self.apply_style_preset(fig)
        return fig
