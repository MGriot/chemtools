import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import numpy as np

from ..base import BasePlotter


class PCAplots(BasePlotter):
    """Class to generate various plots related to Principal Component Analysis (PCA).

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        **kwargs: Keyword arguments passed to the Plotter class.
    """

    def __init__(self, pca_object, library="matplotlib", theme="classic_professional_light", style_preset="default", **kwargs):
        super().__init__(library=library, theme=theme, style_preset=style_preset, **kwargs)
        self.pca_object = pca_object

    def _normalize_colors(self, colors):
        """Convert color arrays to proper format for matplotlib."""
        if colors is None:
            return self.colors["theme_color"]
        if isinstance(colors, list):
            return colors
        return [colors] * len(self.pca_object.variables)

    def plot_correlation_matrix(self, cmap="coolwarm", threshold=None, **kwargs):
        """Plots the correlation matrix of the data used in the PCA."""
        params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=params["figsize"])

        if not hasattr(self.pca_object, "correlation_matrix"):
            raise AttributeError("Model must have 'correlation_matrix'.")

        im = ax.imshow(self.pca_object.correlation_matrix, cmap=cmap)

        # Add colorbar with themed styling
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Correlation value", color=self.colors["text_color"])
        cbar.ax.yaxis.set_tick_params(color=self.colors["text_color"])
        plt.setp(cbar.ax.get_yticklabels(), color=self.colors["text_color"])

        # Set up axes
        ax.set_xticks(np.arange(len(self.pca_object.variables)))
        ax.set_yticks(np.arange(len(self.pca_object.variables)))
        ax.set_xticklabels(self.pca_object.variables, rotation=45, ha="right")
        ax.set_yticklabels(self.pca_object.variables)

        # Calculate threshold
        if threshold is None:
            threshold = (
                self.pca_object.correlation_matrix.max()
                + self.pca_object.correlation_matrix.min()
            ) / 2

        # Add correlation values as text
        for i in range(len(self.pca_object.variables)):
            for j in range(len(self.pca_object.variables)):
                text_color = (
                    self.colors["text_color"]
                    if abs(self.pca_object.correlation_matrix[i, j]) < threshold
                    else self.colors["bg_color"]
                )
                ax.text(
                    j,
                    i,
                    f"{self.pca_object.correlation_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

        # Set title and apply style
        self._set_labels(ax, subplot_title=params.get("subplot_title", "Correlation Matrix"))
        self._apply_common_layout(fig, params)
        return fig

    def plot_eigenvalues(self, criteria=None, **kwargs):
        """Plots the eigenvalues and highlights them based on the chosen criteria.

        Args:
            criteria (list, optional): A list of criteria to use for highlighting
                                       eigenvalues. Options are: 'greater_than_one',
                                       'variance', 'cumulative_variance',
                                       'average_eigenvalue', 'kp', 'kl', 'caec',
                                       'broken_stick'. If None, no highlighting is
                                       applied. Defaults to None.
        """
        params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=params["figsize"])

        # Main eigenvalues plot with theme colors
        ax.plot(
            range(len(self.pca_object.V_ordered)),
            self.pca_object.V_ordered,
            marker="o",
            linestyle="-",
            color=self.colors["theme_color"],
            label="Eigenvalues",
        )

        # Set up axes with theme colors
        ax.set_xticks(range(len(self.pca_object.PC_index)))
        ax.grid(False)

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
            self._add_eigenvalue_criteria(ax, criteria)

        # Set labels and title with theme colors
        self._set_labels(
            ax,
            xlabel=r"$PC_i$",
            ylabel="Eigenvalue",
            subplot_title=params.get("subplot_title", "Eigenvalues with Selected Criteria"),
        )

        self._apply_common_layout(fig, params)
        return fig

    def _add_eigenvalue_criteria(self, ax, criteria):
        """Add eigenvalue selection criteria with themed colors."""
        criteria_colors = {
            "greater_than_one": self.colors["theme_color"],
            "variance": self.colors["accent_color"],
            "cumulative_variance": self.colors["text_color"],
            "average_eigenvalue": self.colors["theme_color"],
            "kp": self.colors["accent_color"],
            "kl": self.colors["theme_color"],
            "caec": self.colors["accent_color"],
            "broken_stick": self.colors["theme_color"],
        }

        for criterion in criteria:
            if criterion == "greater_than_one":
                self._plot_eigenvalues_greater_than_one(ax, criteria_colors)
            elif criterion == "variance":
                self._plot_eigenvalues_variance(ax, criteria_colors)
            elif criterion == "cumulative_variance":
                self._plot_cumulative_variance(ax, criteria_colors)
            elif criterion == "average_eigenvalue":
                self._plot_average_eigenvalue_criterion(ax, criteria_colors)
            elif criterion == "kp":
                self._plot_KP_criterion(ax, criteria_colors)
            elif criterion == "kl":
                self._plot_KL_criterion(ax, criteria_colors)
            elif criterion == "caec":
                self._plot_CAEC_criterion(ax, criteria_colors)
            elif criterion == "broken_stick":
                self._plot_broken_stick(ax, criteria_colors)
            else:
                raise ValueError(f"Invalid criterion: {criterion}")

    def _plot_eigenvalues_greater_than_one(self, ax, criteria_colors):
        """Highlights eigenvalues greater than one."""
        num_eigenvalues_greater_than_one = np.argmax(self.pca_object.V_ordered < 1)
        ax.axvline(
            x=num_eigenvalues_greater_than_one - 0.5,
            color=criteria_colors["greater_than_one"],
            linestyle="-",
            label="Eigenvalues > 1",
        )

    def _plot_eigenvalues_variance(self, ax, criteria_colors):
        """Plots the percentage of variance explained by each principal component."""
        variance_explained = (
            self.pca_object.V_ordered / self.pca_object.V_ordered.sum()
        ) * 100
        ax.bar(
            x=self.pca_object.PC_index,
            height=variance_explained,
            fill=False,
            edgecolor=criteria_colors["variance"],
            label="Variance Explained (%)",
        )

    def _plot_cumulative_variance(self, ax, criteria_colors):
        """Plots the cumulative percentage of variance explained by the principal components."""
        cumulative_variance = (
            np.cumsum(self.pca_object.V_ordered / self.pca_object.V_ordered.sum()) * 100
        )
        ax.bar(
            x=self.pca_object.PC_index,
            height=cumulative_variance,
            fill=False,
            edgecolor=criteria_colors["cumulative_variance"],
            linestyle="--",
            width=0.6,
            label="Cumulative Variance Explained (%)",
        )

    def _plot_average_eigenvalue_criterion(self, ax, criteria_colors):
        """Highlights eigenvalues greater than the average eigenvalue."""
        ax.axvline(
            x=np.argmax(self.pca_object.V_ordered < self.pca_object.V_ordered.mean())
            - 0.5,
            color=criteria_colors["average_eigenvalue"],
            alpha=0.5,
            linestyle="-",
            label="AEC (Average Eigenvalue)",
        )

    def _plot_KP_criterion(self, ax, criteria_colors):
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
            color=criteria_colors["kp"],
            alpha=0.5,
            linestyle="--",
            label="KP Criterion",
        )

    def _plot_KL_criterion(self, ax, criteria_colors):
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
            color=criteria_colors["kl"],
            alpha=0.5,
            linestyle="-",
            label="KL Criterion",
        )

    def _plot_CAEC_criterion(self, ax, criteria_colors):
        """Marks the CAEC (Cumulative Average Eigenvalue Criterion) on the plot."""
        ax.axvline(
            x=np.argmax(
                self.pca_object.V_ordered < 0.7 * self.pca_object.V_ordered.mean()
            )
            - 0.5,
            color=criteria_colors["caec"],
            alpha=0.5,
            linestyle="--",
            label="CAEC (70% of Avg. Eigenvalue)",
        )

    def _plot_broken_stick(self, ax, criteria_colors):
        """Plots the broken stick criterion for selecting the number of PCs."""
        n = self.pca_object.V_ordered.shape[0]
        dm = (100 / n) * np.cumsum(1 / np.arange(1, n + 1)[::-1])
        ax.plot(
            self.pca_object.PC_index,
            dm,
            color=criteria_colors["broken_stick"],
            label="Broken Stick",
        )

    def plot_hotteling_t2_vs_q(self, **kwargs):
        """Plots the Hotelling's T2 statistic versus the Q statistic (Squared Prediction Error).

        Args:
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=params["figsize"])

        for i in range(len(self.pca_object.Q)):
            ax.plot(
                self.pca_object.Q[i],
                self.pca_object.T2[i],
                "o",
                label=self.pca_object.objects[i],
                color=self.colors["theme_color"],
            )

        self._set_labels(
            ax,
            xlabel=r"$Q$ (Squared Prediction Error)",
            ylabel=r"$Hotelling's T^2$",
            subplot_title=params.get("subplot_title", "Hotelling's T2 vs. Q"),
        )
        ax.grid(False)

        self._apply_common_layout(fig, params)
        return fig

    def plot_pci_contribution(self, **kwargs):
        """Plots the contribution of each variable to each principal component."""
        params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=params["figsize"])
        for i in range(self.pca_object.W.shape[1]):
            ax.plot(
                np.arange(self.pca_object.n_variables),
                self.pca_object.W[:, i],
                marker="o",
                markerfacecolor="none",
                label=f"PC$_{i+1}$",
                color=self.colors["theme_color"],
            )

        self._set_labels(
            ax,
            xlabel="Variable",
            ylabel="Value of Loading",
            subplot_title=params.get("subplot_title", "Contributions of Variables to Each PC"),
        )

        plt.xticks(
            np.arange(self.pca_object.n_variables),
            self.pca_object.variables,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.grid(False)
        self._apply_common_layout(fig, params)
        return fig

    def plot_loadings(self, components=None, show_arrows=True, **kwargs):
        """Plot loadings with themed colors."""
        params = self._process_common_params(**kwargs)
        if components is not None:
            i, j = components
            fig = self._plot_loadings_single(i, j, show_arrows, params)
        else:
            fig = self._plot_loadings_matrix(show_arrows, params)
        
        self._apply_common_layout(fig, params)
        return fig

    def _plot_loadings_single(self, i, j, show_arrows, params):
        """Plots loadings for a single pair of components."""
        fig, ax = self._create_figure(figsize=params["figsize"])
        self._set_labels(ax, subplot_title=params.get("subplot_title", f"Loadings plot PC{i+1} vs PC{j+1}"))

        x_data = self.pca_object.W[:, i]
        y_data = self.pca_object.W[:, j]
        scatter_colors = self._normalize_colors(self.pca_object.variables_colors)
        scatter = ax.scatter(
            x_data, y_data, c=scatter_colors, label=self.pca_object.variables
        )

        if show_arrows:
            for d in range(self.pca_object.n_variables):
                ax.arrow(
                    0,
                    0,
                    self.pca_object.W[d, i],
                    self.pca_object.W[d, j],
                    length_includes_head=True,
                    width=0.01,
                    color=self.colors["theme_color"],
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
                color=self.colors["theme_color"],
            )

        self._set_labels(ax, xlabel=rf"PC$_{i+1}$", ylabel=rf"PC$_{j+1}$")
        return fig

    def _plot_loadings_matrix(self, show_arrows, params):
        fig, axs = plt.subplots(
            self.pca_object.n_component,
            self.pca_object.n_component,
            figsize=params["figsize"],
        )
        self._set_labels(fig, title=params.get("title", "Loadings Plot"))

        if self.pca_object.n_component == 1:
            axs = np.array([[axs]])
        elif self.pca_object.n_component == 2:
            axs = np.array([[axs[0], axs[1]], [axs[0], axs[1]]])

        for i in range(self.pca_object.n_component):
            for j in range(self.pca_object.n_component):
                ax = axs[i, j]
                if i != j:
                    self._plot_loadings_on_axis(ax, i, j, show_arrows)
                else:
                    self._plot_empty_loadings_on_axis(ax, i)
                    
        return fig

    def _plot_loadings_on_axis(self, ax, i, j, show_arrows=True):
        """Plots loadings for a specific pair of components on a given axis."""
        x_data = self.pca_object.W[:, i]
        y_data = self.pca_object.W[:, j]
        scatter_colors = self._normalize_colors(self.pca_object.variables_colors)
        ax.scatter(x_data, y_data, c=scatter_colors, label=self.pca_object.variables)

        if show_arrows:
            for d in range(self.pca_object.n_variables):
                ax.arrow(
                    0,
                    0,
                    self.pca_object.W[d, i],
                    self.pca_object.W[d, j],
                    length_includes_head=True,
                    width=0.01,
                    color=self.colors["theme_color"],
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
                color=self.colors["theme_color"],
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

    def plot_scores(self, components=None, label_points=False, **kwargs):
        """Plot scores with proper theme colors."""
        params = self._process_common_params(**kwargs)
        if components is not None:
            i, j = components
            fig = self._plot_scores_single(i, j, label_points, params)
        else:
            fig = self._plot_scores_matrix(label_points, params)
        
        self._apply_common_layout(fig, params)
        return fig

    def _plot_scores_single(self, i, j, label_points, params):
        """Plots scores for a single pair of components."""
        fig, ax = self._create_figure(figsize=params["figsize"])
        self._set_labels(ax, subplot_title=params.get("subplot_title", f"Scores plot PC{i+1} vs PC{j+1}"))

        x_data = self.pca_object.T[:, i]
        y_data = self.pca_object.T[:, j]
        scatter_colors = self._normalize_colors(self.pca_object.objects_colors)
        scatter = ax.scatter(
            x_data, y_data, c=scatter_colors, label=self.pca_object.objects
        )

        if label_points:
            for d in range(self.pca_object.n_objects):
                ax.annotate(
                    text=self.pca_object.objects[d][0],
                    xy=(self.pca_object.T[d, i], self.pca_object.T[d, j]),
                    color=self.colors["theme_color"],
                )

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")
        return fig

    def _plot_scores_matrix(self, label_points, params):
        fig, axs = plt.subplots(
            self.pca_object.n_component,
            self.pca_object.n_component,
            figsize=params["figsize"],
        )
        self._set_labels(fig, title=params.get("title", "Scores Plot"))

        if self.pca_object.n_component == 1:
            axs = np.array([[axs]])
        elif self.pca_object.n_component == 2:
            axs = np.array([[axs[0], axs[1]], [axs[0], axs[1]]])

        for i in range(self.pca_object.n_component):
            for j in range(self.pca_object.n_component):
                ax = axs[i, j]
                if i != j:
                    self._plot_scores_on_axis(ax, i, j, label_points)
                else:
                    self._plot_empty_scores_on_axis(ax, i)
                    
        return fig

    def _plot_scores_on_axis(self, ax, i, j, label_points=False):
        """Plots scores for a specific pair of components on a given axis."""
        x_data = self.pca_object.T[:, i]
        y_data = self.pca_object.T[:, j]
        scatter_colors = self._normalize_colors(self.pca_object.objects_colors)
        ax.scatter(x_data, y_data, c=scatter_colors, label=self.pca_object.objects)

        if label_points:
            for d in range(self.pca_object.n_objects):
                ax.annotate(
                    text=self.pca_object.objects[d][0],
                    xy=(self.pca_object.T[d, i], self.pca_object.T[d, j]),
                    color=self.colors["theme_color"],
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
        **kwargs,
    ):
        """Plots a biplot, combining scores and loadings in one plot.

        Args:
            components (tuple, optional): The principal components to display.
                                        Defaults to None (all components).
            label_points (bool, optional): Whether to label the score points.
                                        Defaults to False.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if components is not None:
            i, j = components
            fig = self._plot_biplot_single(
                i, j, label_points, params
            )
        else:
            fig = self._plot_biplot_matrix(
                label_points, params
            )
        
        self._apply_common_layout(fig, params)
        return fig

    def _plot_biplot_single(
        self, i, j, label_points, params
    ):
        """Plots a biplot for a single pair of components."""
        fig, ax = self._create_figure(figsize=params["figsize"])
        self._set_labels(ax, subplot_title=params.get("subplot_title", f"Biplot PC{i+1} vs PC{j+1}"))

        x_data = self.pca_object.T[:, i]
        y_data = self.pca_object.T[:, j]
        scatter_colors = self._normalize_colors(self.pca_object.objects_colors)
        scatter = ax.scatter(
            x_data, y_data, c=scatter_colors, label=self.pca_object.objects
        )

        if label_points:
            for e in range(self.pca_object.n_objects):
                ax.annotate(
                    text=self.pca_object.objects[e],
                    xy=(self.pca_object.T[e, i], self.pca_object.T[e, j]),
                    color=self.colors["theme_color"],
                )

        for d in range(self.pca_object.n_variables):
            ax.arrow(
                0,
                0,
                self.pca_object.W[d, i],
                self.pca_object.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=self.colors["theme_color"],
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
                color=self.colors["theme_color"],
            )

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")
        return fig

    def _plot_biplot_matrix(
        self, label_points, params
    ):
        """Plots biplots for all pairs of components in a matrix."""
        height_ratios = [1] * self.pca_object.n_component
        width_ratios = [1] * self.pca_object.n_component

        fig, axs = plt.subplots(
            self.pca_object.n_component,
            self.pca_object.n_component,
            figsize=params["figsize"],
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        self._set_labels(fig, title=params.get("title", "Biplots plot"))

        all_handles = []
        all_labels = []

        for i in range(self.pca_object.n_component):
            for j in range(self.pca_object.n_component):
                ax = axs[i, j] if self.pca_object.n_component > 1 else axs
                if i != j:
                    (
                        handles,
                        labels,
                    ) = self._plot_biplot_on_axis(ax, i, j, label_points)
                    all_handles.extend(handles)
                    all_labels.extend(labels)
                else:
                    self._plot_empty_biplot_on_axis(ax, i)
        return fig

    def _plot_biplot_on_axis(self, ax, i, j, label_points=False):
        """Plots a biplot for a specific pair of components on a given axis."""
        x_data = self.pca_object.T[:, i]
        y_data = self.pca_object.T[:, j]
        scatter_colors = self._normalize_colors(self.pca_object.objects_colors)
        scatter = ax.scatter(
            x_data, y_data, c=scatter_colors, label=self.pca_object.objects
        )

        if label_points:
            for e in range(self.pca_object.n_objects):
                ax.annotate(
                    text=self.pca_object.objects[e],
                    xy=(self.pca_object.T[e, i], self.pca_object.T[e, j]),
                    color=self.colors["theme_color"],
                )

        for d in range(self.pca_object.n_variables):
            ax.arrow(
                0,
                0,
                self.pca_object.W[d, i],
                self.pca_object.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=self.colors["theme_color"],
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
                color=self.colors["theme_color"],
            )

        self._set_labels(ax, xlabel=f"PC{i+1}", ylabel=f"PC{j+1}")
        handles, labels = scatter.legend_elements()
        return handles, labels

    def _plot_empty_biplot_on_axis(self, ax, i):
        """Plots an empty space for diagonal cells in the biplot matrix."""
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

    def plot_classes_pca(
        self, class_labels, new_data_point=None, colors=None, ellipse_std=2, **kwargs
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
        params = self._process_common_params(**kwargs)
        pca_models = self.pca_object
        if colors is None:
            if len(pca_models) <= len(self.colors["category_color_scale"]):
                colors = self.colors["category_color_scale"][:len(pca_models)]
            else:
                # Fallback to a matplotlib colormap if not enough theme colors
                colors = plt.cm.get_cmap("viridis", len(pca_models)).colors

        fig, ax = self._create_figure(figsize=params["figsize"])

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
                color=self.colors["theme_color"],
                markeredgecolor=self.colors["bg_color"],
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
            ax, xlabel="PC1", ylabel="PC2", subplot_title=params.get("subplot_title", "Class Separation in PCA Space")
        )
        self._apply_common_layout(fig, params)
        return fig