import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from ..base import BasePlotter

class SIMCAPlot(BasePlotter):
    """
    A plotter for visualizing Soft Independent Modeling of Class Analogies (SIMCA) results.
    """

    def __init__(self, simca_model, library="matplotlib", **kwargs):
        super().__init__(library=library, **kwargs)
        if self.library != "matplotlib":
            raise NotImplementedError("SIMCA plots are currently only supported for matplotlib.")
        self.simca_model = simca_model

    def plot_scores(self, components=(0, 1), confidence_level=0.95, show_legend=True, **kwargs):
        """
        Plots the scores of each class model with confidence ellipses.

        Args:
            components (tuple, optional): The principal components to plot (e.g., (0, 1) for PC1 vs PC2). Defaults to (0, 1).
            confidence_level (float, optional): The confidence level for the ellipses (e.g., 0.95 for 95%). Defaults to 0.95.
            show_legend (bool, optional): Whether to display the legend. Defaults to True.
            **kwargs: Additional keyword arguments passed to the plotter. Can include 'title', 'figsize', etc.

        Returns:
            A matplotlib figure object.
        """
        params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=params.get("figsize"))

        idx_i, idx_j = components
        class_labels = sorted(self.simca_model.class_models.keys())
        colors = self.colors['category_color_scale']
        
        legend_handles = []

        for i, class_label in enumerate(class_labels):
            pca_model = self.simca_model.class_models.get(class_label)
            if not pca_model:
                continue

            class_scores = pca_model.T
            if class_scores.shape[1] <= max(idx_i, idx_j):
                print(f"Warning: Class '{class_label}' model has fewer than {max(idx_i, idx_j) + 1} components. Skipping plot.")
                continue

            x_data = class_scores[:, idx_i]
            y_data = class_scores[:, idx_j]
            color = colors[i % len(colors)]

            # Scatter plot for the class scores
            ax.scatter(x_data, y_data, color=color, alpha=0.6, label=f"Scores ({class_label})")
            
            # Centroid
            centroid = np.mean(class_scores, axis=0)
            ax.plot(centroid[idx_i], centroid[idx_j], marker='*', markersize=12, color=color, markeredgecolor=self.colors.get('bg_color', 'white'))

            # Confidence Ellipse
            cov_matrix = np.cov(x_data, y_data)
            if np.all(np.linalg.eigvals(cov_matrix) > 1e-9):
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                angle_deg = np.degrees(angle_rad)

                s = chi2.ppf(confidence_level, df=2)
                width = 2 * np.sqrt(s * eigenvalues[1]) # Eig sorted ascending
                height = 2 * np.sqrt(s * eigenvalues[0])

                ellipse = Ellipse(
                    xy=(centroid[idx_i], centroid[idx_j]),
                    width=width,
                    height=height,
                    angle=angle_deg,
                    edgecolor=color,
                    facecolor='none',
                    linestyle='--',
                    linewidth=2,
                    label=f"{confidence_level*100:.0f}% CI ({class_label})"
                )
                ax.add_patch(ellipse)
            else:
                self.notes.append(f"Warning: Could not draw ellipse for class '{class_label}' due to singular covariance matrix.")
        
        # Collect handles for a clean legend
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            # This can be more sophisticated to avoid duplicate labels, but for now it's okay
            ax.legend(handles=handles, labels=labels, loc='best')

        self._set_labels(ax, 
                         xlabel=f'Principal Component {idx_i + 1}', 
                         ylabel=f'Principal Component {idx_j + 1}', 
                         subplot_title=params.get("subplot_title", 'SIMCA Class Models'))
        self._apply_common_layout(fig, params)
        
        return fig
