import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmaps
from scipy.spatial.distance import pdist, squareform
from typing import List, Optional, Tuple


class DistanceMatrix:
    """
    Calculates, stores, and visualizes pairwise distance matrices from data.
    """

    def __init__(self, data: np.ndarray, metric: str = "euclidean"):
        """
        Initializes the processor and computes the distance matrices.

        Args:
            data (np.ndarray): The input data (n_samples, n_features).
            metric (str): The distance metric accepted by scipy.spatial.distance.pdist.
                          Defaults to 'euclidean'.
        """
        self.data = np.array(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError("Data must be a 2D array (n_samples, n_features).")
        if self.data.shape[0] < 2:
            raise ValueError(
                "Data must contain at least two samples to compute distances."
            )

        self.n_samples = self.data.shape[0]
        self.metric = metric

        print(f"Calculating {self.metric} distances for {self.n_samples} samples...")
        try:
            # Calculate condensed distance matrix (vector form)
            self.condensed_distance_matrix: np.ndarray = pdist(
                self.data, metric=self.metric
            )
            # Calculate full square distance matrix
            self.distance_matrix: np.ndarray = squareform(
                self.condensed_distance_matrix
            )
            print("Distance calculation complete.")
        except Exception as e:
            raise RuntimeError(
                f"Error computing distance matrix with metric '{self.metric}': {e}"
            ) from e

    def get_distance_matrix(self) -> np.ndarray:
        """Returns the full square distance matrix."""
        return self.distance_matrix

    def get_condensed_matrix(self) -> np.ndarray:
        """Returns the condensed distance matrix (pdist format)."""
        return self.condensed_distance_matrix

    def plot_heatmap(
        self,
        ax: Optional[plt.Axes] = None,
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        show_values: bool = False,
        value_format: str = ".2f",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the distance matrix as a heatmap.

        Args:
            ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates new figure/axes.
            labels (Optional[List[str]]): Labels for the matrix axes. If None, uses indices.
            title (Optional[str]): Title for the heatmap. Defaults to 'Distance Matrix Heatmap'.
            cmap (str): Colormap for the heatmap. Defaults to 'viridis'.
            show_values (bool): Whether to display the distance values on the heatmap cells.
                                Defaults to False (can be cluttered for large matrices).
            value_format (str): Format string for displaying values if show_values is True.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.figure

        effective_labels: Union[bool, List[str]] = (
            False  # Default for seaborn: no labels
        )
        if labels is not None:
            if len(labels) == self.n_samples:
                effective_labels = labels
            else:
                print(
                    f"Warning: Number of labels ({len(labels)}) does not match number of samples "
                    f"({self.n_samples}). Ignoring labels for heatmap."
                )

        sns.heatmap(
            self.distance_matrix,
            ax=ax,
            cmap=cmap,
            annot=show_values,  # Show values?
            fmt=value_format,  # Format for values
            square=True,  # Ensure cells are square
            xticklabels=effective_labels,
            yticklabels=effective_labels,
            linewidths=0.5,  # Add lines between cells
            cbar_kws={"shrink": 0.8},  # Adjust color bar size
        )

        if title is None:
            title = f"{self.metric.capitalize()} Distance Matrix Heatmap ({self.n_samples}x{self.n_samples})"
        ax.set_title(title, fontsize=14, weight="bold")

        # Rotate x-axis labels if they are provided and long
        if effective_labels and isinstance(effective_labels, list):
            plt.setp(
                ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor"
            )
            plt.setp(ax.get_yticklabels(), rotation=0)

        fig.tight_layout()
        return fig, ax
