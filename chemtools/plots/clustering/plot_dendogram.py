import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from ..Plotter import Plotter
import plotly.figure_factory as ff


class DendrogramPlotter(Plotter):
    """
    A plotter class for dendrogram visualization that supports both matplotlib and plotly.
    Inherits from the master Plotter class.
    """

    def plot_dendrogram(self, model, **kwargs):
        """
        Plot a dendrogram from a fitted HierarchicalClustering model.

        Parameters:
        -----------
        model : HierarchicalClustering
            Fitted hierarchical clustering model.
        **kwargs : dict
            Common parameters for both backends:
            - figsize : tuple, optional (default=(10, 7))
                Figure size as (width, height)
            - title : str, optional
                Plot title
            - orientation : str, optional (default='top')
                'top', 'right', 'bottom', or 'left'
            - labels : list, optional
                Leaf labels
            - color_threshold : float, optional
                Color threshold for clusters
            - height : int, optional (default=600)
                Figure height in pixels
            - width : int, optional (default=800)
                Figure width in pixels
        """
        if not hasattr(model, "children_") or not hasattr(model, "distances_"):
            raise ValueError(
                "Model doesn't appear to be a fitted HierarchicalClustering instance."
            )

        n_samples = len(getattr(model, "labels_", []))
        if n_samples == 0:
            raise ValueError("Model doesn't have any labels.")

        # Calculate counts
        counts = np.zeros(model.children_.shape[0])
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        # Create linkage matrix
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Process common parameters
        params = self._process_common_params(**kwargs)

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params['figsize'])
            dendrogram(
                linkage_matrix,
                ax=ax,
                orientation=params['orientation'],
                labels=params['labels'],
                color_threshold=params['color_threshold']
            )
            fig = self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            fig = ff.create_dendrogram(
                linkage_matrix,
                orientation=params['orientation'],
                labels=params['labels'],
                color_threshold=params['color_threshold']
            )
            fig = self._apply_common_layout(fig, params)
            return fig
