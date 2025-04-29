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
            Additional keyword arguments:
            - figsize : tuple, optional
                Figure size (width, height)
            - title : str, optional
                Plot title
            - orientation : str, optional
                'top', 'right', 'bottom', or 'left'
            - labels : list, optional
                Leaf labels
            
        Returns:
        --------
        fig : Figure object
            The generated figure object (matplotlib.Figure or plotly.graph_objects.Figure)
        """
        if not hasattr(model, 'children_') or not hasattr(model, 'distances_'):
            raise ValueError("Model doesn't appear to be a fitted HierarchicalClustering instance.")

        n_samples = len(getattr(model, 'labels_', []))
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
        linkage_matrix = np.column_stack([
            model.children_, model.distances_, counts
        ]).astype(float)

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=kwargs.get('figsize', (10, 7)))
            dendrogram(
                linkage_matrix,
                ax=ax,
                orientation=kwargs.get('orientation', 'top'),
                labels=kwargs.get('labels', None),
                color_threshold=kwargs.get('color_threshold', None)
            )
            if 'title' in kwargs:
                ax.set_title(kwargs['title'])
            return fig

        elif self.library == "plotly":
            fig = ff.create_dendrogram(
                linkage_matrix,
                orientation=kwargs.get('orientation', 'top'),
                labels=kwargs.get('labels', None)
            )
            if 'title' in kwargs:
                fig.update_layout(title=kwargs['title'])
            fig.update_layout(template="chemtools")
            return fig
