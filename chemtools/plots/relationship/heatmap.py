import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..base import BasePlotter

class HeatmapPlot(BasePlotter):
    """
    A plotter for creating heatmaps.
    """

    def plot(self, data: pd.DataFrame, **kwargs):
        """
        Plots a heatmap of a given matrix.

        Args:
            data (pd.DataFrame): The matrix to plot.
            **kwargs: Additional keyword arguments passed to the plotter.
                      Can include 'title' for the figure title.

        Returns:
            A matplotlib or plotly figure object.
        """
        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            im = ax.imshow(data, cmap=kwargs.get("cmap", 'coolwarm'), aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(data.columns)))
            ax.set_yticks(range(len(data.index)))
            ax.set_xticklabels(data.columns, rotation=90)
            ax.set_yticklabels(data.index)
            ax.grid(False) # Disable grid for heatmap
            self._set_labels(ax, subplot_title=params.get('subplot_title', 'Heatmap'))
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            title = params.get('title', 'Heatmap')
            fig = go.Figure(data=go.Heatmap(z=data.values, x=data.columns, y=data.index, colorscale=kwargs.get("colorscale", 'RdBu')))
            self._set_labels(fig, title=title)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")