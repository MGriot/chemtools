import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from chemtools.plots.Plotter import Plotter

class HeatmapPlotter(Plotter):
    """
    A plotter for creating heatmaps.
    """

    def plot(self, data: pd.DataFrame, **kwargs):
        """
        Plots a heatmap of the correlation matrix.

        Args:
            data (pd.DataFrame): The correlation matrix to plot.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib figure object.
        """
        if self.library == "matplotlib":
            fig, ax = self._create_figure()
            im = ax.imshow(data, cmap=kwargs.get("cmap", 'coolwarm'), aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(data.columns)))
            ax.set_yticks(range(len(data.index)))
            ax.set_xticklabels(data.columns, rotation=90)
            ax.set_yticklabels(data.index)
            self._set_labels(ax, title='Correlation Heatmap')
            return self.apply_style_preset(fig)
        elif self.library == "plotly":
            fig = go.Figure(data=go.Heatmap(z=data.values, x=data.columns, y=data.index, colorscale=kwargs.get("colorscale", 'RdBu')))
            self._set_labels(fig, title='Correlation Heatmap')
            return self.apply_style_preset(fig)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")