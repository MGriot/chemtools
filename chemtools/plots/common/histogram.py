import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from chemtools.plots.Plotter import Plotter

class HistogramPlotter(Plotter):
    """
    A plotter for creating histograms.
    """

    def plot(self, data: pd.DataFrame, column: str, **kwargs):
        """
        Plots a histogram for a single numerical variable.

        Args:
            data (pd.DataFrame): The dataset to use.
            column (str): The name of the column to plot.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"Column '{column}' must be numeric to plot a histogram.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(**kwargs)
            ax.hist(data[column], bins=kwargs.get("bins", 10), color=self.colors['theme_color'], edgecolor=self.colors['text_color'])
            self._set_labels(ax, title=f'Histogram of {column}', xlabel=column, ylabel='Frequency')
            return self.apply_style_preset(fig)
        elif self.library == "plotly":
            fig = px.histogram(data, x=column, title=f'Histogram of {column}', color_discrete_sequence=[self.colors['theme_color']], **kwargs)
            self._set_labels(fig, title=f'Histogram of {column}', xlabel=column, ylabel='Frequency')
            return self.apply_style_preset(fig)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")