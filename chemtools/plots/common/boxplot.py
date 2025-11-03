import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from chemtools.plots.Plotter import Plotter

class BoxPlotter(Plotter):
    """
    A plotter for creating box plots.
    """

    def plot(self, data: pd.DataFrame, column: str, **kwargs):
        """
        Plots a box plot for a single numerical variable.

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
            raise TypeError(f"Column '{column}' must be numeric to plot a box plot.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(**kwargs)
            ax.boxplot(data[column])
            self._set_labels(ax, title=f'Box Plot of {column}', ylabel=column)
            return self.apply_style_preset(fig)
        elif self.library == "plotly":
            fig = px.box(data, y=column, title=f'Box Plot of {column}', color_discrete_sequence=[self.colors['theme_color']], **kwargs)
            self._set_labels(fig, title=f'Box Plot of {column}', ylabel=column)
            return self.apply_style_preset(fig)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")