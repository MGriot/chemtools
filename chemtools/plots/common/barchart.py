import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from chemtools.plots.Plotter import Plotter

class BarChartPlotter(Plotter):
    """
    A plotter for creating bar charts.
    """

    def plot(self, data: pd.DataFrame, column: str, **kwargs):
        """
        Plots a bar chart for a single categorical variable.

        Args:
            data (pd.DataFrame): The dataset to use.
            column (str): The name of the column to plot.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"Column '{column}' must be categorical to plot a bar chart.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(**kwargs)
            counts = data[column].value_counts()
            ax.bar(counts.index, counts.values, color=self.colors['category_color_scale'][:len(counts)])
            self._set_labels(ax, title=f'Bar Chart of {column}', xlabel=column, ylabel='Count')
            return self.apply_style_preset(fig)
        elif self.library == "plotly":
            fig = px.bar(data, x=column, title=f'Bar Chart of {column}', color=column, color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            self._set_labels(fig, title=f'Bar Chart of {column}', xlabel=column, ylabel='Count')
            return self.apply_style_preset(fig)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")