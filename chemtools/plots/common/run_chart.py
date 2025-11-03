import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from chemtools.plots.Plotter import Plotter

class RunChartPlotter(Plotter):
    """
    A plotter for creating run charts.
    """

    def plot(self, data: pd.DataFrame, time_column: str, value_column: str, **kwargs):
        """
        Plots a run chart of a variable over time.

        Args:
            data (pd.DataFrame): The dataset to use.
            time_column (str): The column representing time.
            value_column (str): The column representing the value to plot.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if time_column not in data.columns or value_column not in data.columns:
            raise ValueError("One or both columns not found in the data.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(**kwargs)
            ax.plot(data[time_column], data[value_column], color=self.colors['theme_color'])
            self._set_labels(ax, title=f'Run Chart of {value_column} over {time_column}', xlabel=time_column, ylabel=value_column)
            return self.apply_style_preset(fig)
        elif self.library == "plotly":
            fig = px.line(data, x=time_column, y=value_column, title=f'Run Chart of {value_column} over {time_column}', color_discrete_sequence=[self.colors['theme_color']], **kwargs)
            self._set_labels(fig, title=f'Run Chart of {value_column} over {time_column}', xlabel=time_column, ylabel=value_column)
            return self.apply_style_preset(fig)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")