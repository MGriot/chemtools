import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter

class LinePlot(BasePlotter):
    """
    A plotter for creating line, dot, and area plots.
    """

    def plot(self, data: pd.DataFrame, x_column: str, y_column: str, mode: str = 'line', **kwargs):
        """
        Plots a line, dot, or area plot.

        Args:
            data (pd.DataFrame): The dataset to use.
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
            mode (str, optional): 'line', 'dot', or 'area'. Defaults to 'line'.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        params = self._process_common_params(**kwargs)
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError("One or both columns not found in the data.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            if mode == 'dot':
                ax.plot(data[x_column], data[y_column], linestyle='none', marker='o', color=self.colors['theme_color'])
            elif mode == 'area':
                ax.fill_between(data[x_column], data[y_column], color=self.colors['theme_color'], alpha=0.4)
                ax.plot(data[x_column], data[y_column], color=self.colors['theme_color']) # Add line on top of area
            else: # 'line'
                ax.plot(data[x_column], data[y_column], color=self.colors['theme_color'])
            self._set_labels(ax, subplot_title=params.get("subplot_title", f'{y_column} vs. {x_column}'), xlabel=x_column, ylabel=y_column)
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            title = params.get("title", f'{y_column} vs. {x_column}')
            if mode == 'dot':
                fig = px.scatter(data, x=x_column, y=y_column, title=title, color_discrete_sequence=[self.colors['theme_color']], **kwargs)
            elif mode == 'area':
                fig = px.area(data, x=x_column, y=y_column, title=title, color_discrete_sequence=[self.colors['theme_color']], **kwargs)
            else: # 'line'
                fig = px.line(data, x=x_column, y=y_column, title=title, color_discrete_sequence=[self.colors['theme_color']], **kwargs)
            self._set_labels(fig, xlabel=x_column, ylabel=y_column)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")