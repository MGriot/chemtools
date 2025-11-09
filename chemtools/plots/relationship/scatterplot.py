import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter

class ScatterPlot(BasePlotter):
    """
    A plotter for creating 2D, 3D, and bubble scatter plots.
    """

    def plot_2d(self, data: pd.DataFrame, x_column: str, y_column: str, size_column: str = None, **kwargs):
        """
        Creates a 2D scatter plot or a bubble chart.

        Args:
            data (pd.DataFrame): The dataset to use.
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
            size_column (str, optional): The column for bubble size. Defaults to None.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError("One or both columns not found in the data.")
        if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]):
            raise TypeError("Both x and y columns must be numeric for a scatter plot.")
        if size_column and size_column not in data.columns:
            raise ValueError(f"Size column '{size_column}' not found in the data.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            size = data[size_column] * 100 if size_column else None # Scale size for visibility
            ax.scatter(data[x_column], data[y_column], s=size, color=self.colors['accent_color'], alpha=0.6)
            self._set_labels(ax, subplot_title=params.get("subplot_title", f'{y_column} vs. {x_column}'), xlabel=x_column, ylabel=y_column)
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            title = params.get("title", f'{y_column} vs. {x_column}')
            fig = px.scatter(data, x=x_column, y=y_column, size=size_column, title=title, color_discrete_sequence=[self.colors['accent_color']], **kwargs)
            self._set_labels(fig, xlabel=x_column, ylabel=y_column)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")

    def plot_3d(self, data: pd.DataFrame, x_column: str, y_column: str, z_column: str, **kwargs):
        """
        Creates a 3D scatter plot of three numerical variables.
        """
        params = self._process_common_params(**kwargs)
        if x_column not in data.columns or y_column not in data.columns or z_column not in data.columns:
            raise ValueError("One or more columns not found in the data.")
        if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]) or not pd.api.types.is_numeric_dtype(data[z_column]):
            raise TypeError("All three columns must be numeric for a 3D scatter plot.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(subplot_kw={'projection': '3d'}, figsize=params["figsize"])
            ax.scatter(data[x_column], data[y_column], data[z_column], c=self.colors['accent_color'])
            ax.set_zlabel(z_column)
            self._set_labels(ax, subplot_title=params.get("subplot_title", f'3D Scatter Plot: {x_column}, {y_column}, {z_column}'), xlabel=x_column, ylabel=y_column)
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            title = params.get("title", f'3D Scatter Plot: {x_column}, {y_column}, {z_column}')
            fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, title=title, color_discrete_sequence=[self.colors['accent_color']], **kwargs)
            # px.scatter_3d handles axis labels from column names.
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")
