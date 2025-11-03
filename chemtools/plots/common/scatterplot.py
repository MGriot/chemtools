import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from chemtools.plots.Plotter import Plotter

class ScatterPlotter(Plotter):
    """
    A plotter for creating scatter plots.
    """

    def plot_2d(self, data: pd.DataFrame, x_column: str, y_column: str, **kwargs):
        """
        Creates a 2D scatter plot of two numerical variables.

        Args:
            data (pd.DataFrame): The dataset to use.
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError("One or both columns not found in the data.")
        if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]):
            raise TypeError("Both columns must be numeric for a scatter plot.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(**kwargs)
            ax.scatter(data[x_column], data[y_column], color=self.colors['accent_color'])
            self._set_labels(ax, title=f'{y_column} vs. {x_column}', xlabel=x_column, ylabel=y_column)
            return self.apply_style_preset(fig)
        elif self.library == "plotly":
            fig = px.scatter(data, x=x_column, y=y_column, title=f'{y_column} vs. {x_column}', color_discrete_sequence=[self.colors['accent_color']], **kwargs)
            self._set_labels(fig, title=f'{y_column} vs. {x_column}', xlabel=x_column, ylabel=y_column)
            return self.apply_style_preset(fig)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")

    def plot_3d(self, data: pd.DataFrame, x_column: str, y_column: str, z_column: str, **kwargs):
        """
        Creates a 3D scatter plot of three numerical variables.

        Args:
            data (pd.DataFrame): The dataset to use.
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
            z_column (str): The column for the z-axis.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib figure object.
        """
        if x_column not in data.columns or y_column not in data.columns or z_column not in data.columns:
            raise ValueError("One or more columns not found in the data.")
        if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]) or not pd.api.types.is_numeric_dtype(data[z_column]):
            raise TypeError("All three columns must be numeric for a 3D scatter plot.")

        if self.library == "matplotlib":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(data[x_column], data[y_column], data[z_column], c=self.colors['accent_color'])
            
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_zlabel(z_column)
            ax.set_title(f'3D Scatter Plot: {x_column}, {y_column}, {z_column}')
            
            return fig
        elif self.library == "plotly":
            fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, title=f'3D Scatter Plot: {x_column}, {y_column}, {z_column}', color_discrete_sequence=[self.colors['accent_color']], **kwargs)
            self._set_labels(fig, title=f'3D Scatter Plot: {x_column}, {y_column}, {z_column}', xlabel=x_column, ylabel=y_column)
            return self.apply_style_preset(fig)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")