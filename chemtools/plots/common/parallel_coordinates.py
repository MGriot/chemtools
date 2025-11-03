import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from chemtools.plots.Plotter import Plotter

class ParallelCoordinatesPlotter(Plotter):
    """
    A plotter for creating parallel coordinates plots.
    """

    def plot(self, data: pd.DataFrame, class_column: str, **kwargs):
        """
        Creates a parallel coordinates plot.

        Args:
            data (pd.DataFrame): The dataset to use.
            class_column (str): The column to color the lines by.
            **kwargs: Additional keyword arguments passed to pandas.plotting.parallel_coordinates.

        Returns:
            A matplotlib figure object.
        """
        if class_column not in data.columns:
            raise ValueError(f"Column '{class_column}' not found in the data.")

        # Select only numerical columns and the class column for the plot
        numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
        plot_data = data[numerical_cols + [class_column]]

        if self.library == "matplotlib":
            fig, ax = self._create_figure()
            pd.plotting.parallel_coordinates(plot_data, class_column, ax=ax, color=self.colors['category_color_scale'], **kwargs)
            self._set_labels(ax, title='Parallel Coordinates Plot')
            return self.apply_style_preset(fig)
        elif self.library == "plotly":
            fig = px.parallel_coordinates(plot_data, color=class_column, color_continuous_scale=self.colors['category_color_scale'], title='Parallel Coordinates Plot', **kwargs)
            self._set_labels(fig, title='Parallel Coordinates Plot')
            return self.apply_style_preset(fig)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")