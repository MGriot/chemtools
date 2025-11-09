import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter

class ParallelCoordinatesPlot(BasePlotter):
    """
    A plotter for creating parallel coordinates plots.
    """

    def plot(self, data: pd.DataFrame, class_column: str, **kwargs):
        """
        Creates a parallel coordinates plot.

        Args:
            data (pd.DataFrame): The dataset to use.
            class_column (str): The column to color the lines by.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        params = self._process_common_params(**kwargs)
        if class_column not in data.columns:
            raise ValueError(f"Column '{class_column}' not found in the data.")

        # Select only numerical columns and the class column for the plot
        numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
        plot_data = data[numerical_cols + [class_column]]

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            
            # Don't pass plotter-specific kwargs to pandas plotting
            plot_kwargs = kwargs.copy()
            invalid_pd_kwargs = ['title', 'subplot_title', 'xlabel', 'ylabel', 'labels', 'height', 'width', 'showlegend']
            for k in invalid_pd_kwargs:
                plot_kwargs.pop(k, None)

            pd.plotting.parallel_coordinates(plot_data, class_column, ax=ax, color=self.colors['category_color_scale'], **plot_kwargs)
            
            self._set_labels(ax, subplot_title=params.get('subplot_title', 'Parallel Coordinates Plot'))
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            title = params.get('title', 'Parallel Coordinates Plot')
            fig = px.parallel_coordinates(plot_data, color=class_column, title=title, color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            self._set_labels(fig, title=title)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")