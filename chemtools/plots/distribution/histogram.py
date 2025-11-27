import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from ..base import BasePlotter

class HistogramPlot(BasePlotter):
    """
    A plotter for creating histograms and density curves.
    """

    def plot(self, data: pd.DataFrame, column: str, mode: str = 'hist', **kwargs):
        """
        Plots a histogram or a density curve for a single numerical variable.

        Args:
            data (pd.DataFrame): The dataset to use.
            column (str): The name of the column to plot.
            mode (str, optional): 'hist' for histogram, 'density' for density curve. Defaults to 'hist'.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        params = self._process_common_params(**kwargs)
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"Column '{column}' must be numeric to plot a histogram.")
        
        clean_data = data[column].dropna()

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            if mode == 'density':
                density = gaussian_kde(clean_data)
                xs = np.linspace(clean_data.min(), clean_data.max(), 200)
                ax.plot(xs, density(xs), color=self.colors['theme_color'])
                ax.fill_between(xs, 0, density(xs), alpha=0.4, color=self.colors['theme_color'])
                self._set_labels(ax, subplot_title=params.get("subplot_title", f'Density of {column}'), xlabel=column, ylabel='Density')
            else: # 'hist'
                ax.hist(clean_data, bins=kwargs.get("bins", 10), color=self.colors['theme_color'], edgecolor=self.colors['text_color'])
                self._set_labels(ax, subplot_title=params.get("subplot_title", f'Histogram of {column}'), xlabel=column, ylabel='Frequency')
            
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            if mode == 'density':
                title = params.get("title", f'Density of {column}')
                density = gaussian_kde(clean_data)
                xs = np.linspace(clean_data.min(), clean_data.max(), 200)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=density(xs), fill='tozeroy', mode='lines', line_color=self.colors['theme_color']))
                self._set_labels(fig, title=title, xlabel=column, ylabel='Density')
            else: # 'hist'
                title = params.get("title", f'Histogram of {column}')
                fig = px.histogram(data, x=column, title=title, color_discrete_sequence=[self.colors['theme_color']], **kwargs) # px.histogram handles NaNs
                self._set_labels(fig, xlabel=column, ylabel='Frequency')

            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")