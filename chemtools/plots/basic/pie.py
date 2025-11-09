import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter

class PiePlot(BasePlotter):
    """
    A plotter for creating pie and donut charts.
    """

    def plot(self, data: pd.DataFrame, names_column: str, values_column: str, hole: float = 0, **kwargs):
        """
        Plots a pie or donut chart.

        Args:
            data (pd.DataFrame): The dataset to use.
            names_column (str): The column with the names of the slices.
            values_column (str): The column with the values of the slices.
            hole (float, optional): The size of the hole for a donut chart (0 to 1). Defaults to 0 (pie chart).
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if names_column not in data.columns or values_column not in data.columns:
            raise ValueError("Names or values column not found in the data.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            
            # The width of the wedges. 1 for pie, <1 for donut.
            width = 1 - hole 
            
            wedges, texts, autotexts = ax.pie(data[values_column], labels=data[names_column], autopct='%1.1f%%',
                                              colors=self.colors['category_color_scale'],
                                              radius=1, wedgeprops=dict(width=width, edgecolor='w'))

            self._set_labels(ax, subplot_title=params.get('subplot_title', f'{values_column} by {names_column}'))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            title = params.get('title', f'{values_column} by {names_column}')
            fig = px.pie(data, names=names_column, values=values_column, title=title,
                         hole=hole, color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")