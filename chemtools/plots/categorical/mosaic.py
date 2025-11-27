import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import plotly.express as px
from ..base import BasePlotter

class MosaicPlot(BasePlotter):
    """
    A plotter for creating mosaic plots to visualize relationships between categorical variables.
    """

    def plot(self, data: pd.DataFrame, columns: list, **kwargs):
        """
        Creates a mosaic plot for two or more categorical variables.

        Args:
            data (pd.DataFrame): The dataset to use.
            columns (list): A list of 2 or more column names to include in the plot.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)

        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in the data.")

        if self.library == "matplotlib":
            if len(columns) < 2:
                raise ValueError("Matplotlib mosaic plots require at least two categorical variables.")
            
            plot_data = data.groupby(columns).size()
            
            fig, ax = self._create_figure(figsize=params["figsize"])
            mosaic(plot_data, ax=ax, title=params.get("subplot_title", "Mosaic Plot"))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            title = params.get("title", "Mosaic Plot (Treemap)")
            
            plot_data = data.groupby(columns).size().reset_index(name='count')

            fig = px.treemap(plot_data, path=columns, values='count', title=title,
                             color=columns[0], color_discrete_sequence=self.colors['category_color_scale'])
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")