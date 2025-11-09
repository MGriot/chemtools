import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter

class FunnelPlot(BasePlotter):
    """
    A plotter for creating funnel charts.
    """

    def plot(self, data: pd.DataFrame, stage_column: str, values_column: str, **kwargs):
        """
        Plots a funnel chart.

        Args:
            data (pd.DataFrame): The dataset to use. It should be sorted in the order of the funnel stages.
            stage_column (str): The column with the names of the funnel stages.
            values_column (str): The column with the values for each stage.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if stage_column not in data.columns or values_column not in data.columns:
            raise ValueError("Stage or values column not found in the data.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            
            # Manual funnel chart with fill_between
            y = range(len(data))
            x1 = [(1 - v / data[values_column].max()) / 2 for v in data[values_column]]
            x2 = [1 - x for x in x1]

            ax.fill_betweenx(y, x1, x2, color=self.colors['theme_color'], alpha=0.6)
            
            for i, val in enumerate(data[values_column]):
                ax.text(0.5, y[i], f"{data[stage_column].iloc[i]}: {val}", ha='center', va='center', color=self.colors['text_color'])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            self._set_labels(ax, subplot_title=params.get('subplot_title', 'Funnel Chart'))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            title = params.get('title', 'Funnel Chart')
            
            # Make a copy of kwargs to modify
            plotly_kwargs = kwargs.copy()
            
            # 'title' is passed explicitly, so remove it from plotly_kwargs to avoid conflict
            plotly_kwargs.pop('title', None)

            fig = px.funnel(data, x=values_column, y=stage_column, title=title,
                            color_discrete_sequence=[self.colors['theme_color']], **plotly_kwargs)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")