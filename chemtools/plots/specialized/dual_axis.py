import pandas as pd
import matplotlib.pyplot as plt
from ..base import BasePlotter

class DualAxisPlot(BasePlotter):
    """
    A plotter for creating dual-axis charts.
    Note: This is currently only supported for the matplotlib library.
    """

    def plot(self, data: pd.DataFrame, x_column: str, y1_column: str, y2_column: str,
             plot1_kind: str = 'bar', plot2_kind: str = 'line', **kwargs):
        """
        Plots a dual-axis chart with two different y-axes.

        Args:
            data (pd.DataFrame): The dataset to use.
            x_column (str): The column for the shared x-axis.
            y1_column (str): The column for the first y-axis.
            y2_column (str): The column for the second y-axis.
            plot1_kind (str, optional): The kind of plot for the first y-axis ('bar' or 'line'). Defaults to 'bar'.
            plot2_kind (str, optional): The kind of plot for the second y-axis ('bar' or 'line'). Defaults to 'line'.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if self.library != "matplotlib":
            raise NotImplementedError("Dual-axis charts are currently only supported for the matplotlib library.")

        fig, ax1 = self._create_figure(figsize=params["figsize"])
        ax2 = ax1.twinx()

        # Plot 1
        if plot1_kind == 'bar':
            ax1.bar(data[x_column], data[y1_column], color=self.colors['theme_color'], alpha=0.7, label=y1_column)
        else: # 'line'
            ax1.plot(data[x_column], data[y1_column], color=self.colors['theme_color'], label=y1_column)
        
        ax1.set_xlabel(x_column)
        ax1.set_ylabel(y1_column, color=self.colors['theme_color'])
        ax1.tick_params(axis='y', labelcolor=self.colors['theme_color'])

        # Plot 2
        if plot2_kind == 'bar':
            ax2.bar(data[x_column], data[y2_column], color=self.colors['accent_color'], alpha=0.7, label=y2_column)
        else: # 'line'
            ax2.plot(data[x_column], data[y2_column], color=self.colors['accent_color'], label=y2_column)

        ax2.set_ylabel(y2_column, color=self.colors['accent_color'])
        ax2.tick_params(axis='y', labelcolor=self.colors['accent_color'])

        self._set_labels(ax1, subplot_title=params.get('subplot_title', f'{y1_column} and {y2_column} vs. {x_column}'))
        
        # Add legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if params.get("showlegend", True):
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        self._apply_common_layout(fig, params)
        return fig