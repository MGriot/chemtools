import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..base import BasePlotter

class RadarPlot(BasePlotter):
    """
    A plotter for creating radar charts (also known as spider or star charts).
    """

    def plot(self, data: pd.DataFrame, labels=None, fill=False, tick_label_padding=10, **kwargs):
        """
        Plots a radar chart.

        Args:
            data (pd.DataFrame): The dataset to use. Each row represents a dataset, and each column an axis.
            labels (list, optional): Labels for each axis. If None, column names are used.
            fill (bool, optional): Whether to fill the area under the lines. Defaults to False.
            tick_label_padding (int, optional): Padding for the tick labels. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if self.library != "matplotlib":
            raise NotImplementedError("Radar chart is currently only implemented for matplotlib.")

        metrics = labels or data.columns.tolist()
        data_values = data.values
        
        num_vars = len(metrics)
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        fig, ax = self._create_figure(figsize=params["figsize"], subplot_kw={'projection': 'polar'})

        # Plot each dataset
        for i, row in enumerate(data_values):
            values = np.concatenate((row, [row[0]]))
            current_theta = theta + theta[:1]
            color = self.colors['category_color_scale'][i % len(self.colors['category_color_scale'])]
            
            ax.plot(current_theta, values, color=color, linewidth=2, linestyle='solid', label=f"Dataset {i+1}")
            if fill:
                ax.fill(current_theta, values, color=color, alpha=0.25)
        
        ax.set_thetagrids(np.degrees(theta), metrics)
        ax.tick_params(axis='x', which='major', pad=tick_label_padding)

        self._set_labels(ax, subplot_title=params.get("subplot_title", "Radar Chart"))
        self._apply_common_layout(fig, params)
        
        return fig
