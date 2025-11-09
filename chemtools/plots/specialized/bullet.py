import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..base import BasePlotter

class BulletPlot(BasePlotter):
    """
    A plotter for creating bullet charts.
    """

    def plot(self, value: float, target: float, ranges: list, title: str = "Bullet Chart", **kwargs):
        """
        Plots a bullet chart.

        Args:
            value (float): The main value to display.
            target (float): The target value.
            ranges (list): A list of 3 values for the qualitative ranges (e.g., poor, average, good).
            title (str, optional): The title of the chart. Defaults to "Bullet Chart".
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])

            # Ranges
            ax.barh([0], ranges[2], color=self.colors['detail_light_color'])
            ax.barh([0], ranges[1], color=self.colors['detail_medium_color'])
            ax.barh([0], ranges[0], color=self.colors['text_color'])

            # Value
            ax.barh([0], value, color=self.colors['theme_color'])

            # Target
            ax.axvline(target, color=self.colors['accent_color'], linewidth=2)

            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            self._set_labels(ax, subplot_title=params.get("subplot_title", title))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            fig = go.Figure(go.Indicator(
                mode = "number+delta+gauge",
                value = value,
                gauge = {
                    'shape': "bullet",
                    'axis': {'range': [None, max(target, value, ranges[2]) * 1.1]},
                    'threshold': {
                        'line': {'color': self.colors['accent_color'], 'width': 2},
                        'thickness': 0.75, 'value': target},
                    'steps': [
                        {'range': [0, ranges[0]], 'color': self.colors['text_color']},
                        {'range': [ranges[0], ranges[1]], 'color': self.colors['detail_medium_color']},
                        {'range': [ranges[1], ranges[2]], 'color': self.colors['detail_light_color']}],
                    'bar': {'color': self.colors['theme_color']}
                },
                title = {'text': title}
            ))
            self._set_labels(fig, title=params.get("title", title))
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")