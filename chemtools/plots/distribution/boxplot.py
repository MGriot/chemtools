import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter

class BoxPlot(BasePlotter):
    """
    A plotter for creating box and violin plots.
    """

    def plot(self, data: pd.DataFrame, x: str = None, y: str = None, mode: str = 'box', **kwargs):
        """
        Plots a box or violin plot.

        Args:
            data (pd.DataFrame): The dataset to use.
            x (str, optional): The categorical column for the x-axis. Defaults to None.
            y (str): The numerical column for the y-axis.
            mode (str, optional): 'box' or 'violin'. Defaults to 'box'.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if y and y not in data.columns:
            raise ValueError(f"Y-column '{y}' not found in the data.")
        if y and not pd.api.types.is_numeric_dtype(data[y]):
            raise TypeError(f"Y-column '{y}' must be numeric.")
        if x and x not in data.columns:
            raise ValueError(f"X-column '{x}' not found in the data.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            
            plot_data = [data[col].dropna() for col in (y if isinstance(y, list) else [y])]
            labels = [y]
            if x:
                labels = data[x].unique()
                plot_data = [data[y][data[x] == cat].dropna() for cat in labels]

            if mode == 'violin':
                parts = ax.violinplot(plot_data, showmeans=False, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(self.colors['theme_color'])
                    pc.set_edgecolor(self.colors['text_color'])
                    pc.set_alpha(0.6)
            else: # 'box'
                bp = ax.boxplot(plot_data, patch_artist=True) # patch_artist=True allows coloring boxes
                
                # Set colors for boxes
                for i, box in enumerate(bp['boxes']):
                    box.set_facecolor(self.colors['category_color_scale'][i % len(self.colors['category_color_scale'])])
                    box.set_edgecolor(self.colors['text_color'])
                
                # Set colors for medians
                for median in bp['medians']:
                    median.set_color(self.colors['accent_color'])
                
                # Set colors for whiskers and caps
                for whisker in bp['whiskers']:
                    whisker.set_color(self.colors['detail_medium_color'])
                for cap in bp['caps']:
                    cap.set_color(self.colors['detail_medium_color'])
                
                # Set colors for fliers (outliers)
                for flier in bp['fliers']:
                    flier.set_markerfacecolor(self.colors['accent_color'])
                    flier.set_markeredgecolor(self.colors['accent_color'])
                    flier.set_marker('o')
                    flier.set_markersize(6)

            if x:
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels)
            self._set_labels(ax, subplot_title=params.get("subplot_title", f'{mode.capitalize()} Plot of {y}'), xlabel=x, ylabel=y)
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            title = params.get("title", f'{mode.capitalize()} Plot of {y}')
            if mode == 'violin':
                fig = px.violin(data, x=x, y=y, title=title, color=x, color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            else: # 'box'
                fig = px.box(data, x=x, y=y, title=title, color=x, color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            
            self._set_labels(fig, xlabel=x, ylabel=y)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")