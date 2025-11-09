import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from .base import BasePlotter

class ViolinPlot(BasePlotter):
    """
    Creates a Violin Plot. Inherits from BasePlotter.
    """

    def __init__(self, **kwargs):
        """
        Initializes the ViolinPlot class.

        Args:
            **kwargs: Keyword arguments for the BasePlotter.
        """
        super().__init__(**kwargs)

    def plot(self, data: pd.DataFrame, y: str, x: str = None, **kwargs):
        """
        Plots a violin plot.

        Args:
            data (pd.DataFrame): The dataset to use.
            y (str): The numerical column for the violin plot.
            x (str, optional): The categorical column to group by. Defaults to None.
            **kwargs: Additional keyword arguments passed to the plotter.
        
        Returns:
            A matplotlib or plotly figure object.
        """
        # Pop plotter-specific kwargs before passing to plotly
        plotter_kwargs = ['library', 'theme', 'style_preset', 'watermark', 'figsize']
        for k in plotter_kwargs:
            kwargs.pop(k, None)

        params = self._process_common_params(**kwargs)
        if y not in data.columns:
            raise ValueError(f"Value column '{y}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(data[y]):
            raise TypeError(f"Value column '{y}' must be numeric for a violin plot.")
        if x and x not in data.columns:
            raise ValueError(f"Category column '{x}' not found in the data.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            
            plot_data = [data[y].dropna()]
            labels = [y]
            if x:
                labels = sorted(data[x].dropna().unique())
                plot_data = [data[y][data[x] == cat].dropna() for cat in labels]

            parts = ax.violinplot(plot_data, showmeans=False, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                # Assign a color from the category_color_scale for each violin body
                color_index = i % len(self.colors['category_color_scale'])
                pc.set_facecolor(self.colors['category_color_scale'][color_index])
                pc.set_edgecolor(self.colors['text_color'])
                pc.set_alpha(0.6)
            
            # Style the median and other lines
            for part_name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if part_name in parts:
                    vp = parts[part_name]
                    vp.set_edgecolor(self.colors['text_color'])
                    vp.set_linewidth(1)

            if x:
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45)
            
            self._set_labels(ax, subplot_title=params.get("subplot_title"), xlabel=x, ylabel=y)
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            # Pop title from kwargs to avoid conflict
            kwargs.pop('title', None)
            
            # Create a color map for each unique category in x
            unique_categories = data[x].dropna().unique()
            color_map = {
                cat: self.colors['category_color_scale'][i % len(self.colors['category_color_scale'])]
                for i, cat in enumerate(unique_categories)
            }
            
            fig = px.violin(data, x=x, y=y, title=params.get('title'), color=x, 
                            color_discrete_map=color_map, **kwargs)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")
