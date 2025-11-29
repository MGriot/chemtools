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
            
            color_by_col = params.get("color_by")
            
            legend_handles = []
            legend_labels = []

            mosaic_columns = list(columns) # Base columns for the mosaic plot
            
            if color_by_col and color_by_col in data.columns:
                # Ensure color_by_col is in the list of columns for grouping and for mosaic function
                if color_by_col not in mosaic_columns:
                    mosaic_columns.append(color_by_col)
                # Make color_by_col the last element in the grouping for consistent key extraction
                else:
                    mosaic_columns.remove(color_by_col)
                    mosaic_columns.append(color_by_col)
                
                plot_data = data.groupby(mosaic_columns).size()

                unique_color_categories = data[color_by_col].unique()
                color_palette = self.colors['category_color_scale']
                color_map = {
                    category: color_palette[i % len(color_palette)]
                    for i, category in enumerate(unique_color_categories)
                }

                def get_tile_color(key):
                    if isinstance(key, tuple) and len(key) > 0:
                        category_value = key[-1] 
                    else:
                        return self.colors['theme_color'] # Fallback
                    
                    color = color_map.get(category_value, self.colors['theme_color'])
                    return color

                properties = lambda k: {'facecolor': get_tile_color(k), 'edgecolor': self.colors['text_color']}
                
                # Create legend handles and labels
                from matplotlib.patches import Patch
                for category, color in color_map.items():
                    legend_handles.append(Patch(color=color, label=str(category)))
                    legend_labels.append(str(category))
                
            else:
                properties = None
                plot_data = data.groupby(mosaic_columns).size() # Use original columns for grouping if no color_by
            
            fig, ax = self._create_figure(figsize=params["figsize"])
            mosaic(plot_data, ax=ax, title=params.get("subplot_title", "Mosaic Plot"), properties=properties, gap=0.01)
            
            # Pass legend info to _apply_common_layout via params
            if legend_handles and legend_labels:
                params['legend_handles'] = legend_handles
                params['legend_labels'] = legend_labels
                params['showlegend'] = True # Ensure legend is shown if handles are provided

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