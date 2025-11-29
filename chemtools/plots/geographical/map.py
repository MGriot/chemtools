import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter

class MapPlot(BasePlotter):
    """
    A plotter for creating map-based plots (Choropleth and Geo Scatter).
    """

    def plot_choropleth(self, data: pd.DataFrame, locations_column: str, values_column: str, **kwargs):
        """
        Plots a choropleth map.

        Args:
            data (pd.DataFrame): The dataset to use.
            locations_column (str): The column with country names or codes.
            values_column (str): The column with the values to plot.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            print("Warning: Matplotlib has limited support for choropleth maps without external libraries like geopandas. A basic plot will be rendered.")
            # This is a very basic representation and not a real map.
            fig, ax = self._create_figure(figsize=params["figsize"])
            # Sort data for a more visually appealing bar chart representation
            sorted_data = data.sort_values(by=values_column, ascending=False)
            ax.bar(sorted_data[locations_column], sorted_data[values_column], color=self.colors['theme_color'])
            plt.xticks(rotation=90)
            self._set_labels(ax, subplot_title=params.get("subplot_title", f'Choropleth Data for {values_column}'), xlabel='Location', ylabel=values_column)
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            title = params.get("title", f'Choropleth Map of {values_column}')
            
            plotly_kwargs = kwargs.copy()
            plotly_kwargs.pop('title', None)
            plotly_kwargs.pop('subplot_title', None)

            # Get width and height from params and pass explicitly
            plot_width = params.get("width")
            plot_height = params.get("height")
            
            fig = px.choropleth(data, locations=locations_column, color=values_column,
                                title=title, color_continuous_scale=px.colors.sequential.Plasma,
                                width=plot_width, height=plot_height, **plotly_kwargs)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")

    def plot_scatter_geo(self, data: pd.DataFrame, lat_column: str, lon_column: str, **kwargs):
        """
        Plots a scatter plot on a map.

        Args:
            data (pd.DataFrame): The dataset to use.
            lat_column (str): The column with latitude values.
            lon_column (str): The column with longitude values.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            print("Warning: Matplotlib has limited support for geo scatter plots without external libraries like geopandas or cartopy. A basic scatter plot of lat/lon will be rendered.")
            fig, ax = self._create_figure(figsize=params["figsize"])
            ax.scatter(data[lon_column], data[lat_column], color=self.colors['accent_color'], alpha=0.6)
            self._set_labels(ax, subplot_title=params.get("subplot_title", 'Geo Scatter Data'), xlabel='Longitude', ylabel='Latitude')
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            title = params.get("title", 'Geo Scatter Plot')
            
            plotly_kwargs = kwargs.copy()
            plotly_kwargs.pop('title', None) 
            plotly_kwargs.pop('subplot_title', None) 

            color_col = plotly_kwargs.pop('color', None) # Remove 'color' from kwargs to handle it explicitly

            # Get width and height from params and pass explicitly
            plot_width = params.get("width")
            plot_height = params.get("height")

            if color_col:
                fig = px.scatter_geo(data, lat=lat_column, lon=lon_column, title=title,
                                     color=color_col, color_discrete_sequence=self.colors['category_color_scale'],
                                     width=plot_width, height=plot_height, **plotly_kwargs)
            else:
                fig = px.scatter_geo(data, lat=lat_column, lon=lon_column, title=title,
                                     color_discrete_sequence=[self.colors['accent_color']],
                                     width=plot_width, height=plot_height, **plotly_kwargs)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")