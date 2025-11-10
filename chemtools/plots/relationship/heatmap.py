import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..base import BasePlotter

class HeatmapPlot(BasePlotter):
    """
    A plotter for creating heatmaps.
    """

    def plot(self, data: pd.DataFrame, annot: bool = False, **kwargs):
        """
        Plots a heatmap of a given matrix.

        Args:
            data (pd.DataFrame): The matrix to plot.
            annot (bool): If True, write the data value in each cell.
            **kwargs: Additional keyword arguments passed to the plotter.
                      Can include 'title' for the figure title.

        Returns:
            A matplotlib or plotly figure object.
        """
        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            cmap = kwargs.get("cmap", 'coolwarm')
            im = ax.imshow(data, cmap=cmap, aspect='auto')
            
            plt.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(len(data.columns)))
            ax.set_yticks(np.arange(len(data.index)))
            ax.set_xticklabels(data.columns, rotation=90)
            ax.set_yticklabels(data.index)
            ax.grid(False)

            if annot:
                cmap = plt.get_cmap(cmap)
                norm = plt.Normalize(vmin=data.values.min(), vmax=data.values.max())
                
                for i in range(len(data.index)):
                    for j in range(len(data.columns)):
                        val = data.iloc[i, j]
                        cell_color = cmap(norm(val))
                        luminance = 0.299*cell_color[0] + 0.587*cell_color[1] + 0.114*cell_color[2]
                        text_color = "white" if luminance < 0.5 else "black"
                        
                        fmt = "{:.2f}"
                        if np.issubdtype(type(val), np.integer):
                            fmt = "{:d}"
                        
                        ax.text(j, i, fmt.format(val),
                                ha="center", va="center", color=text_color)

            self._set_labels(ax, subplot_title=params.get('subplot_title', 'Heatmap'))
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            title = params.get('title', 'Heatmap')
            fig = go.Figure(data=go.Heatmap(
                z=data.values, 
                x=data.columns, 
                y=data.index, 
                colorscale=kwargs.get("colorscale", 'RdBu'),
                text=data.values if annot else None,
                texttemplate="%{text}" if annot else None
            ))
            self._set_labels(fig, title=title)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")

    def plot_categorical(self, data: pd.DataFrame, x_column: str, y_column: str, **kwargs):
        """
        Creates a heatmap of the co-occurrence frequency of two categorical variables.

        Args:
            data (pd.DataFrame): The dataset to use.
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError("One or both columns not found in the data.")
        
        crosstab_data = pd.crosstab(data[x_column], data[y_column])
        
        # Call the existing plot method with the crosstab data
        return self.plot(crosstab_data, **kwargs)