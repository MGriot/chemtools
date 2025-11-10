import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter

class BarPlot(BasePlotter):
    """
    A plotter for creating bar charts.
    Supports simple, stacked, and grouped bar charts.
    """

    def plot_counts(self, data: pd.DataFrame, column: str, **kwargs):
        """
        Plots a bar chart of value counts for a single categorical variable.
        """
        params = self._process_common_params(**kwargs)
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"Column '{column}' must be categorical to plot a bar chart.")

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            counts = data[column].value_counts()
            ax.bar(counts.index, counts.values, color=self.colors['category_color_scale'][:len(counts)])
            self._set_labels(ax, subplot_title=params.get("subplot_title", f'Bar Chart of {column}'), xlabel=column, ylabel='Count')
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            title = params.get("title", f'Bar Chart of {column}')
            # For plotly, we can use histogram for value counts of categorical data
            fig = px.histogram(data, x=column, title=title, color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            fig.update_layout(bargap=0.2) # A bit of gap for categorical data
            self._set_labels(fig, xlabel=column, ylabel='Count')
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")


    def plot(self, data: pd.DataFrame, x: str, y: str, color: str = None, mode: str = 'group', **kwargs):
        """
        Plots a bar chart, with options for stacking or grouping.
        """
        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            if color:
                pivot_df = data.groupby([x, color])[y].mean().unstack()
                if mode == 'stack':
                    pivot_df.plot(kind='bar', stacked=True, ax=ax, color=self.colors['category_color_scale'])
                    ax.tick_params(axis='x', rotation=0)
                else: # 'group'
                    import numpy as np
                    n_bars = len(pivot_df.columns)
                    n_groups = len(pivot_df.index)
                    
                    bar_width = 0.8 / n_bars
                    index = np.arange(n_groups)

                    for i, col in enumerate(pivot_df.columns):
                        # Center the group of bars around the tick
                        offset = (i - (n_bars - 1) / 2) * bar_width
                        ax.bar(index + offset, pivot_df[col], bar_width,
                               label=col, color=self.colors['category_color_scale'][i % len(self.colors['category_color_scale'])])

                    ax.set_xticks(index)
                    ax.set_xticklabels(pivot_df.index)
                    ax.tick_params(axis='x', rotation=0)
                    # The legend will be handled by _apply_common_layout
                self._set_labels(ax, subplot_title=params.get("subplot_title", f'{y} by {x} and {color}'), xlabel=x, ylabel=y)
            else:
                ax.bar(data[x], data[y], color=self.colors['theme_color'])
                self._set_labels(ax, subplot_title=params.get("subplot_title", f'{y} vs. {x}'), xlabel=x, ylabel=y)
            
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            barmode = 'group' if mode == 'group' else 'stack'
            title = params.get("title", f'{y} by {x}')
            fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, title=title, color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            self._set_labels(fig, xlabel=x, ylabel=y)
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")

    def plot_crosstab(self, crosstab_data: pd.DataFrame, stacked: bool = True, normalize: bool = False, **kwargs):
        """
        Plots a bar chart directly from a crosstab-like DataFrame.

        Args:
            crosstab_data (pd.DataFrame): A DataFrame where the index represents the x-axis ticks,
                                          and columns represent the categories to stack or group.
            stacked (bool): If True, creates a stacked bar chart. If False, creates a grouped bar chart.
            normalize (bool): If True and stacked is True, normalizes the data to create a 100% stacked bar chart.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        
        plot_data = crosstab_data.copy()
        if normalize and stacked:
            plot_data = plot_data.div(plot_data.sum(axis=1), axis=0)

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            
            plot_data.plot(kind='bar', stacked=stacked, ax=ax, color=self.colors['category_color_scale'], rot=0)
            
            ylabel = "Percentage" if normalize and stacked else ("Count" if stacked else "Value")
            self._set_labels(ax, subplot_title=params.get("subplot_title", "Crosstab Plot"), 
                             xlabel=plot_data.index.name, ylabel=ylabel)
            
            if normalize and stacked:
                ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            melted_data = plot_data.reset_index().melt(
                id_vars=plot_data.index.name, 
                var_name='category', 
                value_name='value'
            )
            
            barmode = 'stack' if stacked else 'group'
            title = params.get("title", "Crosstab Plot")
            
            kwargs.pop('title', None)
            
            fig = px.bar(melted_data, x=plot_data.index.name, y='value', color='category', 
                         barmode=barmode, title=title, 
                         color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            
            if normalize and stacked:
                fig.update_layout(yaxis_tickformat='.0%')

            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")