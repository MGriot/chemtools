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
                else: # 'group'
                    pivot_df.plot(kind='bar', stacked=False, ax=ax, color=self.colors['category_color_scale'])
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

    def plot_crosstab(self, crosstab_data: pd.DataFrame, stacked: bool = True, **kwargs):
        """
        Plots a bar chart directly from a crosstab-like DataFrame.

        Args:
            crosstab_data (pd.DataFrame): A DataFrame where the index represents the x-axis ticks,
                                          and columns represent the categories to stack or group.
            stacked (bool): If True, creates a stacked bar chart. If False, creates a grouped bar chart.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        params = self._process_common_params(**kwargs)
        
        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            
            # Use the DataFrame's own plot method, but pass the themed ax
            crosstab_data.plot(kind='bar', stacked=stacked, ax=ax, color=self.colors['category_color_scale'], rot=0)
            
            # Set labels and title using our themed functions
            self._set_labels(ax, subplot_title=params.get("subplot_title", "Crosstab Plot"), 
                             xlabel=crosstab_data.index.name, ylabel="Count" if stacked else "Value")
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            # Plotly's px.bar works best with long-form data. We need to melt the crosstab data.
            melted_data = crosstab_data.reset_index().melt(
                id_vars=crosstab_data.index.name, 
                var_name='category', 
                value_name='value'
            )
            
            barmode = 'stack' if stacked else 'group'
            title = params.get("title", "Crosstab Plot")
            
            # Pop title from kwargs to avoid conflict
            kwargs.pop('title', None)
            
            fig = px.bar(melted_data, x=crosstab_data.index.name, y='value', color='category', 
                         barmode=barmode, title=title, 
                         color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")