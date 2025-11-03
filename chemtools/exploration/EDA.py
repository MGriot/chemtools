"""
chemtools.exploration.EDA
-------------------------

This module provides the ExploratoryDataAnalysis class for performing 
comprehensive Exploratory Data Analysis (EDA) on a dataset.

EDA is used to analyze and investigate data sets and summarize their main
characteristics, often employing data visualization methods. This module provides
a suite of tools to perform univariate and multivariate analysis, both
graphically and non-graphically.

This implementation is inspired by the work of John Tukey.

References:
- Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.

Example usage:
>>> from chemtools.exploration import ExploratoryDataAnalysis
>>> import pandas as pd
>>> # Assuming 'df' is your pandas DataFrame
>>> eda = ExploratoryDataAnalysis(df)
>>> summary = eda.get_univariate_summary()
>>> print(summary)
>>> fig_hist = eda.plot_histogram('numerical_column')
>>> fig_hist.show()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chemtools.plots.Plotter import Plotter

class ExploratoryDataAnalysis:
    """
    A class to perform Exploratory Data Analysis (EDA) on a pandas DataFrame.
    """

    def __init__(self, data: pd.DataFrame, plotter_kwargs: dict = None):
        """
        Initializes the ExploratoryDataAnalysis class.

        Args:
            data (pd.DataFrame): The dataset to analyze.
            plotter_kwargs (dict, optional): Keyword arguments for the Plotter class.
                                             Defaults to {'library': 'matplotlib'}.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        self.data = data
        self.plotter_kwargs = plotter_kwargs if plotter_kwargs is not None else {'library': 'matplotlib'}
        self.plotter = Plotter(**self.plotter_kwargs)

    def get_univariate_summary(self) -> pd.DataFrame:
        """
        Provides a non-graphical univariate summary of the data.

        For numerical columns, it calculates count, mean, std, min, 25%, 50%, 75%, max.
        For categorical columns, it calculates count, unique, top, freq.

        Returns:
            pd.DataFrame: A DataFrame containing the summary statistics.
        """
        return self.data.describe(include='all')

    def plot_histogram(self, column: str, **kwargs):
        """
        Plots a histogram for a single numerical variable.

        Args:
            column (str): The name of the column to plot.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' must be numeric to plot a histogram.")

        fig, ax = self.plotter._create_figure(**kwargs)
        sns.histplot(data=self.data, x=column, ax=ax, kde=True, color=self.plotter.colors['theme_color'])
        self.plotter._set_labels(ax, title=f'Histogram of {column}', xlabel=column, ylabel='Frequency')
        return self.plotter.apply_style_preset(fig)

    def plot_boxplot(self, column: str, **kwargs):
        """
        Plots a box plot for a single numerical variable.

        Args:
            column (str): The name of the column to plot.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' must be numeric to plot a box plot.")

        fig, ax = self.plotter._create_figure(**kwargs)
        sns.boxplot(data=self.data, y=column, ax=ax, color=self.plotter.colors['theme_color'])
        self.plotter._set_labels(ax, title=f'Box Plot of {column}', ylabel=column)
        return self.plotter.apply_style_preset(fig)

    def get_correlation_matrix(self, **kwargs) -> pd.DataFrame:
        """
        Calculates the pairwise correlation of columns.

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        numerical_data = self.data.select_dtypes(include=np.number)
        return numerical_data.corr(**kwargs)

    def plot_heatmap(self, **kwargs):
        """
        Plots a heatmap of the correlation matrix.

        Args:
            **kwargs: Additional keyword arguments passed to sns.heatmap.

        Returns:
            A matplotlib figure object.
        """
        corr_matrix = self.get_correlation_matrix()
        fig, ax = self.plotter._create_figure()
        sns.heatmap(corr_matrix, ax=ax, annot=True, cmap='coolwarm', **kwargs)
        self.plotter._set_labels(ax, title='Correlation Heatmap')
        return self.plotter.apply_style_preset(fig)

    def plot_scatter(self, x_column: str, y_column: str, **kwargs):
        """
        Creates a scatter plot of two numerical variables.

        Args:
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if x_column not in self.data.columns or y_column not in self.data.columns:
            raise ValueError("One or both columns not found in the data.")
        if not pd.api.types.is_numeric_dtype(self.data[x_column]) or not pd.api.types.is_numeric_dtype(self.data[y_column]):
            raise TypeError("Both columns must be numeric for a scatter plot.")

        fig, ax = self.plotter._create_figure(**kwargs)
        sns.scatterplot(data=self.data, x=x_column, y=y_column, ax=ax, color=self.plotter.colors['accent_color'])
        self.plotter._set_labels(ax, title=f'{y_column} vs. {x_column}', xlabel=x_column, ylabel=y_column)
        return self.plotter.apply_style_preset(fig)

    def plot_barchart(self, column: str, **kwargs):
        """
        Plots a bar chart for a single categorical variable.

        Args:
            column (str): The name of the column to plot.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' must be categorical to plot a bar chart.")

        fig, ax = self.plotter._create_figure(**kwargs)
        
        # Fix for warnings
        num_categories = self.data[column].nunique()
        palette = self.plotter.colors['category_color_scale'][:num_categories]
        sns.countplot(data=self.data, x=column, ax=ax, hue=column, palette=palette, legend=False)

        self.plotter._set_labels(ax, title=f'Bar Chart of {column}', xlabel=column, ylabel='Count')
        return self.plotter.apply_style_preset(fig)

    def plot_run_chart(self, time_column: str, value_column: str, **kwargs):
        """
        Plots a run chart of a variable over time.

        Args:
            time_column (str): The column representing time.
            value_column (str): The column representing the value to plot.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib or plotly figure object.
        """
        if time_column not in self.data.columns or value_column not in self.data.columns:
            raise ValueError("One or both columns not found in the data.")

        fig, ax = self.plotter._create_figure(**kwargs)
        sns.lineplot(data=self.data, x=time_column, y=value_column, ax=ax, color=self.plotter.colors['theme_color'])
        self.plotter._set_labels(ax, title=f'Run Chart of {value_column} over {time_column}', xlabel=time_column, ylabel=value_column)
        return self.plotter.apply_style_preset(fig)

    def plot_stem_and_leaf(self, column: str):
        """
        Generates and prints a stem-and-leaf plot for a numerical variable.

        Args:
            column (str): The name of the column to plot.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' must be numeric for a stem-and-leaf plot.")

        data = self.data[column].dropna().values
        
        # Improved logic for stem-and-leaf
        if len(data) == 0:
            print(f"No data to plot for {column}")
            return
            
        scaled_data = data * 10
        stems = np.floor(scaled_data / 10)
        leaves = np.round(scaled_data % 10)

        stem_dict = {}
        for s, l in zip(stems, leaves):
            s = int(s)
            if s not in stem_dict:
                stem_dict[s] = []
            stem_dict[s].append(int(l))

        print(f"Stem-and-leaf plot for {column}")
        print("Stem | Leaf")
        print("---- | ----")
        for stem in sorted(stem_dict.keys()):
            leaves_str = ' '.join(map(str, sorted(stem_dict[stem])))
            print(f"{stem:4d} | {leaves_str}")

    def plot_scatter_3d(self, x_column: str, y_column: str, z_column: str, **kwargs):
        """
        Creates a 3D scatter plot of three numerical variables.

        Args:
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
            z_column (str): The column for the z-axis.
            **kwargs: Additional keyword arguments passed to the plotter.

        Returns:
            A matplotlib figure object.
        """
        if x_column not in self.data.columns or y_column not in self.data.columns or z_column not in self.data.columns:
            raise ValueError("One or more columns not found in the data.")
        if not pd.api.types.is_numeric_dtype(self.data[x_column]) or not pd.api.types.is_numeric_dtype(self.data[y_column]) or not pd.api.types.is_numeric_dtype(self.data[z_column]):
            raise TypeError("All three columns must be numeric for a 3D scatter plot.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(self.data[x_column], self.data[y_column], self.data[z_column], c=self.plotter.colors['accent_color'])
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_zlabel(z_column)
        ax.set_title(f'3D Scatter Plot: {x_column}, {y_column}, {z_column}')
        
        return fig

    def plot_parallel_coordinates(self, class_column: str, **kwargs):
        """
        Creates a parallel coordinates plot.

        Args:
            class_column (str): The column to color the lines by.
            **kwargs: Additional keyword arguments passed to pandas.plotting.parallel_coordinates.

        Returns:
            A matplotlib figure object.
        """
        if class_column not in self.data.columns:
            raise ValueError(f"Column '{class_column}' not found in the data.")

        # Select only numerical columns and the class column for the plot
        numerical_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        plot_data = self.data[numerical_cols + [class_column]]

        fig, ax = self.plotter._create_figure()
        pd.plotting.parallel_coordinates(plot_data, class_column, ax=ax, **kwargs)
        self.plotter._set_labels(ax, title='Parallel Coordinates Plot')
        return self.plotter.apply_style_preset(fig)