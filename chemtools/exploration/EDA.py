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
>>> hist_plotter = eda.histogram_plotter()
>>> fig_hist = hist_plotter.plot(eda.data, 'numerical_column')
>>> fig_hist.show()
"""

import pandas as pd
import numpy as np
from chemtools.plots.common.histogram import HistogramPlotter
from chemtools.plots.common.boxplot import BoxPlotter
from chemtools.plots.common.barchart import BarChartPlotter
from chemtools.plots.common.scatterplot import ScatterPlotter
from chemtools.plots.common.heatmap import HeatmapPlotter
from chemtools.plots.common.run_chart import RunChartPlotter
from chemtools.plots.common.parallel_coordinates import ParallelCoordinatesPlotter

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

    def get_univariate_summary(self) -> pd.DataFrame:
        """
        Provides a non-graphical univariate summary of the data.

        For numerical columns, it calculates count, mean, std, min, 25%, 50%, 75%, max.
        For categorical columns, it calculates count, unique, top, freq.

        Returns:
            pd.DataFrame: A DataFrame containing the summary statistics.
        """
        return self.data.describe(include='all')

    def get_correlation_matrix(self, **kwargs) -> pd.DataFrame:
        """
        Calculates the pairwise correlation of columns.

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        numerical_data = self.data.select_dtypes(include=np.number)
        return numerical_data.corr(**kwargs)

    def histogram_plotter(self) -> HistogramPlotter:
        """
        Returns a HistogramPlotter instance.
        """
        return HistogramPlotter(**self.plotter_kwargs)

    def boxplot_plotter(self) -> BoxPlotter:
        """
        Returns a BoxPlotter instance.
        """
        return BoxPlotter(**self.plotter_kwargs)

    def barchart_plotter(self) -> BarChartPlotter:
        """
        Returns a BarChartPlotter instance.
        """
        return BarChartPlotter(**self.plotter_kwargs)

    def scatter_plotter(self) -> ScatterPlotter:
        """
        Returns a ScatterPlotter instance.
        """
        return ScatterPlotter(**self.plotter_kwargs)

    def heatmap_plotter(self) -> HeatmapPlotter:
        """
        Returns a HeatmapPlotter instance.
        """
        return HeatmapPlotter(**self.plotter_kwargs)

    def run_chart_plotter(self) -> RunChartPlotter:
        """
        Returns a RunChartPlotter instance.
        """
        return RunChartPlotter(**self.plotter_kwargs)

    def parallel_coordinates_plotter(self) -> ParallelCoordinatesPlotter:
        """
        Returns a ParallelCoordinatesPlotter instance.
        """
        return ParallelCoordinatesPlotter(**self.plotter_kwargs)

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
