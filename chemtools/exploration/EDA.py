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
- https://en.wikipedia.org/wiki/Exploratory_data_analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from typing import Union

from chemtools.plots.distribution.histogram import HistogramPlot
from chemtools.plots.distribution.boxplot import BoxPlot
from chemtools.plots.basic.bar import BarPlot
from chemtools.plots.basic.pie import PiePlot
from chemtools.plots.relationship.scatterplot import ScatterPlot
from chemtools.plots.relationship.heatmap import HeatmapPlot
from chemtools.plots.temporal.run_chart import RunChartPlot
from chemtools.plots.specialized.parallel_coordinates import ParallelCoordinatesPlot
from chemtools.plots.relationship.pairplot import PairPlot
from chemtools.plots.violin import ViolinPlot
from chemtools.stats.univariate import descriptive_statistics


class ExploratoryDataAnalysis:
    """
    A class to perform Exploratory Data Analysis (EDA) on a pandas DataFrame.

    This class provides a comprehensive suite of tools for data quality assessment,
    preprocessing, and visualization to understand the main characteristics of a dataset.

    It includes methods for:
    - Univariate and multivariate statistical summaries.
    - Missing value analysis and imputation.
    - Outlier detection.
    - Feature redundancy analysis (correlation, VIF).
    - A factory for generating various plot types (histograms, scatter plots, etc.).
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
        self.plotter_kwargs = (
            plotter_kwargs if plotter_kwargs is not None else {"library": "matplotlib"}
        )

    def classify_variables(self) -> tuple[list, list]:
        """
        Classifies columns into numerical and categorical lists.

        Returns:
            tuple[list, list]: A tuple containing a list of numerical column names
                               and a list of categorical column names.
        """
        numerical_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        return numerical_cols, categorical_cols

    def get_categorical_summary(self) -> pd.DataFrame:
        """
        Provides a summary for categorical columns, including cardinality (unique values),
        the mode, and number of missing values.

        Returns:
            pd.DataFrame: A DataFrame summarizing each categorical column.
        """
        _, categorical_cols = self.classify_variables()
        if not categorical_cols:
            return pd.DataFrame(columns=["Cardinality", "Mode", "Missing"])

        summary_list = []
        for col in categorical_cols:
            summary_list.append({
                'Column': col,
                'Cardinality': self.data[col].nunique(),
                'Mode': self.data[col].mode()[0] if not self.data[col].isnull().all() else np.nan,
                'Missing': self.data[col].isnull().sum()
            })
        return pd.DataFrame(summary_list).set_index('Column')

    def get_crosstab(self, index_col: str, col_col: str, normalize: Union[bool, str] = False) -> pd.DataFrame:
        """
        Computes a cross-tabulation of two categorical variables.

        Args:
            index_col (str): The column to display as the index.
            col_col (str): The column to display as the columns.
            normalize (str, optional): If 'index', 'columns', or 'all', computes percentages
                                       over the given axis. Defaults to None (counts).

        Returns:
            pd.DataFrame: The contingency table.
        """
        _, categorical_cols = self.classify_variables()
        if index_col not in categorical_cols or col_col not in categorical_cols:
            raise ValueError(f"Both columns must be categorical. Got: {index_col}, {col_col}")
        
        return pd.crosstab(self.data[index_col], self.data[col_col], normalize=normalize)

    def get_numerical_by_categorical_summary(self, numerical_col: str, categorical_col: str) -> pd.DataFrame:
        """
        Provides summary statistics for a numerical variable grouped by a categorical variable.

        Args:
            numerical_col (str): The numerical column to analyze.
            categorical_col (str): The categorical column to group by.

        Returns:
            pd.DataFrame: A DataFrame with summary statistics (mean, median, std) for each category.
        """
        numerical_cols, categorical_cols = self.classify_variables()
        if numerical_col not in numerical_cols or categorical_col not in categorical_cols:
            raise ValueError("You must provide one valid numerical and one valid categorical column.")

        return self.data.groupby(categorical_col)[numerical_col].agg(['mean', 'median', 'std']).round(3)

    def get_univariate_summary(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Provides a non-graphical univariate summary of the numerical data,
        using the comprehensive descriptive_statistics function.

        Args:
            alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.

        Returns:
            pd.DataFrame: A DataFrame containing descriptive statistics for each numerical column.
        """
        numerical_cols = self.data.select_dtypes(include=np.number).columns
        if numerical_cols.empty:
            return pd.DataFrame()

        summary_data = {}
        for col in numerical_cols:
            summary_data[col] = descriptive_statistics(self.data[col], alpha=alpha)
        
        # Convert the dictionary of dictionaries to a DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Format confidence interval tuples for better display in DataFrame
        def format_ci(ci_tuple):
            if isinstance(ci_tuple, tuple) and len(ci_tuple) == 2:
                if np.isnan(ci_tuple[0]) or np.isnan(ci_tuple[1]):
                    return "NaN"
                return f"({ci_tuple[0]:.3f}, {ci_tuple[1]:.3f})"
            return ci_tuple

        ci_key = f"Confidence Interval (alpha={alpha})"
        if ci_key in summary_df.index:
            summary_df.loc[ci_key] = summary_df.loc[ci_key].apply(format_ci)

        return summary_df.round(3)

    def get_univariate_report(self, column: str, alpha: float = 0.05) -> dict:
        """
        Returns a comprehensive dictionary of descriptive statistics for a single column.

        Args:
            column (str): The name of the column to analyze.
            alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.

        Returns:
            dict: A dictionary containing descriptive statistics for the specified column.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' must be numeric for univariate analysis.")
        
        return descriptive_statistics(self.data[column], alpha=alpha)

    def get_correlation_matrix(self, **kwargs) -> pd.DataFrame:
        """
        Calculates the pairwise correlation of numerical columns.
        """
        numerical_data = self.data.select_dtypes(include=np.number)
        return numerical_data.corr(**kwargs)

    def get_missing_values_summary(self) -> pd.DataFrame:
        """
        Calculates the count and percentage of missing values for each column.
        """
        missing_count = self.data.isnull().sum()
        missing_percent = (missing_count / len(self.data)) * 100
        summary_df = pd.DataFrame(
            {"missing_count": missing_count, "missing_percent": missing_percent}
        )
        summary_df = summary_df[summary_df["missing_count"] > 0].sort_values(
            by="missing_percent", ascending=False
        )
        return summary_df

    def plot_missing_values(self, **kwargs):
        """
        Generates a heatmap to visualize the location of missing values.
        """
        if self.data.isnull().sum().sum() == 0:
            print("No missing values to plot.")
            return None

        temp_plotter = HeatmapPlot(**self.plotter_kwargs)
        temp_plotter.library = "matplotlib"
        temp_plotter._init_matplotlib_style()

        fig, ax = temp_plotter._create_figure(**kwargs)
        sns.heatmap(
            self.data.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax
        )
        ax.set_title(kwargs.get("subplot_title", "Missing Value Matrix"))
        temp_plotter._apply_common_layout(fig, kwargs)
        return fig

    def impute_missing_values(
        self, strategy: str = "mean", columns: list = None, value=None
    ) -> pd.DataFrame:
        """
        Imputes missing values in the dataset. Modifies the internal DataFrame in place.
        """
        target_cols = columns or self.data.columns[self.data.isnull().any()].tolist()

        for col in target_cols:
            if self.data[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    if strategy == "mean":
                        fill_value = self.data[col].mean()
                    elif strategy == "median":
                        fill_value = self.data[col].median()
                    elif strategy == "mode":
                        fill_value = self.data[col].mode()[0]
                    elif strategy == "constant":
                        if value is None:
                            raise ValueError("A 'value' must be provided for 'constant' strategy.")
                        fill_value = value
                    else:
                        raise ValueError(f"Unknown strategy '{strategy}' for numeric column '{col}'.")
                    self.data[col].fillna(fill_value, inplace=True)
                else:
                    if strategy == "mode":
                        fill_value = self.data[col].mode()[0]
                    elif strategy == "constant":
                        if value is None:
                            raise ValueError("A 'value' must be provided for 'constant' strategy.")
                        fill_value = value
                    else:
                        print(f"Warning: Strategy '{strategy}' not applicable for non-numeric column '{col}'. Using 'mode'.")
                        fill_value = self.data[col].mode()[0]
                    self.data[col].fillna(fill_value, inplace=True)
        return self.data

    def get_outliers(self, column: str, method: str = "iqr", threshold: float = None) -> pd.Series:
        """
        Detects outliers in a numerical column using the specified method.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' must be numeric for outlier detection.")

        data_col = self.data[column].dropna()

        if method == "iqr":
            iqr_threshold = threshold if threshold is not None else 1.5
            q1, q3 = data_col.quantile(0.25), data_col.quantile(0.75)
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - (iqr_threshold * iqr), q3 + (iqr_threshold * iqr)
            return data_col[(data_col < lower_bound) | (data_col > upper_bound)]
        elif method == "zscore":
            z_threshold = threshold if threshold is not None else 3.0
            mean, std = data_col.mean(), data_col.std()
            if std == 0: return pd.Series(dtype=data_col.dtype)
            z_scores = (data_col - mean) / std
            return data_col[np.abs(z_scores) > z_threshold]
        else:
            raise ValueError(f"Unknown outlier detection method: '{method}'. Use 'iqr' or 'zscore'.")

    def get_data_types(self) -> pd.Series:
        """Returns the data types of each column."""
        return self.data.dtypes

    def get_mixed_type_columns(self) -> dict:
        """Identifies columns that contain mixed data types."""
        mixed_type_cols = {}
        for col in self.data.select_dtypes(include=["object"]).columns:
            unique_types = self.data[col].dropna().apply(type).unique()
            if len(unique_types) > 1:
                mixed_type_cols[col] = [t.__name__ for t in unique_types]
        return mixed_type_cols

    def get_high_correlation_pairs(self, threshold: float = 0.9) -> pd.DataFrame:
        """Finds pairs of numerical features with correlation above a threshold."""
        corr_matrix = self.get_correlation_matrix()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = upper_triangle[abs(upper_triangle) > threshold].stack().reset_index()
        high_corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        return high_corr_pairs

    def get_vif(self) -> pd.DataFrame:
        """Calculates the Variance Inflation Factor (VIF) for each numerical feature."""
        numerical_data = self.data.select_dtypes(include=np.number).dropna()
        if numerical_data.shape[1] < 2:
            print("VIF requires at least two numerical features.")
            return pd.DataFrame(columns=["Feature", "VIF"])

        X = add_constant(numerical_data)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data[vif_data["Feature"] != "const"].sort_values("VIF", ascending=False)

    def histogram_plotter(self) -> HistogramPlot:
        return HistogramPlot(**self.plotter_kwargs)

    def boxplot_plotter(self) -> BoxPlot:
        return BoxPlot(**self.plotter_kwargs)

    def barchart_plotter(self) -> BarPlot:
        return BarPlot(**self.plotter_kwargs)

    def scatter_plotter(self) -> ScatterPlot:
        return ScatterPlot(**self.plotter_kwargs)

    def heatmap_plotter(self) -> HeatmapPlot:
        return HeatmapPlot(**self.plotter_kwargs)

    def run_chart_plotter(self) -> RunChartPlot:
        return RunChartPlot(**self.plotter_kwargs)

    def parallel_coordinates_plotter(self) -> ParallelCoordinatesPlot:
        return ParallelCoordinatesPlot(**self.plotter_kwargs)

    def pairplot_plotter(self) -> PairPlot:
        return PairPlot(**self.plotter_kwargs)

    def pie_chart_plotter(self) -> PiePlot:
        return PiePlot(**self.plotter_kwargs)

    def violin_plotter(self) -> ViolinPlot:
        return ViolinPlot(**self.plotter_kwargs)

    def plot_numerical_by_categorical(self, numerical_col: str, categorical_col: str, plot_type: str = "box", **kwargs):
        """
        Generates a plot of a numerical variable's distribution across categories of a categorical variable.

        This is a key visualization for mixed-type EDA.

        Args:
            numerical_col (str): The numerical column (y-axis).
            categorical_col (str): The categorical column (x-axis).
            plot_type (str, optional): The type of plot to generate.
                                       Can be "box" or "violin". Defaults to "box".
            **kwargs: Additional keyword arguments passed to the plotter. Can include
                      `plotter_kwargs` to override theme/library for this specific plot.

        Returns:
            A matplotlib or plotly figure object.
        """
        # Extract plotter_kwargs for plotter instantiation, remove them from kwargs for the plot method
        plotter_specific_kwargs = kwargs.pop('plotter_kwargs', self.plotter_kwargs)

        if plot_type == "box":
            plotter = self.boxplot_plotter() # This uses instance default, let's change it
            plotter = BoxPlot(**plotter_specific_kwargs)
        elif plot_type == "violin":
            plotter = ViolinPlot(**plotter_specific_kwargs)
        else:
            raise ValueError("plot_type must be 'box' or 'violin'")

        # Set default title if not provided
        if 'title' not in kwargs and 'subplot_title' not in kwargs:
            kwargs['subplot_title'] = f'{numerical_col} Distribution by {categorical_col}'

        return plotter.plot(data=self.data, x=categorical_col, y=numerical_col, **kwargs)

    def plot_stem_and_leaf(self, column: str):
        """Generates and prints a stem-and-leaf plot for a numerical variable."""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' must be numeric for a stem-and-leaf plot.")

        data = self.data[column].dropna().values
        if len(data) == 0:
            print(f"No data to plot for {column}")
            return

        scaled_data = data * 10
        stems, leaves = np.floor(scaled_data / 10), np.round(scaled_data % 10)
        stem_dict = {}
        for s, l in zip(stems, leaves):
            s = int(s)
            stem_dict.setdefault(s, []).append(int(l))

        print(f"Stem-and-leaf plot for {column}\nStem | Leaf\n---- | ----")
        for stem in sorted(stem_dict.keys()):
            leaves_str = " ".join(map(str, sorted(stem_dict[stem])))
            print(f"{stem:4d} | {leaves_str}")