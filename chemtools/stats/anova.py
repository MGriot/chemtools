import numpy as np
import pandas as pd
from scipy.stats import f
from typing import Union, List, Dict, Any, Optional

from chemtools.base.base_models import BaseModel
from chemtools.stats.regression_stats import calculate_degrees_of_freedom


class OneWayANOVA(BaseModel):
    """
    Performs One-Way Analysis of Variance (ANOVA).

    One-Way ANOVA is used to determine whether there are any statistically
    significant differences between the means of two or more independent
    (unrelated) groups.

    Attributes:
        model_name (str): The name of the model.
        method (str): The method name, "One-Way ANOVA".
        data (pd.DataFrame): The input data.
        group_column (str): The name of the column containing group labels.
        value_column (str): The name of the column containing the values to analyze.
        q (int): Number of groups.
        n_total (int): Total number of observations.
        group_means (pd.Series): Mean of values for each group.
        group_counts (pd.Series): Number of observations in each group.
        total_mean (float): Grand mean of all observations.
        SScorr (float): Total sum of squares corrected for the mean.
        SSfact (float): Sum of squares between groups (factor).
        SSR (float): Residual sum of squares within groups.
        df_fact (int): Degrees of freedom for the factor.
        df_res (int): Degrees of freedom for the residuals.
        MSfact (float): Mean square for the factor.
        MSR (float): Mean square for the residuals.
        F_value (float): Calculated F-statistic.
        p_value (float): P-value corresponding to the F-statistic.

    References:
        - https://en.wikipedia.org/wiki/One-way_analysis_of_variance
    """

    def __init__(self):
        super().__init__()
        self.model_name = "One-Way Analysis of Variance"
        self.method = "One-Way ANOVA"
        self.data: Optional[pd.DataFrame] = None
        self.group_column: Optional[str] = None
        self.value_column: Optional[str] = None
        self.q: Optional[int] = None
        self.n_total: Optional[int] = None
        self.group_means: Optional[pd.Series] = None
        self.group_counts: Optional[pd.Series] = None
        self.total_mean: Optional[float] = None
        self.SScorr: Optional[float] = None
        self.SSfact: Optional[float] = None
        self.SSR: Optional[float] = None
        self.df_fact: Optional[int] = None
        self.df_res: Optional[int] = None
        self.MSfact: Optional[float] = None
        self.MSR: Optional[float] = None
        self.F_value: Optional[float] = None
        self.p_value: Optional[float] = None

    def fit(self, data: pd.DataFrame, value_column: str, group_column: str):
        """
        Fits the One-Way ANOVA model to the provided data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            value_column (str): The name of the column containing the numerical values.
            group_column (str): The name of the column containing the group labels.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if value_column not in data.columns:
            raise ValueError(f"Value column '{value_column}' not found in data.")
        if group_column not in data.columns:
            raise ValueError(f"Group column '{group_column}' not found in data.")
        if not pd.api.types.is_numeric_dtype(data[value_column]):
            raise TypeError(f"Value column '{value_column}' must be numeric.")

        self.data = data.dropna(subset=[value_column, group_column]).copy()
        self.value_column = value_column
        self.group_column = group_column

        self.q = self.data[group_column].nunique()
        self.n_total = len(self.data)

        if self.q < 2:
            raise ValueError("One-Way ANOVA requires at least two groups.")
        if self.n_total < self.q:
            raise ValueError(
                "Total number of observations must be greater than or equal to the number of groups."
            )

        self.group_means = self.data.groupby(group_column)[value_column].mean()
        self.group_counts = self.data.groupby(group_column)[value_column].count()
        self.total_mean = self.data[value_column].mean()

        # Calculate Sum of Squares Corrected for the Mean (SScorr)
        self.SScorr = np.sum((self.data[value_column] - self.total_mean) ** 2)

        # Calculate Sum of Squares Between Groups (SSfact)
        self.SSfact = np.sum(
            self.group_counts * (self.group_means - self.total_mean) ** 2
        )

        # Calculate Residual Sum of Squares Within Groups (SSR)
        self.SSR = np.sum(
            (self.data.groupby(group_column)[value_column].apply(
                lambda x: (x - x.mean()) ** 2
            )).explode()
        )

        # Degrees of Freedom
        self.df_fact = self.q - 1
        self.df_res = self.n_total - self.q

        # Mean Squares
        self.MSfact = self.SSfact / self.df_fact
        self.MSR = self.SSR / self.df_res

        # F-value
        self.F_value = self.MSfact / self.MSR

        # P-value
        self.p_value = 1 - f.cdf(self.F_value, self.df_fact, self.df_res)

    def _get_summary_data(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing summary data for the One-Way ANOVA model.
        """
        if self.F_value is None:
            return {}

        summary = self._create_general_summary(
            n_variables=1,  # Only one dependent variable
            n_objects=self.n_total,
            Groups=self.q,
            Value_Column=self.value_column,
            Group_Column=self.group_column,
        )

        anova_table = [
            ["Source of Variation", "df", "Sum of Squares", "Mean Square", "F-value", "P-value"],
            ["Between Groups (Factor)", self.df_fact, f"{self.SSfact:.3f}", f"{self.MSfact:.3f}", f"{self.F_value:.3f}", f"{self.p_value:.3f}"],
            ["Within Groups (Residual)", self.df_res, f"{self.SSR:.3f}", f"{self.MSR:.3f}", "", ""],
            ["Total (Corrected)", self.df_fact + self.df_res, f"{self.SScorr:.3f}", "", "", ""],
        ]
        summary["tables"] = {"ANOVA Table": anova_table}

        return summary


class TwoWayANOVA(BaseModel):
    """
    Performs Two-Way Analysis of Variance (ANOVA) for a balanced design with repetitions.

    Two-Way ANOVA is used to evaluate simultaneously the effect of two categorical
    independent variables (factors) on a continuous dependent variable, and also
    to assess if there is an interaction effect between the two factors.

    This implementation assumes a balanced design (equal number of repetitions
    in each cell) and requires at least two repetitions per cell to calculate
    the interaction effect.

    Attributes:
        model_name (str): The name of the model.
        method (str): The method name, "Two-Way ANOVA".
        data (pd.DataFrame): The input data.
        value_column (str): The name of the column containing the values to analyze.
        factor1_column (str): The name of the first factor column.
        factor2_column (str): The name of the second factor column.
        a (int): Number of levels for Factor 1.
        b (int): Number of levels for Factor 2.
        n_rep (int): Number of repetitions per cell.
        N (int): Total number of observations.
        grand_mean (float): Grand mean of all observations.
        SS_Total (float): Total sum of squares.
        SS_A (float): Sum of squares for Factor 1.
        SS_B (float): Sum of squares for Factor 2.
        SS_AB (float): Sum of squares for Interaction (Factor 1 x Factor 2).
        SS_Error (float): Sum of squares for Error.
        df_A (int): Degrees of freedom for Factor 1.
        df_B (int): Degrees of freedom for Factor 2.
        df_AB (int): Degrees of freedom for Interaction.
        df_Error (int): Degrees of freedom for Error.
        MS_A (float): Mean square for Factor 1.
        MS_B (float): Mean square for Factor 2.
        MS_AB (float): Mean square for Interaction.
        MS_Error (float): Mean square for Error.
        F_A (float): F-statistic for Factor 1.
        F_B (float): F-statistic for Factor 2.
        F_AB (float): F-statistic for Interaction.
        p_A (float): P-value for Factor 1.
        p_B (float): P-value for Factor 2.
        p_AB (float): P-value for Interaction.

    References:
        - https://en.wikipedia.org/wiki/Two-way_analysis_of_variance
    """

    def __init__(self):
        super().__init__()
        self.model_name = "Two-Way Analysis of Variance"
        self.method = "Two-Way ANOVA"
        self.data: Optional[pd.DataFrame] = None
        self.value_column: Optional[str] = None
        self.factor1_column: Optional[str] = None
        self.factor2_column: Optional[str] = None
        self.a: Optional[int] = None
        self.b: Optional[int] = None
        self.n_rep: Optional[int] = None
        self.N: Optional[int] = None
        self.grand_mean: Optional[float] = None
        self.SS_Total: Optional[float] = None
        self.SS_A: Optional[float] = None
        self.SS_B: Optional[float] = None
        self.SS_AB: Optional[float] = None
        self.SS_Error: Optional[float] = None
        self.df_A: Optional[int] = None
        self.df_B: Optional[int] = None
        self.df_AB: Optional[int] = None
        self.df_Error: Optional[int] = None
        self.MS_A: Optional[float] = None
        self.MS_B: Optional[float] = None
        self.MS_AB: Optional[float] = None
        self.MS_Error: Optional[float] = None
        self.F_A: Optional[float] = None
        self.F_B: Optional[float] = None
        self.F_AB: Optional[float] = None
        self.p_A: Optional[float] = None
        self.p_B: Optional[float] = None
        self.p_AB: Optional[float] = None

    def fit(self, data: pd.DataFrame, value_column: str, factor1_column: str, factor2_column: str):
        """
        Fits the Two-Way ANOVA model to the provided data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            value_column (str): The name of the column containing the numerical values.
            factor1_column (str): The name of the first factor column.
            factor2_column (str): The name of the second factor column.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        for col in [value_column, factor1_column, factor2_column]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")
        if not pd.api.types.is_numeric_dtype(data[value_column]):
            raise TypeError(f"Value column '{value_column}' must be numeric.")

        self.data = data.dropna(subset=[value_column, factor1_column, factor2_column]).copy()
        self.value_column = value_column
        self.factor1_column = factor1_column
        self.factor2_column = factor2_column

        # Get unique levels for each factor
        levels_A = self.data[factor1_column].unique()
        levels_B = self.data[factor2_column].unique()
        self.a = len(levels_A)
        self.b = len(levels_B)

        # Calculate repetitions per cell
        cell_counts = self.data.groupby([factor1_column, factor2_column])[value_column].count()
        if not np.all(cell_counts == cell_counts.iloc[0]):
            raise ValueError("Two-Way ANOVA requires a balanced design (equal repetitions per cell).")
        self.n_rep = cell_counts.iloc[0]

        if self.n_rep < 2:
            raise ValueError(
                "Two-Way ANOVA with interaction requires at least two repetitions per cell. "
                "For one repetition per cell, interaction cannot be calculated separately from error."
            )

        self.N = self.a * self.b * self.n_rep
        if self.N != len(self.data):
            raise ValueError("Calculated total observations do not match DataFrame length after dropping NaNs. Check data integrity.")

        # Calculate means
        self.grand_mean = self.data[value_column].mean()

        # Sum of Squares calculations
        # SS_Total (corrected)
        self.SS_Total = np.sum((self.data[value_column] - self.grand_mean) ** 2)

        # SS_A (Factor 1)
        mean_A_levels = self.data.groupby(factor1_column)[value_column].mean()
        self.SS_A = self.b * self.n_rep * np.sum((mean_A_levels - self.grand_mean) ** 2)

        # SS_B (Factor 2)
        mean_B_levels = self.data.groupby(factor2_column)[value_column].mean()
        self.SS_B = self.a * self.n_rep * np.sum((mean_B_levels - self.grand_mean) ** 2)

        # SS_AB (Interaction)
        mean_AB_cells = self.data.groupby([factor1_column, factor2_column])[value_column].mean()
        SS_cells = self.n_rep * np.sum((mean_AB_cells - self.grand_mean) ** 2)
        self.SS_AB = SS_cells - self.SS_A - self.SS_B

        # SS_Error
        self.SS_Error = np.sum(
            (self.data.groupby([factor1_column, factor2_column])[value_column].apply(
                lambda x: (x - x.mean()) ** 2
            )).explode()
        )
        
        # Verify SS_Total = SS_A + SS_B + SS_AB + SS_Error
        # This check is important for correctness
        calculated_total_ss = self.SS_A + self.SS_B + self.SS_AB + self.SS_Error
        if not np.isclose(self.SS_Total, calculated_total_ss):
            self.notes.append(f"Warning: Sum of squares do not add up. Discrepancy: {self.SS_Total - calculated_total_ss:.4e}")


        # Degrees of Freedom
        self.df_A = self.a - 1
        self.df_B = self.b - 1
        self.df_AB = (self.a - 1) * (self.b - 1)
        self.df_Error = self.a * self.b * (self.n_rep - 1)
        self.df_Total = self.N - 1

        # Mean Squares
        self.MS_A = self.SS_A / self.df_A
        self.MS_B = self.SS_B / self.df_B
        self.MS_AB = self.SS_AB / self.df_AB
        self.MS_Error = self.SS_Error / self.df_Error

        # F-statistics
        self.F_A = self.MS_A / self.MS_Error
        self.F_B = self.MS_B / self.MS_Error
        self.F_AB = self.MS_AB / self.MS_Error

        # P-values
        self.p_A = f.sf(self.F_A, self.df_A, self.df_Error)
        self.p_B = f.sf(self.F_B, self.df_B, self.df_Error)
        self.p_AB = f.sf(self.F_AB, self.df_AB, self.df_Error)

    def _get_summary_data(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing summary data for the Two-Way ANOVA model.
        """
        if self.F_A is None:
            return {}

        summary = self._create_general_summary(
            n_variables=1,  # Only one dependent variable
            n_objects=self.N,
            Factor1=self.factor1_column,
            Factor2=self.factor2_column,
            Levels_Factor1=self.a,
            Levels_Factor2=self.b,
            Repetitions_per_Cell=self.n_rep,
        )

        anova_table = [
            ["Source of Variation", "df", "Sum of Squares", "Mean Square", "F-value", "P-value"],
            [self.factor1_column, self.df_A, f"{self.SS_A:.3f}", f"{self.MS_A:.3f}", f"{self.F_A:.3f}", f"{self.p_A:.3f}"],
            [self.factor2_column, self.df_B, f"{self.SS_B:.3f}", f"{self.MS_B:.3f}", f"{self.F_B:.3f}", f"{self.p_B:.3f}"],
            ["Interaction", self.df_AB, f"{self.SS_AB:.3f}", f"{self.MS_AB:.3f}", f"{self.F_AB:.3f}", f"{self.p_AB:.3f}"],
            ["Residual", self.df_Error, f"{self.SS_Error:.3f}", f"{self.MS_Error:.3f}", "", ""],
            ["Total (Corrected)", self.df_Total, f"{self.SS_Total:.3f}", "", "", ""],
        ]
        summary["tables"] = {"ANOVA Table": anova_table}

        return summary


class MultiwayANOVA(BaseModel):
    """
    Placeholder for Multiway Analysis of Variance (ANOVA).

    Multiway ANOVA extends Two-Way ANOVA to analyze the effects of three or more
    independent categorical variables (factors) on a continuous dependent variable.
    It can also assess interaction effects among these factors.

    Full implementation from scratch is highly complex due to the combinatorial
    increase in interaction terms and requires a robust statistical framework.
    This class currently serves as a placeholder.
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Multiway Analysis of Variance"
        self.method = "Multiway ANOVA"

    def fit(self, data: pd.DataFrame, value_column: str, factor_columns: List[str]):
        """
        Fits the Multiway ANOVA model to the provided data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            value_column (str): The name of the column containing the numerical values.
            factor_columns (List[str]): A list of column names for the independent factors.
        """
        raise NotImplementedError(
            "Multiway ANOVA is a complex statistical method. "
            "A full implementation from scratch is not provided in this version. "
            "Consider using specialized statistical libraries like `statsmodels` or `scipy.stats` "
            "for multiway ANOVA functionality."
        )

    def _get_summary_data(self) -> Dict[str, Any]:
        return {"notes": ["Multiway ANOVA is not yet implemented in full."]}


class MANOVA(BaseModel):
    """
    Placeholder for Multivariate Analysis of Variance (MANOVA).

    MANOVA is an extension of ANOVA that analyzes the effects of independent
    categorical variables on two or more continuous dependent variables simultaneously.
    It tests for differences in group means across multiple dependent variables.

    Full implementation from scratch is highly complex, involving multivariate
    test statistics (e.g., Wilks' Lambda, Pillai's Trace) and their distributions.
    This class currently serves as a placeholder.
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Multivariate Analysis of Variance"
        self.method = "MANOVA"

    def fit(self, data: pd.DataFrame, value_columns: List[str], group_column: str):
        """
        Fits the MANOVA model to the provided data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            value_columns (List[str]): A list of column names for the numerical dependent variables.
            group_column (str): The name of the column containing the group labels.
        """
        raise NotImplementedError(
            "Multivariate Analysis of Variance (MANOVA) is a complex statistical method. "
            "A full implementation from scratch is not provided in this version. "
            "Consider using specialized statistical libraries like `statsmodels` or `scipy.stats` "
            "for MANOVA functionality."
        )

    def _get_summary_data(self) -> Dict[str, Any]:
        return {"notes": ["MANOVA is not yet implemented in full."]}
