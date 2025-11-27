import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from ..base.base_models import BaseModel

class ChiSquaredTest(BaseModel):
    """
    Performs a Chi-Squared (χ²) test of independence for two categorical variables.

    This test is used to determine if there is a significant association between
    the two variables.

    Attributes:
        chi2 (float): The Chi-Squared test statistic.
        p_value (float): The p-value of the test.
        dof (int): The degrees of freedom.
        expected_freq (np.ndarray): The expected frequencies, based on the null hypothesis of independence.
        cramers_v (float): Cramér's V, a measure of association between the two variables (0 to 1).
        contingency_table (pd.DataFrame): The contingency table (crosstab) of the observed frequencies.
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Chi-Squared Test"
        self.method = "χ² test of independence"
        self.chi2 = None
        self.p_value = None
        self.dof = None
        self.expected_freq = None
        self.cramers_v = None
        self.contingency_table = None

    def fit(self, x: pd.Series = None, y: pd.Series = None, contingency_table: pd.DataFrame = None):
        """
        Fits the model by performing the Chi-Squared test.

        The method can be run in two ways:
        1. By providing two pandas Series `x` and `y` containing the categorical data.
        2. By providing a pre-computed `contingency_table`.

        Args:
            x (pd.Series, optional): The first categorical variable.
            y (pd.Series, optional): The second categorical variable.
            contingency_table (pd.DataFrame, optional): A pre-computed contingency table.
        """
        if contingency_table is not None:
            if not isinstance(contingency_table, pd.DataFrame):
                raise TypeError("contingency_table must be a pandas DataFrame.")
            self.contingency_table = contingency_table
        elif x is not None and y is not None:
            if not all(isinstance(i, pd.Series) for i in [x, y]):
                raise TypeError("x and y must be pandas Series.")
            self.contingency_table = pd.crosstab(x, y)
        else:
            raise ValueError("You must provide either x and y Series or a contingency_table DataFrame.")

        if self.contingency_table.min().min() < 5:
            self.notes.append("Warning: Some cells in the contingency table have expected frequencies below 5. "
                              "The Chi-Squared test may not be accurate.")

        self.chi2, self.p_value, self.dof, self.expected_freq = chi2_contingency(self.contingency_table)
        self._calculate_cramers_v()
        
        return self

    def _calculate_cramers_v(self):
        """
        Calculates Cramér's V, a measure of association.
        """
        n = self.contingency_table.sum().sum()
        phi2 = self.chi2 / n
        r, k = self.contingency_table.shape
        self.cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))

    def _get_summary_data(self) -> dict:
        """
        Generates the summary data dictionary for the model.
        """
        # Convert DataFrame to a dictionary of dictionaries for BaseModel's table formatting
        contingency_table_dict = self.contingency_table.to_dict(orient='index')
        
        summary_data = {
            "general_info": {
                "Model": self.model_name,
                "Method": self.method,
            },
            "results": {
                "Chi-Squared (χ²)": f"{self.chi2:.4f}",
                "P-value": f"{self.p_value:.4f}",
                "Degrees of Freedom": self.dof,
            },
            "association": {
                "Cramér's V": f"{self.cramers_v:.4f}",
            },
            "tables": {
                "Contingency Table (Observed)": contingency_table_dict,
            },
            "notes": self.notes
        }
        return summary_data