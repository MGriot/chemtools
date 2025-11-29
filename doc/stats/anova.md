# Analysis of Variance (ANOVA)

The `chemtools.stats.anova` module provides classes for performing various types of Analysis of Variance (ANOVA). ANOVA is a collection of statistical models used to analyze the differences among group means in a sample.

## Available Classes

### `OneWayANOVA`

Performs One-Way Analysis of Variance. This is used to determine whether there are any statistically significant differences between the means of two or more independent groups.

#### Usage

```python
import pandas as pd
from chemtools.stats.anova import OneWayANOVA

# Sample data
data = pd.DataFrame({
    'Value': [10, 12, 11, 15, 14, 18, 17, 20, 19],
    'Group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
})

# Create and fit the model
anova_model = OneWayANOVA()
anova_model.fit(data, value_column='Value', group_column='Group')

# Print the summary
print(anova_model.summary)
```

#### API Reference

```python
class OneWayANOVA(BaseModel):
    def __init__(self)
    def fit(self, data: pd.DataFrame, value_column: str, group_column: str)
    def _get_summary_data(self) -> Dict[str, Any]
```

### `TwoWayANOVA`

Performs Two-Way Analysis of Variance for a balanced design with repetitions. This is used to evaluate the effects of two independent categorical variables (factors) on a continuous dependent variable, and to assess their interaction effect.

#### Usage

```python
import pandas as pd
from chemtools.stats.anova import TwoWayANOVA

# Sample data (balanced design with repetitions)
data_two_way = pd.DataFrame({
    'Value': [10, 11, 15, 16, 18, 19, 22, 23],
    'Factor1': ['A1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A2'],
    'Factor2': ['B1', 'B1', 'B2', 'B2', 'B1', 'B1', 'B2', 'B2']
})

# Create and fit the model
anova_model_two_way = TwoWayANOVA()
anova_model_two_way.fit(data_two_way, value_column='Value', factor1_column='Factor1', factor2_column='Factor2')

# Print the summary
print(anova_model_two_way.summary)
```

#### API Reference

```python
class TwoWayANOVA(BaseModel):
    def __init__(self)
    def fit(self, data: pd.DataFrame, value_column: str, factor1_column: str, factor2_column: str)
    def _get_summary_data(self) -> Dict[str, Any]
```

### `MultiwayANOVA` (Placeholder)

Extends Two-Way ANOVA to analyze the effects of three or more independent categorical variables (factors) on a continuous dependent variable.

```python
class MultiwayANOVA(BaseModel):
    def __init__(self)
    def fit(self, data: pd.DataFrame, value_column: str, factor_columns: List[str])
    def _get_summary_data(self) -> Dict[str, Any]
```
**Note:** A full implementation from scratch is not provided in this version due to its complexity. Consider using specialized statistical libraries for multiway ANOVA functionality.

### `MANOVA` (Placeholder)

Multivariate Analysis of Variance (MANOVA) analyzes the effects of independent categorical variables on two or more continuous dependent variables simultaneously.

```python
class MANOVA(BaseModel):
    def __init__(self)
    def fit(self, data: pd.DataFrame, value_columns: List[str], group_column: str)
    def _get_summary_data(self) -> Dict[str, Any]
```
**Note:** A full implementation from scratch is not provided in this version due to its complexity. Consider using specialized statistical libraries for MANOVA functionality.
