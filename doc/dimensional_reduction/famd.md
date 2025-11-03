# Factor Analysis of Mixed Data (FAMD)

Factor Analysis of Mixed Data (FAMD) is a principal component method that allows for the analysis of datasets containing both quantitative (numerical) and qualitative (categorical) variables. It is a powerful technique for exploring the underlying structure of mixed data, as it balances the influence of both types of variables in the analysis.

This method is particularly useful in fields like survey analysis, sensory analysis, and ecology, where datasets often consist of a mix of measurements, counts, and categorical information.

## How FAMD Works

FAMD works by performing a weighted Principal Component Analysis (PCA) on the combined dataset. The key steps are:

1.  **Preprocessing Quantitative Variables**: The quantitative variables are standardized (centered and scaled to unit variance).
2.  **Preprocessing Qualitative Variables**: The qualitative variables are transformed into a disjunctive table (one-hot encoding). This indicator matrix is then scaled using a method similar to Multiple Correspondence Analysis (MCA) to ensure that the influence of each qualitative variable is balanced.
3.  **Concatenation**: The preprocessed quantitative and qualitative data are concatenated into a single matrix.
4.  **Global PCA**: A PCA is performed on the final concatenated and weighted matrix. The resulting principal components, scores, and loadings reveal the relationships between variables and the structure of the individuals (observations).

## Usage

```python
import pandas as pd
from chemtools.dimensional_reduction import FactorAnalysisOfMixedData
from chemtools.plots.dimensional_reduction import DimensionalityReductionPlot

# Create a sample DataFrame with mixed data
data = pd.DataFrame({
    'quant1': [1.2, 2.3, 3.4, 4.5, 5.6],
    'quant2': [10.1, 9.0, 8.9, 7.8, 6.7],
    'qual1': ['A', 'A', 'B', 'B', 'A'],
    'qual2': ['X', 'Y', 'X', 'Y', 'X']
})

# Specify which variables are qualitative
qualitative_vars = ['qual1', 'qual2']

# Create and fit the FAMD model
famd = FactorAnalysisOfMixedData(n_components=2)
famd.fit(data, qualitative_variables=qualitative_vars)

# Print the summary
print(famd.summary)

# Initialize the plotter
plotter = DimensionalityReductionPlot(famd)

# Generate plots
fig_scores = plotter.plot_scores()
fig_loadings = plotter.plot_loadings()

fig_scores.show()
fig_loadings.show()
```

## API Reference

### `FactorAnalysisOfMixedData` Class

```python
class FactorAnalysisOfMixedData(DimensionalityReduction):
    def __init__(self, n_components: int = 2)
    def fit(self, X: pd.DataFrame, qualitative_variables: list)
    def transform(self, X_new: pd.DataFrame) -> np.ndarray
```

### Citing Sources

This implementation is based on the principles of FAMD as described in the following publications:

- Pagès, J. (2004). Analyse factorielle de données mixtes. *Revue de Statistique Appliquée*, 52(4), 93-111.
- Escofier, B., & Pagès, J. (1994). Multiple factor analysis (AFMULT). *Computational Statistics & Data Analysis*, 18(1), 121-140.

Further reading can be found on Wikipedia:
- [Factor analysis of mixed data](https://en.wikipedia.org/wiki/Factor_analysis_of_mixed_data)
