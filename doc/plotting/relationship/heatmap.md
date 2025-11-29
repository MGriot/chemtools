# Heatmap

Heatmaps are graphical representations of data where the individual values contained in a matrix are represented as colors. They are particularly useful for visualizing correlation matrices and the co-occurrence of categorical variables. The `HeatmapPlot` class provides methods to create both numerical and categorical heatmaps.

## Numerical Heatmap

This is the standard heatmap used to visualize a matrix of numerical data, such as a correlation matrix.

### Usage
```python
from chemtools.plots.relationship import HeatmapPlot
import pandas as pd
import numpy as np

# Sample Data (e.g., a correlation matrix)
data = pd.DataFrame(np.random.rand(5, 5), 
                    columns=[f'Var{i+1}' for i in range(5)],
                    index=[f'Var{i+1}' for i in range(5)])

# Create Plot
plotter = HeatmapPlot(theme='classic_professional_light')
# 'annot=True' displays the numerical value in each cell
fig = plotter.plot(data, annot=True, title="Numerical Heatmap")
fig.savefig("heatmap_numerical.png")
```

### Parameters for `plot`
- `data` (pd.DataFrame): The numerical matrix to plot.
- `annot` (bool): If `True`, the data value for each cell is written on the heatmap.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter` or the underlying plotting function (`cmap` for matplotlib, `colorscale` for plotly).

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/heatmap_classic_professional_dark.png">
  <img alt="Numerical Heatmap" src="../../img/plots/relationship/heatmap_classic_professional_light.png">
</picture>

---

## Categorical Heatmap

This variation of the heatmap is used to visualize the relationship between two categorical variables. It first computes a contingency table (crosstab) of co-occurrence frequencies and then visualizes that table as a heatmap.

### Usage
```python
from chemtools.plots.relationship import HeatmapPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'Hair Color': np.random.choice(['Black', 'Brown', 'Blonde', 'Red'], 100),
    'Eye Color': np.random.choice(['Brown', 'Blue', 'Green'], 100),
})

# Create Plot
plotter = HeatmapPlot(theme='classic_professional_light')
fig = plotter.plot_categorical(data, 
                               x_column='Hair Color', 
                               y_column='Eye Color', 
                               annot=True, 
                               title="Categorical Heatmap (Co-occurrence)")
fig.savefig("heatmap_categorical.png")
```

### Parameters for `plot_categorical`
- `data` (pd.DataFrame): The input DataFrame containing the raw categorical data.
- `x_column` (str): The column to be used for the x-axis of the contingency table.
- `y_column` (str): The column to be used for the y-axis of the contingency table.
- `annot` (bool): If `True`, the frequency count is written on each cell.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/categorical/heatmap_categorical_classic_professional_dark.png">
  <img alt="Categorical Heatmap" src="../../img/plots/categorical/heatmap_categorical_classic_professional_light.png">
</picture>
