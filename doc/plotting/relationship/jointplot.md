# Joint Plot (Marginal Plot)

A joint plot, also known as a marginal plot, is a powerful visualization that combines a 2D plot (e.g., scatter plot or 2D KDE) with 1D distribution plots (histograms or KDEs) along its margins. This allows for simultaneous inspection of the relationship between two variables and their individual distributions.

The `JointPlot` class provides a flexible way to create these combined visualizations.

## `plot`

This method generates a joint plot.

### Usage
```python
from chemtools.plots.relationship import JointPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'var_a': np.random.normal(0, 1, 300),
    'var_b': np.random.normal(5, 2, 300) + np.random.normal(0, 0.5, 300) * np.random.normal(0, 1, 300),
    'group': np.random.choice(['Group X', 'Group Y'], 300)
})

# Create Plotter
plotter = JointPlot(theme='classic_professional_light', figsize=(8, 8))

# Example 1: Scatter plot with marginal histograms
fig1 = plotter.plot(data, 
                    x='var_a', y='var_b', 
                    central_kind='scatter', marginal_kind='hist',
                    central_kwargs={'color': 'darkblue', 'alpha': 0.7},
                    marginal_kwargs={'color': 'skyblue', 'bins': 20},
                    title="Joint Plot: Scatter with Histograms")
fig1.savefig("joint_scatter_hist.png")

# Example 2: 2D KDE with marginal 1D KDEs
fig2 = plotter.plot(data, 
                    x='var_a', y='var_b', 
                    central_kind='kde2d', marginal_kind='kde1d',
                    central_kwargs={'cmap': 'Blues', 'levels': 10},
                    marginal_kwargs={'color': 'red', 'lw': 1.5},
                    title="Joint Plot: 2D KDE with 1D KDEs")
fig2.savefig("joint_kde_kde.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x` (str): The column name for the x-axis (main variable).
- `y` (str): The column name for the y-axis (main variable).
- `central_kind` (str): Type of central plot. Options: `'scatter'`, `'kde2d'`. Defaults to `'scatter'`.
- `marginal_kind` (str): Type of marginal plot. Options: `'hist'`, `'kde1d'`. Defaults to `'hist'`.
- `central_kwargs` (dict, optional): Additional keyword arguments passed to the central plot function (e.g., `s`, `alpha`, `color` for scatter; `cmap`, `levels` for kde2d).
- `marginal_kwargs` (dict, optional): Additional keyword arguments passed to the marginal plot functions (e.g., `bins`, `alpha`, `color` for histograms; `color`, `lw` for kde1d).
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter` (e.g., `figsize`, `title`, `xlabel`, `ylabel`).

### Example Output (Scatter with Histograms)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/joint_scatter_hist_classic_professional_dark.png">
  <img alt="Joint Plot Scatter with Histograms" src="../../img/plots/relationship/joint_scatter_hist_classic_professional_light.png">
</picture>

### Example Output (2D KDE with 1D KDEs)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/joint_kde_kde_classic_professional_dark.png">
  <img alt="Joint Plot 2D KDE with 1D KDEs" src="../../img/plots/relationship/joint_kde_kde_classic_professional_light.png">
</picture>
