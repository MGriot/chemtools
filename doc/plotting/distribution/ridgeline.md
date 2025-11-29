# Ridgeline Plot

A ridgeline plot (also known as a joyplot) is a powerful visualization for showing the distribution of a numerical variable across several categories. By overlapping the density plots for each category, it creates a "mountain range" effect that is both beautiful and effective for comparing distributions.

The `RidgelinePlot` class in `chemtools` provides two distinct methods for creating these plots, both built on `matplotlib`.

## `plot()` - Simple Ridgeline

This method creates a clean ridgeline plot where each category's density is on a separate, overlapping axis.

### Usage
```python
from chemtools.plots.distribution import RidgelinePlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'Value': np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(3, 1.5, 100),
        np.random.normal(-2, 0.8, 100)
    ]),
    'Category': ['Group A'] * 100 + ['Group B'] * 100 + ['Group C'] * 100
})

# Create Plot
plotter = RidgelinePlot(theme='classic_professional_light')
fig = plotter.plot(data, x='Value', y='Category', overlap=0.5, title="Simple Ridgeline Plot")
fig.savefig("ridgeline_simple.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x` (str): The column name for the numerical variable.
- `y` (str): The column name for the categorical variable that defines the rows.
- `overlap` (float): The amount of vertical overlap between plots. A value of 0 means no overlap, while 1 means plots are fully on top of each other. Defaults to `0.5`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/ridgeline_simple_classic_professional_dark.png">
  <img alt="Simple Ridgeline Plot" src="../../img/plots/distribution/ridgeline_simple_classic_professional_light.png">
</picture>

---

## `plot_annotated()` - Detailed Ridgeline

This method provides extensive customization for creating highly detailed and publication-ready ridgeline plots, including quantile shading, mean value markers, and a descriptive inset legend.

### Usage
```python
from chemtools.plots.distribution import RidgelinePlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'Price': np.concatenate([
        np.random.gamma(4, 500, 100),
        np.random.gamma(6, 400, 100),
        np.random.gamma(5, 600, 100)
    ]),
    'Adjective': ['Nice'] * 100 + ['Spacious'] * 100 + ['Clean'] * 100
})

# Annotations to add to the plot
annotations = {
    'title': "Adjectives vs. Rental Prices",
    'description': "Distribution of rental prices based on descriptive adjectives found in listings.",
    'xlabel': "Rent in USD",
    'credit': "Data: Fictional rental listings."
}

# Create Plot
plotter = RidgelinePlot(theme='classic_professional_light', figsize=(8, 6))
fig = plotter.plot_annotated(
    data, 
    x='Price', 
    y='Adjective',
    annotations=annotations,
    show_legend=True
)
fig.savefig("ridgeline_annotated.png", bbox_inches='tight')
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x` (str): The column for the numerical variable.
- `y` (str): The column for the categorical variable.
- `bandwidth` (float): Controls the smoothness of the KDE plot. Defaults to `1.0`.
- `show_mean_line` (bool): If `True`, shows a global mean line and individual mean points. Defaults to `True`.
- `show_quantiles` (bool): If `True`, shades quantile regions on each distribution. Defaults to `True`.
- `annotations` (dict): A dictionary for custom text annotations. Keys can include `title`, `description`, `xlabel`, and `credit`.
- `show_legend` (bool): If `True`, displays the detailed inset legend. Defaults to `True`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/ridgeline_annotated_classic_professional_dark.png">
  <img alt="Annotated Ridgeline Plot" src="../../img/plots/distribution/ridgeline_annotated_classic_professional_light.png">
</picture>
