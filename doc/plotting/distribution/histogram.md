# Histogram and Density Plot

Histograms and density plots are fundamental tools for visualizing the distribution of a single numerical variable.

- A **Histogram** groups numbers into ranges (bins) and the height of the bar shows the frequency of data points falling into that range.
- A **Density Plot** (or Kernel Density Estimate - KDE) creates a smooth curve to represent the distribution, which can be useful for observing the shape of the distribution more clearly than with a histogram.

The `HistogramPlot` class can generate both types of plots.

## `plot`

This method creates either a histogram or a density plot.

### Usage
```python
from chemtools.plots.distribution import HistogramPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({'Value': np.random.randn(500) * 5 + 10})

plotter = HistogramPlot()

# Histogram
fig_hist = plotter.plot(data, column='Value', mode='hist', bins=20, title="Histogram")
fig_hist.savefig("histogram.png")

# Density Plot
fig_density = plotter.plot(data, column='Value', mode='density', title="Density Plot")
fig_density.savefig("density_plot.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `column` (str): The name of the numerical column to plot.
- `mode` (str): `'hist'` for a histogram or `'density'` for a density plot. Defaults to `'hist'`.
- `bins` (int): The number of bins to use for the histogram (only applicable when `mode='hist'`). Defaults to `10`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output (Histogram)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/histogram_classic_professional_dark.png">
  <img alt="Histogram" src="../../img/plots/distribution/histogram_classic_professional_light.png">
</picture>

### Example Output (Density)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/density_curve_classic_professional_dark.png">
  <img alt="Density Plot" src="../../img/plots/distribution/density_curve_classic_professional_light.png">
</picture>
