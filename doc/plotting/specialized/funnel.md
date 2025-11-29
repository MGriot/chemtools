# Funnel Chart

A funnel chart is used to visualize the progressive reduction of data as it passes from one phase to another. It is commonly used to represent stages in a sales process, user engagement flow, or any other process with sequential steps. The width of each segment of the funnel represents the quantity or value at that stage.

## `plot`

This method generates the funnel chart.

### Usage
```python
from chemtools.plots.specialized import FunnelPlot
import pandas as pd

# Sample Data
# Data should be pre-sorted in the order of the funnel stages
data = pd.DataFrame({
    'Stage': ['Website Visits', 'Downloads', 'Registrations', 'Purchases'],
    'Value': [10000, 4000, 1500, 500]
})

# Create Plot
plotter = FunnelPlot()
fig = plotter.plot(data, stage_column='Stage', values_column='Value', title="Sales Funnel")
fig.savefig("funnel_chart.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame, which should be sorted in the order of the funnel stages.
- `stage_column` (str): The column containing the names of the funnel stages.
- `values_column` (str): The column containing the numerical values for each stage.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/specialized/funnel_plot_classic_professional_dark.png">
  <img alt="Funnel Chart" src="../../img/plots/specialized/funnel_plot_classic_professional_light.png">
</picture>
