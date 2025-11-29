# Run Chart

A run chart is a line graph of data plotted over time. It is a simple yet powerful tool used to find trends or patterns in data, showing how a process or variable performs over a specific period.

## `plot`

This method generates the run chart.

### Usage
```python
from chemtools.plots.temporal import RunChartPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=50)),
    'Measurement': np.random.randn(50).cumsum() + 50
})


# Create Plot
plotter = RunChartPlot()
fig = plotter.plot(data, 
                   time_column='Date', 
                   value_column='Measurement', 
                   title="Run Chart of Measurement Over Time")
fig.savefig("run_chart.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `time_column` (str): The column representing the time or sequence for the x-axis.
- `value_column` (str): The column representing the numerical value to plot on the y-axis.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/temporal/run_chart_classic_professional_dark.png">
  <img alt="Run Chart" src="../../img/plots/temporal/run_chart_classic_professional_light.png">
</picture>
