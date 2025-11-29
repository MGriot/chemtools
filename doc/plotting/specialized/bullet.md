# Bullet Chart

A bullet chart is a variation of a bar chart designed to compare a primary measure (the "bullet") to a target measure, all within the context of qualitative ranges (e.g., poor, average, good). It is a compact and efficient way to display performance data.

## `plot`

This method generates the bullet chart.

### Usage
```python
from chemtools.plots.specialized import BulletPlot

# Data for the chart
current_value = 275
target_value = 250
performance_ranges = [150, 225, 300] # Represents poor, average, good thresholds

# Create Plot
plotter = BulletPlot()
fig = plotter.plot(value=current_value, 
                   target=target_value, 
                   ranges=performance_ranges, 
                   title="Revenue Q3 (in thousands)")
fig.savefig("bullet_chart.png")
```

### Parameters
- `value` (float): The main value (the "bullet" bar) to display.
- `target` (float): The target value, represented by a vertical line.
- `ranges` (list): A list of 3 numerical values for the qualitative background ranges (e.g., [poor, average, good]).
- `title` (str, optional): The title of the chart.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/specialized/bullet_plot_classic_professional_dark.png">
  <img alt="Bullet Chart" src="../../img/plots/specialized/bullet_plot_classic_professional_light.png">
</picture>
