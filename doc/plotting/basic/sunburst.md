# Sunburst Plot

A sunburst plot is a multi-layered pie chart used to visualize hierarchical data. Each layer of the sunburst represents a level in the hierarchy, with the innermost circle being the highest level. The size of each segment in a layer is proportional to its value, and the segments radiate outwards from the center. This plot type is effective for showing part-to-whole relationships across multiple categories and subcategories.

The `SunburstPlot` class is designed to generate such charts, providing options for customization including handling of 'Other' categories and transparency for status indication.

## `plot`

This method generates a multi-layered sunburst chart.

### Usage
```python
import pandas as pd
import numpy as np
from chemtools.plots.basic import SunburstPlot

# Sample Data
np.random.seed(42)
data = {
    'Market': np.random.choice(['North America', 'Europe', 'Asia', 'Other Market'], 200),
    'Meta-Supplier': np.random.choice(['Supplier A', 'Supplier B', 'Supplier C', 'Other Supplier'], 200),
    'Material': [f'Mat{i}' for i in range(200)],
    'Status': np.random.choice(['Ok', 'Not Ok'], 200, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Create Plotter
plotter = SunburstPlot()

# Generate the sunburst plot
fig = plotter.plot(
    df,
    cols=['Market', 'Meta-Supplier'],
    count_col='Material',
    status_col='Status',
    status_ok_val='Ok',
    title="Material Status Breakdown by Market and Supplier"
)
fig.savefig("sunburst_plot.png")
```

### Parameters
- `df` (pd.DataFrame): The input DataFrame containing the hierarchical and status data.
- `cols` (List[str]): A list of 2 column names representing the hierarchy (e.g., `['Layer1', 'Layer2']`).
- `count_col` (str): The column whose unique values will be counted for segment sizes (e.g., `'Material'`).
- `status_col` (str): The column used to determine segment transparency (e.g., `'Status'`).
- `status_ok_val` (str): The value in `status_col` that represents "solid" or "robust" status, resulting in full opacity.
- `start_angle` (int): The starting angle for the first segment in degrees. Defaults to `90`.
- `top_n_limits` (Tuple[int, int]): A tuple specifying `(Top N for Layer 1, Top N for Layer 2)`. Smaller categories beyond this limit will be grouped into an 'Other' category. Defaults to `(6, 10)`.
- `label_color` (str): The color of the text labels for outside annotations. Defaults to `'black'`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter` (e.g., `figsize`, `subplot_title`, `title`).

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/basic/sunburst_plot_classic_professional_dark.png">
  <img alt="Sunburst Plot" src="../../img/plots/basic/sunburst_plot_classic_professional_light.png">
</picture>
