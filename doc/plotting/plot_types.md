# Plot Types

This document provides a detailed overview of the various plot types available in the `chemtools` plotting system. Each plot type is organized by category and includes a brief description, a code snippet for usage, and a list of its specific parameters. All plots leverage the `BasePlotter` for consistent theming and backend selection.

## Basic Plots

Located in `chemtools.plots.basic`.

### Bar Plot
```python
from chemtools.plots.basic.bar import BarPlot

# plotter = BarPlot()
# For value counts of a single category:
# plotter.plot_counts(data, column, **kwargs)
# For plotting y vs x, with optional grouping/stacking:
# plotter.plot(data, x, y, color=None, mode='group', **kwargs)
# For plotting a pre-computed crosstab dataframe:
# plotter.plot_crosstab(crosstab_df, stacked=True, **kwargs)
```
-   `plot_counts`: Plots frequency of a single categorical column.
-   `plot`: Plots `y` vs `x` from long-format data. `mode` can be `'group'` or `'stack'`.
-   `plot_crosstab`: Plots a wide-format (crosstab) DataFrame directly. `stacked` can be `True` or `False`.

### Line Plot
```python
from chemtools.plots.basic.line import LinePlot

# plotter = LinePlot()
# plotter.plot(data, x_column, y_column, mode='line', **kwargs)
```
-   `mode`: Can be `'line'`, `'dot'`, or `'area'`.

### Pie Plot
```python
from chemtools.plots.basic.pie import PiePlot

# plotter = PiePlot()
# plotter.plot(data, names_column, values_column, hole=0, **kwargs)
```
-   `hole`: Creates a donut chart if > 0.

## Distribution Plots

Located in `chemtools.plots.distribution`.

### Histogram
```python
from chemtools.plots.distribution.histogram import HistogramPlot

# plotter = HistogramPlot()
# plotter.plot(data, column, mode='hist', **kwargs)
```
-   `mode`: Can be `'hist'` or `'density'`.

### Box Plot
```python
from chemtools.plots.distribution.boxplot import BoxPlot

# plotter = BoxPlot()
# plotter.plot(data, y, x=None, **kwargs)
```
-   `y`: The numerical column to plot.
-   `x`: Optional categorical column for grouped plots.

### Violin Plot

The violin plot is a powerful visualization that combines aspects of a box plot with a kernel density estimate. It shows the distribution of quantitative data across several categories, providing a richer understanding of the data's shape, central tendency, and spread than a simple box plot. Our enhanced violin plot also supports "raincloud" visualizations and automatic statistical annotations.

```python
from chemtools.plots.violin import ViolinPlot

# plotter = ViolinPlot()
# plotter.plot(data, y, x=None, show_jitter=False, show_mean=False, show_n=False, perform_stat_test=False, **kwargs)
```
-   `y`: The numerical column to plot.
-   `x`: Optional categorical column for grouped plots.
-   `show_jitter` (bool): If `True`, adds a "raincloud" of jittered data points, showing individual data points alongside the distribution.
-   `show_mean` (bool): If `True`, adds a distinct marker for the mean of each category, providing a clear indication of central tendency.
-   `show_n` (bool): If `True`, shows the sample size for each category on the x-axis labels, giving context to the distribution.
-   `perform_stat_test` (bool): If `True`, automatically performs pairwise t-tests between categories and annotates statistically significant results directly on the plot (matplotlib only).
-   `stat_annotations` (list): A list of dictionaries for manually plotting statistical comparisons (matplotlib only). This overrides `perform_stat_test` if provided.
-   `y_max_override` (float, optional): Manually sets the upper y-axis limit to ensure all annotations are comfortably visible.
-   `violin_alpha` (float, optional): Transparency (alpha) for the violin plot bodies (0.0 to 1.0). Overrides automatic adjustment.
-   `jitter_alpha` (float, optional): Transparency (alpha) for the jittered data points (0.0 to 1.0). Overrides automatic adjustment.

## Relationship Plots

Located in `chemtools.plots.relationship`.

### Scatter Plot
```python
from chemtools.plots.relationship.scatterplot import ScatterPlot

# plotter = ScatterPlot()
# plotter.plot_2d(data, x_column, y_column, size_column=None, **kwargs)
# plotter.plot_3d(data, x_column, y_column, z_column, **kwargs)
```
-   `size_column`: Creates a bubble chart if provided.

### Heatmap
```python
from chemtools.plots.relationship.heatmap import HeatmapPlot

# plotter = HeatmapPlot()
# plotter.plot(data, **kwargs)
```
-   `data`: pandas DataFrame representing the matrix to plot.

## Specialized Plots

Located in `chemtools.plots.specialized`.

### Parallel Coordinates Plot
```python
from chemtools.plots.specialized.parallel_coordinates import ParallelCoordinatesPlot

# plotter = ParallelCoordinatesPlot()
# plotter.plot(data, class_column, **kwargs)
```
-   `class_column`: Column to color the lines by.

### Funnel Chart
```python
from chemtools.plots.specialized.funnel import FunnelPlot

# plotter = FunnelPlot()
# plotter.plot(data, stage_column, values_column, **kwargs)
```
-   `data`: Should be sorted by funnel stage.

### Bullet Chart
```python
from chemtools.plots.specialized.bullet import BulletPlot

# plotter = BulletPlot()
# plotter.plot(value, target, ranges, title, **kwargs)
```
-   `value`, `target`, `ranges`: Numerical inputs for the chart.

### Dual-Axis Chart
```python
from chemtools.plots.specialized.dual_axis import DualAxisPlot

# plotter = DualAxisPlot() # Matplotlib only
# plotter.plot(data, x_column, y1_column, y2_column, plot1_kind='bar', plot2_kind='line', **kwargs)
```
-   **Note:** Matplotlib library only.

## Geographical Plots

Located in `chemtools.plots.geographical`.

### Map Plot
```python
from chemtools.plots.geographical.map import MapPlot

# plotter = MapPlot(library='plotly')
# To plot colored regions:
# plotter.plot_choropleth(data, locations_column, values_column, **kwargs)
# To plot points on a map:
# plotter.plot_scatter_geo(data, lat_column, lon_column, **kwargs)
```
-   **Note:** Plotly library is recommended for these plots.
