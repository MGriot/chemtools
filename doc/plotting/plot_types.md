# Plot Types

This page details the available plot types, organized by category.

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
```python
from chemtools.plots.violin import ViolinPlot

# plotter = ViolinPlot()
# plotter.plot(data, y, x=None, **kwargs)
```
-   `y`: The numerical column to plot.
-   `x`: Optional categorical column for grouped plots.
-   This plot is an alternative to the box plot and shows the probability density of the data.

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
