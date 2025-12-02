# Plotting System Overview

The chemtools plotting system provides a unified interface for creating visualizations using both Matplotlib and Plotly backends.

## Core Features

1.  **Multiple Backend Support**
    -   Matplotlib for static plots
    -   Plotly for interactive visualizations

2.  **Consistent Styling**
    -   Light and dark themes
    -   Predefined style presets
    -   Custom color palettes
    -   Publication-ready defaults

## Base Plotter Class

All plot classes inherit from `BasePlotter`, which handles the styling and backend selection.

```python
from chemtools.plots.base import BasePlotter

# BasePlotter is not typically used directly, but inherited from.
class MyCustomPlot(BasePlotter):
    def __init__(self, library="matplotlib", theme="light", style_preset="default"):
        super().__init__(library, theme, style_preset)

    def plot(self, ...):
        # Your plotting logic here
        ...
```

<h2>Theming</h2>

The <code>chemtools</code> plotting library provides a robust theming system to control the aesthetic appearance of your visualizations. Themes define comprehensive color palettes, font settings, and other visual properties, ensuring a consistent and professional look across all your plots.

<h3>Available Themes</h3>

The following themes are available out-of-the-box:

<ul>
    <li><b><code>light</code></b>: A default light-colored theme suitable for most general-purpose plots.</li>
    <li><b><code>dark</code></b>: A default dark-colored theme, ideal for presentations or environments with low ambient light.</li>
    <li><b><code>oceanic_slate_light</code></b>: A custom light theme with a cool, oceanic color palette.</li>
    <li><b><code>oceanic_slate_dark</code></b>: A custom dark theme complementing the oceanic_slate_light theme.</li>
</ul>

<h3>Using a Theme</h3>

You can select a theme by passing the <code>theme</code> parameter to any plotter class constructor.

```python
from chemtools.plots.basic.line import LinePlot

# Use the dark theme
plotter = LinePlot(theme="dark")
fig = plotter.plot(x=[1, 2, 3], y=[1, 4, 9])
```

To use a custom theme like <code>oceanic_slate_light</code>:

```python
# Use the Oceanic Slate light theme
plotter = LinePlot(theme="oceanic_slate_light")
```

<h3>Creating a Custom Theme</h3>

You can create your own themes by adding a JSON file to the <code>chemtools/plots/themes/</code> directory.

<ol>
    <li><b>Create a JSON file</b>: For example, <code>my_theme.json</code>.</li>
    <li><b>Define your theme</b>: The JSON file should contain a dictionary where keys are your theme names and values define the colors and fonts. It's good practice to create a light and dark version of your theme, e.g., <code>mytheme_light</code> and <code>mytheme_dark</code>.</li>
</ol>

<b>Example <code>my_theme.json</code>:</b>
```json
{
    "mytheme_light": {
        "bg_color": "#f0f0f0",
        "text_color": "#333333",
        "grid_color": "#cccccc",
        "detail_light_color": "#bbbbbb",
        "detail_medium_color": "#999999",
        "theme_color": "#005f73",
        "accent_color": "#ca6702",
        "category_color_scale": [
            "#0a9396",
            "#94d2bd",
            "#e9d8a6",
            "#ee9b00",
            "#bb3e03"
        ],
        "prediction_band": "#94d2bd",
        "confidence_band": "#e9d8a6",
        "annotation_fontsize_small": 9,
        "annotation_fontsize_medium": 11
    }
}
```

The plotting library will automatically discover and load any <code>.json</code> files in the themes directory. You can then use your custom theme in your code:

```python
plotter = LinePlot(theme="mytheme_light")
```

<b>Required Theme Keys:</b>

Your theme must define the following keys:

| Key                          | Description                                         |
| :--------------------------- | :-------------------------------------------------- |
| `bg_color`                   | Background color of the plot.                       |
| `text_color`                 | Color for text, labels, and ticks.                  |
| `grid_color`                 | Color of the grid lines.                            |
| `detail_light_color`         | Lighter color for details like legend borders.      |
| `detail_medium_color`        | Medium color for less prominent details.            |
| `theme_color`                | Primary color for plot elements.                    |
| `accent_color`               | Accent color for highlighting.                      |
| `category_color_scale`       | A list of colors for categorical data.              |
| `prediction_band`            | Color for prediction bands in regression plots.     |
| `confidence_band`            | Color for confidence bands in regression plots.     |
| `annotation_fontsize_small`  | Small font size for annotations.                    |
| `annotation_fontsize_medium` | Medium font size for annotations.                   |

<h3>Generated Colormaps from Palette</h3>

To make plotting continuous data with themed colors easier, <code>BasePlotter</code> automatically generates three <code>matplotlib.colors.LinearSegmentedColormap</code> objects from the <code>category_color_scale</code> defined in the current theme. These are available as attributes on any plotter instance.

<ul>
    <li><b><code>self.sequential_cmap</code></b>: A sequential colormap, created by sorting the palette colors by their perceived brightness (luminance). This is ideal for representing data that goes from low to high.</li>
    <li><b><code>self.diverging_cmap</code></b>: A diverging colormap, created by using the first two (usually most contrasting) colors of the palette and a neutral midpoint. This is useful for data that has a meaningful center point, like zero.</li>
    <li><b><code>self.raw_cmap</code></b>: A colormap created by simply interpolating the palette colors in their original order.</li>
</ul>

<b>Usage Example:</b>

You can pass these colormaps directly to any plot that accepts a <code>cmap</code> argument (typically in <code>matplotlib</code> plots).

```python
from chemtools.plots.relationship import HeatmapPlot
import pandas as pd
import numpy as np

# Sample correlation matrix
data = pd.DataFrame(np.random.rand(10, 10))

# Initialize a plotter
plotter = HeatmapPlot(theme="classic_professional_light")

# Use the automatically generated sequential colormap
fig = plotter.plot(data, annot=True, cmap=plotter.sequential_cmap)
fig.show()
```

---

<h2>Understanding Chemtools Plot Types</h2>

The <code>chemtools</code> plotting system offers a diverse range of visualization tools, categorized by their primary purpose. All plots leverage the <code>BasePlotter</code> for consistent theming and backend selection.

<h3>Categorical Plots</h3>

Designed for visualizing relationships and distributions within categorical datasets.

<ul>
    <li><b>Heatmap of Co-occurrences:</b> Visualizes the relationship between two categorical variables by counting their co-occurrences in a contingency table and displaying the frequencies as a heatmap.</li>
    <li><b>Mosaic Plot:</b> A graphical representation of a contingency table, where the area of each rectangle is proportional to the cell's frequency, ideal for visualizing relationships and proportions between categorical variables.</li>
</ul>

<h3>Basic Plots</h3>

Fundamental plot types for general data visualization.

<ul>
    <li><b>Bar Plot:</b> Uses rectangular bars to represent categorical data, with bar lengths proportional to values. Supports simple counts, grouped, and stacked configurations.</li>
    <li><b>Line Plot:</b> Visualizes data points connected by straight line segments, effective for showing trends over continuous intervals or time series. Also supports dot plots and area charts.</li>
    <li><b>Pie Plot:</b> Circular graphics illustrating numerical proportion, with arc lengths proportional to quantities. Can also create donut charts.</li>
    <li><b>Sunburst Plot:</b> Multi-layered pie charts for hierarchical data, showing part-to-whole relationships across categories and subcategories.</li>
</ul>

<h3>Distribution Plots</h3>

Focus on visualizing the distribution of numerical variables.

<ul>
    <li><b>Histogram & Density Plot:</b> Histograms group data into bins to show frequency, while density plots (KDE) provide smooth curves to represent distribution shapes.</li>
    <li><b>Box Plot:</b> Displays the five-number summary (min, Q1, median, Q3, max) of a dataset, effective for comparing distributions across groups.</li>
    <li><b>Violin Plot:</b> Combines a box plot with a kernel density estimate, offering a richer view of data distribution, central tendency, and spread. Enhanced with jittered points and statistical annotations.</li>
    <li><b>Beeswarm Plot:</b> Displays individual data points for a numerical variable, arranged to avoid overlap, revealing distribution and density.</li>
    <li><b>Raincloud Plot:</b> A comprehensive visualization combining a violin plot, jittered scatter plot, and box plot for a complete view of data distributions.</li>
    <li><b>Ridgeline Plot:</b> Overlaps density plots for multiple categories to visualize the distribution of a numerical variable across several groups in a "mountain range" effect.</li>
</ul>

<h3>Relationship Plots</h3>

Used for exploring the relationships between two or more variables.

<ul>
    <li><b>Scatter Plot:</b> A fundamental tool for visualizing the relationship between two or three numerical variables, identifying correlations, trends, and outliers. Supports 2D, 3D, and bubble charts.</li>
    <li><b>Heatmap:</b> Graphical representation of a data matrix where values are represented by colors, useful for correlation matrices or co-occurrence frequencies.</li>
    <li><b>Pair Plot:</b> Creates a matrix of plots visualizing pairwise relationships between multiple variables, excellent for exploratory data analysis, showing scatter plots, KDEs, and correlation coefficients.</li>
    <li><b>2D Density Plot:</b> Visualizes the distribution of data points using color to represent density, suitable for crowded scatter plots. Options include KDE, 2D histograms, and hexbin plots.</li>
    <li><b>Joint Plot (Marginal Plot):</b> Combines a 2D plot (e.g., scatter, 2D KDE) with 1D distribution plots (histograms, 1D KDEs) on its margins, offering a comprehensive view of variable relationships and individual distributions.</li>
</ul>

<h3>Specialized Plots</h3>

Unique plot types for specific analytical needs.

<ul>
    <li><b>Parallel Coordinates Plot:</b> Visualizes high-dimensional data, where each vertical axis is a variable and each line is an observation, revealing patterns and relationships across many variables.</li>
    <li><b>Funnel Chart:</b> Visualizes the progressive reduction of data through sequential stages, common in sales or process flows.</li>
    <li><b>Bullet Chart:</b> Compares a primary measure to a target measure within qualitative performance ranges, providing a compact view of performance.</li>
    <li><b>Dual-Axis Chart:</b> Displays two different variables with different scales on a single plot using two y-axes, useful for comparing related metrics (Matplotlib only).</li>
    <li><b>Radar Chart:</b> Plots values for multiple quantitative variables on axes radiating from a center, useful for comparing multivariate profiles.</li>
</ul>

<h3>Geographical Plots</h3>

Visualize data on maps.

<ul>
    <li><b>Choropleth Map:</b> Displays geographical areas colored according to a data variable, showing spatial variation.</li>
    <li><b>Geo Scatter Plot:</b> Renders points on a map based on latitude and longitude, indicating specific locations.</li>
</ul>

<h3>Analysis-Specific Plots</h3>

Plots tailored for specific multivariate analysis techniques.

<ul>
    <li><b>SIMCA Plot:</b> Visualizes results of Soft Independent Modeling of Class Analogies (SIMCA), plotting class-specific PCA scores with confidence ellipses.</li>
    <li><b>Dendrogram:</b> Tree-like diagram recording merges/splits in hierarchical clustering, showing cluster relationships.</li>
    <li><b>Regression Results Plot:</b> Comprehensive visualization of linear regression models, including data, regression line, confidence, and prediction bands.</li>
    <li><b>FAMD Plots (Scores & Loadings):</b> Visualizes results from Factor Analysis of Mixed Data, showing patterns among observations and variable contributions.</li>
    <li><b>MCA Plots (Eigenvalues & Objects):</b> Visualizes Multiple Correspondence Analysis results, including eigenvalue plots (scree plots) and object positions.</li>
    <li><b>PCA Plots (Eigenvalues, Scores, Loadings, Biplot, Hotelling's T² vs. Q, PCI Contribution):</b> A comprehensive suite for visualizing Principal Component Analysis results.</li>
</ul>

<hr>

<h2>Inspirational Resources</h2>

For further inspiration on data visualization and advanced plotting techniques, consider exploring these excellent resources:

<ul>
    <li><a href="https://python-graph-gallery.com/">The Python Graph Gallery</a></li>
    <li><a href="https://pythonplot.com/">Python Plot</a>
        <ul>
            <li><a href="https://beautifulplots.readthedocs.io/en/latest/index.html">Beautiful Plots</a></li>
            <li><a href="https://observablehq.com/@observablehq">ObservableHQ</a></li>
        </ul>
    </li>
</ul>