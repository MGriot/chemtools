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

## Theming

The `chemtools` plotting library provides a robust theming system to control the aesthetic appearance of your visualizations. Themes define comprehensive color palettes, font settings, and other visual properties, ensuring a consistent and professional look across all your plots.

### Available Themes

The following themes are available out-of-the-box:

-   **`light`**: A default light-colored theme suitable for most general-purpose plots.
-   **`dark`**: A default dark-colored theme, ideal for presentations or environments with low ambient light.
-   **`oceanic_slate_light`**: A custom light theme with a cool, oceanic color palette.
-   **`oceanic_slate_dark`**: A custom dark theme complementing the oceanic_slate_light theme.

### Using a Theme

You can select a theme by passing the `theme` parameter to any plotter class constructor.

```python
from chemtools.plots.basic.line import LinePlot

# Use the dark theme
plotter = LinePlot(theme="dark")
fig = plotter.plot(x=[1, 2, 3], y=[1, 4, 9])
```

To use a custom theme like `oceanic_slate_light`:

```python
# Use the Oceanic Slate light theme
plotter = LinePlot(theme="oceanic_slate_light")
```

### Creating a Custom Theme

You can create your own themes by adding a JSON file to the `chemtools/plots/themes/` directory.

1.  **Create a JSON file**: For example, `my_theme.json`.
2.  **Define your theme**: The JSON file should contain a dictionary where keys are your theme names and values define the colors and fonts. It's good practice to create a light and dark version of your theme, e.g., `mytheme_light` and `mytheme_dark`.

**Example `my_theme.json`:**
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

The plotting library will automatically discover and load any `.json` files in the themes directory. You can then use your custom theme in your code:

```python
plotter = LinePlot(theme="mytheme_light")
```

**Required Theme Keys:**

Your theme must define the following keys:

| Key                          | Description                                         |
| ---------------------------- | --------------------------------------------------- |
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


## Available Plot Types

The plotting library is organized into categories based on the plot's purpose. For more details on each, see the [Plot Types](plot_types.md) documentation.

1.  **Basic Plots**
    -   [Bar Chart](plot_types.md#bar-plot) (including stacked and grouped)
    -   [Line Plot](plot_types.md#line-plot) (including dot and area)
    -   [Pie Chart](plot_types.md#pie-plot) (including donut)

2.  **Distribution Plots**
    -   [Histogram](plot_types.md#histogram) (including density curve)
    -   [Box Plot](plot_types.md#box-plot)
    -   [Violin Plot](plot_types.md#violin-plot)
    -   [Beeswarm Plot](plot_types.md#beeswarm-plot)

3.  **Relationship Plots**
    -   [Scatter Plot](plot_types.md#scatter-plot) (2D, 3D, and bubble)
    -   [Heatmap](plot_types.md#heatmap)

4.  **Specialized Plots**
    -   [Parallel Coordinates](plot_types.md#parallel-coordinates-plot)
    -   [Funnel Chart](plot_types.md#funnel-chart)
    -   [Bullet Chart](plot_types.md#bullet-chart)
    -   [Dual-Axis Chart](plot_types.md#dual-axis-chart) (Matplotlib only)

5.  **Geographical Plots**
    -   [Choropleth Map](plot_types.md#map-plot) (Plotly recommended)
    -   [Geo Scatter Plot](plot_types.md#map-plot) (Plotly recommended)

6.  **Analysis-Specific Plots**
    -   The existing plots for PCA, FA, MCA, Regression, and Clustering are still available in their respective modules.

---

## Inspirational Resources

For further inspiration on data visualization and advanced plotting techniques, consider exploring these excellent resources:

-   [The Python Graph Gallery](https://python-graph-gallery.com/)
-   [Python Plot](https://pythonplot.com/)
-   [Beautiful Plots](https://beautifulplots.readthedocs.io/en/latest/index.html)