# Plotting System Overview

The chemtools plotting system provides a unified interface for creating visualizations using both Matplotlib and Plotly backends.

## Core Features

1. **Multiple Backend Support**
   - Matplotlib for static plots
   - Plotly for interactive visualizations

2. **Consistent Styling**
   - Light and dark themes
   - Predefined style presets
   - Custom color palettes
   - Publication-ready defaults

3. **Style Presets**
   - Default: Standard scientific plotting style
   - Minimal: Clean, minimalist design
   - Grid: Enhanced grid lines for data comparison
   - Presentation: Bold styles for presentations

## Base Plotter Class

```python
class Plotter:
    def __init__(self, library="matplotlib", theme="light", style_preset="default")
    def set_theme(self, theme)
    def set_style_preset(self, preset)
    def set_custom_colors(self, colors)
```

## Usage Example

```python
from chemtools.plots import Plotter

# Initialize with desired settings
plotter = Plotter(library="matplotlib",
                  theme="light",
                  style_preset="default")

# Change settings as needed
plotter.set_theme("dark")
plotter.set_style_preset("presentation")
```

## Available Plot Types

1. **Exploration Plots**
   - PCA plots (scores, loadings, biplots)
   - Factor Analysis plots
   - MCA plots

2. **Regression Plots**
   - Scatter plots
   - Residual plots
   - Q-Q plots
   - Confidence bands

3. **Clustering Plots**
   - Dendrograms
   - Heatmaps

4. **Statistical Plots**
   - Box plots
   - Violin plots
   - Distribution plots
