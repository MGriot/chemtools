# Plot Types

## Exploration Plots

### PCA Plots
```python
class PCAplots(Plotter):
    def plot_correlation_matrix()
    def plot_eigenvalues()
    def plot_loadings()
    def plot_biplot()
    def plot_scores()
    def plot_hotteling_t2_vs_q()
```

### Factor Analysis Plots
```python
class FAplots(Plotter):
    def plot_loadings()
    def plot_scores()
    def plot_communalities()
```

### MCA Plots
```python
class MCAplots(Plotter):
    def plot_correspondence_matrix()
    def plot_row_profiles()
    def plot_column_profiles()
```

## Regression Plots

### Linear Regression Plots
```python
class RegressionPlots(Plotter):
    def plot_scatter()
    def plot_residuals()
    def plot_qq()
    def plot_confidence_bands()
    def plot_prediction_bands()
```

## Clustering Plots

### Dendrogram
```python
class DendrogramPlotter(Plotter):
    def plot_dendrogram(self, model, orientation="top", color_threshold=None)
```

### Heatmap
```python
class HeatmapPlotter(Plotter):
    def plot_heatmap(self, data, row_labels=None, col_labels=None)
```

## Statistical Plots

### Distribution Plots
```python
class StatisticalPlots(Plotter):
    def plot_box()
    def plot_violin()
    def plot_histogram()
    def plot_density()
```
