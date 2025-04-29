# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a new coordinate system while preserving maximum variance.

## Overview

The PCA module in chemtools provides comprehensive functionality for:
- Data preprocessing and autoscaling
- Principal component calculation
- Variance analysis
- Score and loading computation
- Statistical validation

## Usage

```python
from chemtools.exploration import PrincipalComponentAnalysis
from chemtools.plots.exploration import PCAplots

# Create and fit the model
pca = PrincipalComponentAnalysis()
pca.fit(X, variables_names=variables, objects_names=objects)

# Initialize plots with desired settings
plots = PCAplots(pca, library="matplotlib", theme="light", style_preset="default")

# Generate various plots
plots.plot_correlation_matrix()
plots.plot_eigenvalues()
plots.plot_loadings()
plots.plot_biplot()
plots.plot_scores()
```

## Key Features

### 1. Model Fitting and Transformation
- Automatic data preprocessing
- Eigenvalue decomposition
- Dimensionality reduction
- Data projection

### 2. Component Selection Criteria
- Kaiser criterion (eigenvalues > 1)
- Percentage of explained variance
- Scree plot analysis
- Average eigenvalue criterion
- Cumulative percentage criterion

### 3. Statistical Analysis
- Hotelling's T² statistic
- Q residuals
- Confidence intervals
- Contribution plots

### 4. Visualization
- Correlation matrices
- Eigenvalue plots
- Score plots
- Loading plots
- Biplots

## API Reference

### PrincipalComponentAnalysis Class

```python
class PrincipalComponentAnalysis:
    def __init__(self)
    def fit(self, X, variables_names=None, objects_names=None)
    def reduction(self, n_components)
    def transform(self, X_new)
    def statistics(self, alpha=0.05)
```

### PCAplots Class

```python
class PCAplots:
    def plot_correlation_matrix()
    def plot_eigenvalues()
    def plot_loadings()
    def plot_biplot()
    def plot_scores()
    def plot_hotteling_t2_vs_q()
```

## Examples

See the `examples` directory for complete notebooks demonstrating PCA analysis workflows.
