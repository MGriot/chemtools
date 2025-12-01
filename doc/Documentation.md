# Chemtools Documentation

Welcome to the Chemtools documentation. This document provides an overview of the library's structure and how to use it.

For a high-level overview of the project's goals and core architectural principles, see the [Project Overview](project_overview.md).

## Structure

The library is organized into several modules:

### 1. Data Preprocessing
An overview of all preprocessing functions can be found here:
- [Preprocessing Functions](preprocessing/index.md)

This includes:
- Data autoscaling and Pareto scaling
- Row-wise normalization
- Baseline correction
- Correlation matrix calculation
- And more...

### 2. Exploration
Tools for exploratory data analysis:
- [Principal Component Analysis (PCA)](exploration/principal-component-analysis.md)
- [Extended Principal Component Analysis (XPCA)](exploration/extended_pca.md)
- [Multiple Correspondence Analysis (MCA)](exploration/mca.md)
- [Factor Analysis (FA)](exploration/factor_analysis.md)
- [Factor Analysis for Mixed Data (FAMD)](exploration/famd.md)
- [Exploratory Data Analysis (EDA)](exploration/eda.md)

### 3. Regression
Various regression analysis techniques:
- [Linear Regression](regression/Linear%20Regression.md)

### 4. Classification
Classification methods:
- [Linear Discriminant Analysis (LDA)](classification/lda.md)
- [k-Nearest Neighbors (k-NN)](classification/knn.md)
- [Soft Independent Modeling of Class Analogies (SIMCA)](classification/simca.md)
- [Principal Component Classification Analysis (PCCA)](classification/pcca.md)

### 5. Clustering
Clustering algorithms:
- [k-Means Clustering](clustering/kmeans.md)
- [Hierarchical Clustering](clustering/hierarchical_clustering.md)

### 6. Statistical Functions
Core statistical calculations and models:
- [Univariate Statistics](stats/univariate_stats.md)
- [Regression Statistics](stats/regression_stats.md)
- [Analysis of Variance (ANOVA)](stats/anova.md)

### 7. Semantic Modeling
Tools for building and interacting with semantic data models:
- [Hybrid Semantic Model](semantic/semantic_model.md)

### 8. Plotting System
Unified plotting interface:
- [Plotting Overview](plotting/overview.md)
- [Plot Styles and Themes](plotting/styles_and_themes.md)
- [Plot Types](plotting/plot_types.md)
- [Categorical Plots](plotting/categorical_plots.md)
- [Classification Plots](plotting/classification_plots.md)

## Usage Examples and Workflows

For information on how to set up your development environment and contribute to the project, please refer to the [Development Guide](development_guide.md).

### Analysis Workflows
- **[XRF Data Analysis Workflow](workflows/xrf_analysis.md)**: A complete guide to analyzing X-Ray Fluorescence data using PCA and SIMCA.

### Scripted Examples
The `examples` directory contains Python scripts demonstrating the library's functionality:
- `run_xrf_analysis_example.py`: Demonstrates PCA and SIMCA on synthetic XRF data.
- `run_simca_example.py`: A focused example of the SIMCA classifier.

## Contributing

We welcome contributions! Please see the [CONTRIBUTING.md](../CONTRIBUTING.md) file for guidelines.
