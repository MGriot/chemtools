<div align="center">
  <img src="https://github.com/MGriot/chemtools/blob/main/doc/img/icon.png" alt="Icon">
</div>

# chemtools

**chemtools** is a Python library designed to simplify chemometric analyses. 

## Inspiration 

This project draws heavy inspiration from the following sources:

* **[Statsmodels](https://www.statsmodels.org/stable/index.html):** ror statistical calculations and modeling.
* **[Scikit-learn](https://scikit-learn.org/1.5/index.html):** utilized for machine learning functionalities.

It provides a user-friendly interface for performing various operations, including:

**1. Data Preprocessing:**
- Autoscaling
- Correlation matrix calculation
- Diagonalization of matrices
- Calculation of matrix mean and standard deviation
- Variance calculation

**2. Exploratory Data Analysis:**
- Principal Component Analysis (PCA) with advanced visualization
- Multiple Correspondence Analysis (MCA)
- Hierarchical Clustering with dendrograms

**3. Regression Analysis:**
- Linear Regression with comprehensive plotting tools:
  - Residuals analysis
  - Confidence and prediction bands
  - Regression diagnostics
  - Model evaluation plots

**4. Advanced Plotting System:**
- Unified plotting interface supporting both Matplotlib and Plotly
- Customizable themes (light/dark modes)
- Consistent styling across all visualizations
- Style presets (default, minimal, grid, presentation)
- Publication-ready plots

**5. Visualization Features:**
- PCA visualization suite:
  - Correlation matrices
  - Eigenvalue plots with multiple criteria
  - Loading plots and biplots
  - Score plots and class separation
- Dendrogram plotting
- Regression diagnostic plots
- Interactive plots with Plotly support

**6. Utility Functions:**
- Heatmap generation and annotation
- Array manipulation (e.g., converting arrays to columns, reordering)
- Variable type checking
- Random color generation
- Saving and loading models

## Package Structure

The library is organized into these main modules:

- **preprocessing:** Contains functions for data preprocessing
- **exploration:** Houses classes and methods for exploratory data analysis
- **regression:** Provides implementations of regression models
- **plots:** Implements the unified plotting system with themes
- **utility:** Includes various utility functions

## Plotting System

The library features a powerful plotting system that:
- Provides consistent styling across all visualizations
- Supports both Matplotlib and Plotly backends
- Includes light and dark themes
- Offers multiple style presets
- Ensures publication-quality output
- Handles automatic color management

## How to Use

Refer to the [documentation](doc/Documentation.md) for detailed information on how to use the library. 

## Examples

Check out the Jupyter notebooks (`.ipynb`) in the repository for practical examples of how to use **chemtools**. 

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.
