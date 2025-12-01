# XRF Data Analysis Workflow with `chemtools`

This document provides a guide for using the `chemtools` library to perform common chemometric analyses on X-Ray Fluorescence (XRF) data, including exploratory analysis with Principal Component Analysis (PCA) and classification with Soft Independent Modeling of Class Analogies (SIMCA).

## 1. Data Preparation

The first step in any analysis is to arrange your data into a 2D matrix format where rows represent samples (or pixels) and columns represent variables.

### Point Analysis Data

For standard point-and-shoot XRF, where each measurement corresponds to a distinct sample, the data matrix should be arranged as follows:

*   **Rows**: Each row is a single sample (e.g., `Rock_01`, `Coin_05`).
*   **Columns**: Each column is a measured variable, such as the peak area for a specific element (e.g., `Fe`, `Cu`, `Zn`).

### Imaging (MA-XRF) and Full Spectrum Data

For imaging or full-spectrum data, the initial data is often a 3D hypercube (`height x width x energy_channels`). This must be "unfolded" into a 2D matrix.

The `chemtools` library provides a utility for this:
*   **`chemtools.utils.unfold_hypercube(cube)`**: Converts a `(h, w, d)` hypercube into a `(h*w, d)` matrix.
*   **`chemtools.utils.refold_hypercube(matrix, h, w)`**: Converts the matrix back to a hypercube, useful for visualizing score maps.

## 2. Preprocessing

Raw XRF data often requires preprocessing to remove unwanted variations and prepare it for multivariate analysis. `chemtools` offers several key functions in the `chemtools.preprocessing` module.

*   **Baseline Correction**: To remove background signals, you can use `polynomial_correction(spectrum, poly_order)`. A low order (e.g., 2 or 3) is typically sufficient.
*   **Normalization**: To account for variations in acquisition time or distance, you can normalize the data. `row_normalize_sum(matrix)` normalizes each spectrum (row) to a sum of 1.
*   **Scaling**: Scaling is crucial to balance the influence of major and trace elements.
    *   **`autoscaling(matrix)`**: Standardizes each variable (column) to have a mean of 0 and a standard deviation of 1. This is ideal for point-analysis data.
    *   **`pareto_scale(matrix)`**: Mean-centers and scales each variable by the square root of its standard deviation. This is often preferred for full-spectrum data to avoid amplifying background noise.

## 3. Exploratory Analysis with PCA

Principal Component Analysis (PCA) is an excellent tool for identifying groupings and understanding the main sources of variation in your data.

1.  **Instantiate and Fit**: Create an instance of `chemtools.exploration.PrincipalComponentAnalysis` and fit it to your preprocessed data matrix.
2.  **Visualize**: Use the `chemtools.plots.DimensionalityReductionPlot` class to interpret the results:
    *   **`plot_scores()`**: Shows how your samples cluster. Samples with similar chemical composition will appear close together.
    *   **`plot_loadings()`**: Shows which variables (elements or energy channels) are most influential for each principal component.
    *   **`plot_biplot()`**: Overlays scores and loadings to see the relationships between samples and variables simultaneously.

## 4. Classification with SIMCA

SIMCA is a powerful method for classifying unknown samples based on models built from known classes (e.g., authenticating an artifact against a set of genuine examples).

1.  **Instantiate and Fit**: Create an instance of `chemtools.classification.SIMCA`. Fit it with your training data matrix and a corresponding array of class labels (e.g., 'Alloy A', 'Alloy B'). The class automatically builds a separate PCA model for each class.
2.  **Predict**: Use the `predict(X_new)` method to classify new, unknown samples. The method returns which class(es) each new sample belongs to. A sample can be assigned to multiple classes or flagged as an outlier if it fits no model.
3.  **Visualize**: Use the new `chemtools.plots.SIMCAPlot` class to visualize the results. The `plot_scores()` method will display each class model as a scatter plot of scores, typically with a confidence ellipse, showing the boundaries of each "class box".

## Runnable Example

For a complete, end-to-end demonstration of these workflows, please refer to the following script in the `examples/` directory:

**`examples/run_xrf_analysis_example.py`**

This script generates synthetic data for both point analysis and full-spectrum scenarios and walks through every step from preprocessing to final visualization using PCA and SIMCA.
