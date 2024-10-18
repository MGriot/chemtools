# Principal Component Analysis (PCA)

>Si prende come riferimento il manuale di todeschini e PCA_toolbox per matlab.

## chemtools: Principal Component Analysis Module

This module provides tools for performing Principal Component Analysis (PCA), a powerful dimensionality reduction technique. It includes the `PrincipalComponentAnalysis` class for data fitting and transformation, along with various plotting utilities in `pca_plots.py` to visualize the results.

### 1. Principal Component Analysis Class (`PrincipalComponentAnalysis.py`)

The `PrincipalComponentAnalysis` class is the core component of this module, allowing users to apply PCA to their datasets. 

**Key Features:**

* **Data Preprocessing:** Automatically handles data preprocessing by autoscaling the input data (mean-centering and scaling to unit variance).
* **Eigenvalue and Eigenvector Calculation:** Computes the eigenvalues and eigenvectors of the correlation matrix, representing the principal components and their contributions.
* **Dimensionality Reduction:** Reduces the dataset's dimensionality by selecting a specified number of principal components based on various criteria.
* **Statistical Metrics:** Calculates essential statistical metrics like Hotelling's T-squared and Q statistics (Squared Prediction Error) for assessing model performance.
* **Data Transformation:** Projects new data points onto the reduced principal component space for further analysis.

**Workflow:**

1. **Initialization:** Create a `PrincipalComponentAnalysis` object.
2. **Data Fitting:** Utilize the `fit()` method to fit the PCA model to your dataset. You can provide optional variable names and object names for clarity.
3. **Dimensionality Reduction:** Employ the `reduction()` method to reduce the dataset's dimensionality to a desired number of principal components.
4. **Statistical Analysis:** Use the `statistics()` method to calculate various statistical metrics for assessing the quality of the PCA and interpreting the results.
5. **New Data Projection:**  Apply the `transform()` method to project new data points onto the principal component space defined by the fitted model.

### 2. PCA Plots (`pca_plots.py`)

The `pca_plots.py` module provides numerous plotting functions for visually exploring PCA results. Here's an overview:

**Data Visualization:**

* **`plot_correlation_matrix`:** Visualizes the correlation matrix of the input data, highlighting relationships between variables.
* **`plot_hotteling_t2_vs_q`:** Generates a scatter plot of Hotelling's T-squared against the Q statistic, aiding in outlier detection.

**Component Selection Criteria:**

* **`plot_eigenvalues_greater_than_one`:**  Plots eigenvalues, marking those greater than 1 (Kaiser criterion) for component selection.
* **`plot_eigenvalues_variance`:** Displays the percentage of variance explained by each principal component.
* **`plot_cumulative_variance`:** Shows the cumulative variance explained as the number of components increases.
* **`plot_average_eigenvalue_criterion`:** Visualizes the average eigenvalue criterion (AEC) for component selection.
* **`plot_KP_criterion`:** Demonstrates the Kaiser-Piggott (KP) criterion for component selection.
* **`plot_KL_criterion`:** Illustrates the KL criterion for component selection.
* **`plot_CAEC_criterion`:** Depicts the cumulative average eigenvalue criterion (CAEC) for component selection.
* **`plot_broken_stick`:**  Visualizes the broken stick criterion for component selection.
* **`plot_eigenvalue`:** Generates a comprehensive plot combining various eigenvalue criteria for component selection.

**Component Interpretation:**

* **`plot_pci_contribution`:**  Displays the contributions of each variable to each principal component.
* **`plot_loadings`:** Visualizes the loadings of the principal components, showing the relationship between variables and principal components.
* **`plot_scores`:** Plots the scores of the principal components, representing the projection of the original data onto the principal component space.
* **`plot_biplot`:** Creates a biplot, combining both loadings and scores to visualize relationships between variables and objects in the principal component space.
* **`plot_classes_pca`:**  Specifically designed for visualizing class separation in PCA space, showing centroids, class boundaries (ellipses), and optional new data points.

---

This documentation provides a concise overview of the Principal Component Analysis module within `chemtools`. More detailed descriptions, parameters, and examples for each function and class are available in their respective docstrings within the code.
