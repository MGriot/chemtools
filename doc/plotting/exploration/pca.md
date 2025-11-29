# Principal Component Analysis (PCA) Plots

Visualizing the results of a Principal Component Analysis (PCA) is essential for interpreting the components and understanding the structure of the data. The `PCAplots` class provides a comprehensive suite of visualizations for a fitted `PrincipalComponentAnalysis` object.

## `plot_eigenvalues`
Plots the eigenvalues in a "scree plot", which helps in determining the number of principal components to retain. Various statistical criteria can be overlaid.

**Example:**
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/exploration/pca/pca_eigenvalues_classic_professional_dark.png">
  <img alt="PCA Eigenvalues Plot" src="../../img/exploration/pca/pca_eigenvalues_classic_professional_light.png">
</picture>

## `plot_scores`
Plots the scores of the observations on the principal components. This is useful for identifying clusters, trends, and outliers in the data.

**Example:**
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/exploration/pca/pca_scores_classic_professional_dark.png">
  <img alt="PCA Scores Plot" src="../../img/exploration/pca/pca_scores_classic_professional_light.png">
</picture>

## `plot_loadings`
Plots the loadings of the original variables on the principal components. This shows how much each original variable contributes to each principal component.

**Example:**
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/exploration/pca/pca_loadings_classic_professional_dark.png">
  <img alt="PCA Loadings Plot" src="../../img/exploration/pca/pca_loadings_classic_professional_light.png">
</picture>

## `plot_biplot`
A biplot overlays the scores and loadings plots, allowing you to visualize the relationship between the observations and the variables simultaneously.

**Example:**
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/exploration/pca/pca_biplot_classic_professional_dark.png">
  <img alt="PCA Biplot" src="../../img/exploration/pca/pca_biplot_classic_professional_light.png">
</picture>

## Other Plots
The `PCAplots` class also provides other useful visualizations:
- **`plot_correlation_matrix()`**: A heatmap of the correlation matrix of the original variables.
- **`plot_hotteling_t2_vs_q()`**: A diagnostic plot for outlier detection.
- **`plot_pci_contribution()`**: Shows the contribution of each variable to the principal components.
