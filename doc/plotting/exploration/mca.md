# Multiple Correspondence Analysis (MCA) Plots

Multiple Correspondence Analysis (MCA) is a technique used to analyze patterns in categorical data. The `MCAPlots` class provides visualizations for a fitted `MultipleCorrespondenceAnalysis` object.

## `plot_eigenvalues`
Plots the eigenvalues (or principal inertias) of the analysis. This is analogous to a scree plot in PCA and helps determine the dimensionality of the solution.

**Example:**
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/exploration/mca/mca_eigenvalues_classic_professional_dark.png">
  <img alt="MCA Eigenvalues Plot" src="../../img/exploration/mca/mca_eigenvalues_classic_professional_light.png">
</picture>

## `plot_objects`
Plots the positions of the row categories (objects) in the new dimensional space defined by the principal components. Categories that are close to each other are more similar in their response patterns.

**Example:**
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/exploration/mca/mca_objects_classic_professional_dark.png">
  <img alt="MCA Objects Plot" src="../../img/exploration/mca/mca_objects_classic_professional_light.png">
</picture>
