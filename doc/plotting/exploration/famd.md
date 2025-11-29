# Factor Analysis of Mixed Data (FAMD) Plots

Factor Analysis of Mixed Data (FAMD) is a principal component method used to analyze a dataset containing both quantitative and qualitative variables. The `DimensionalityReductionPlot` class can be used to visualize the results from a fitted `FactorAnalysisOfMixedData` object.

## `plot_scores`
Plots the scores of the observations (rows) in the new dimensional space created by the principal components. This helps to identify clusters and patterns among the observations.

**Example:**
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/exploration/famd_scores_classic_professional_dark.png">
  <img alt="FAMD Scores Plot" src="../../img/plots/exploration/famd_scores_classic_professional_light.png">
</picture>

## `plot_loadings`
Plots the loadings of the original variables (both quantitative and qualitative) on the principal components. This visualization shows how each original variable contributes to the construction of the principal components.

**Example:**
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/exploration/famd_loadings_classic_professional_dark.png">
  <img alt="FAMD Loadings Plot" src="../../img/plots/exploration/famd_loadings_classic_professional_light.png">
</picture>
