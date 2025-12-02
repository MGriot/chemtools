# Statistical Concepts

The `chemtools.stats` module provides a foundational set of statistical tools essential for data analysis in chemometrics. These tools cover descriptive statistics for single variables, various forms of Analysis of Variance (ANOVA) for comparing group means, and specific functions for regression analysis diagnostics. Understanding these statistical concepts is crucial for interpreting data, evaluating models, and drawing scientifically sound conclusions.

## Univariate Statistics: Describing Single Variables

Univariate statistics focus on analyzing a single variable at a time to summarize its characteristics. These measures provide insights into the central tendency, dispersion, and shape of the data distribution.

### Key Measures:

*   **Measures of Central Tendency:** Describe the "typical" or "middle" value of the data.
    *   **Arithmetic Mean:** The average value.
    *   **Median:** The middle value when data is ordered.
    *   **Geometric Mean:** Useful for data that grows exponentially or when dealing with ratios.
*   **Measures of Dispersion:** Describe how spread out the data points are.
    *   **Variance & Standard Deviation:** Quantify the average squared deviation and average deviation from the mean, respectively.
    *   **Relative Standard Deviation (RSD) / Coefficient of Variation (CV):** Expresses the standard deviation as a percentage of the mean, useful for comparing variability across datasets with different scales.
    *   **Standard Error of the Mean:** Estimates how much the sample mean is likely to vary from the population mean.
    *   **Range & Interquartile Range (IQR):** Simple measures of data spread.
    *   **Mean Absolute Deviation (MAD) & Median Absolute Deviation (MAD):** Measures of variability that are less sensitive to outliers than standard deviation.
*   **Measures of Position:** Identify specific points within the data distribution.
    *   **Minimum & Maximum Value:** The smallest and largest observations.
    *   **Quartiles (Q1, Q3):** Divide the data into four equal parts.
*   **Measures of Shape:** Describe the symmetry and "tailedness" of the distribution.
    *   **Skewness:** Indicates the asymmetry of the probability distribution.
    *   **Kurtosis:** Indicates the "tailedness" of the probability distribution, describing the shape of its peaks and tails.
*   **Confidence Intervals:** Provide a range of values, derived from sample statistics, that is likely to contain the true population parameter (e.g., mean) with a certain level of confidence.

## Analysis of Variance (ANOVA): Comparing Group Means

ANOVA is a powerful statistical technique used to compare the means of two or more groups to determine if there are statistically significant differences between them. It achieves this by partitioning the total observed variance into different components.

### Types of ANOVA:

*   **One-Way ANOVA:**
    *   **Purpose:** Compares the means of two or more independent groups on a single continuous dependent variable.
    *   **Hypothesis:** Tests whether the means of all groups are equal versus at least one group mean being different.
*   **Two-Way ANOVA:**
    *   **Purpose:** Extends one-way ANOVA to evaluate the effects of two independent categorical variables (factors) and their interaction on a continuous dependent variable.
    *   **Benefits:** Can detect interaction effects, where the effect of one factor depends on the level of the other factor.
*   **Multiway ANOVA (Concept):**
    *   **Purpose:** Generalizes two-way ANOVA to analyze the effects of three or more independent categorical variables and their interactions. (Typically handled by specialized libraries due to complexity).
*   **Multivariate Analysis of Variance (MANOVA) (Concept):**
    *   **Purpose:** Similar to ANOVA but with two or more continuous dependent variables. It tests whether there are statistically significant differences between groups on a combination of dependent variables. (Typically handled by specialized libraries due to complexity).

<h2>Regression Statistics: Evaluating Predictive Models</h2>

Regression statistics provide crucial metrics and diagnostics for assessing the performance and validity of regression models.

<h3>Key Metrics:</h3>

*   <b>Degrees of Freedom (DF):</b> Reflects the number of independent pieces of information available to estimate a parameter or calculate a statistic.
*   <b>R-squared (Coefficient of Determination):</b> Measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
    *   <b>Centered R-squared:</b> The standard R-squared.
    *   <b>Uncentered R-squared:</b> Used when the regression model is forced through the origin (no intercept).
    *   <b>Uncentered Adjusted R-squared:</b> An adjusted version of uncentered R-squared, accounting for the number of predictors.
*   <b>Student's t-distribution (Critical Values):</b> Used to calculate confidence intervals and perform hypothesis tests on regression coefficients.

## Further Reading

*   [Descriptive statistics](https://en.wikipedia.org/wiki/Descriptive_statistics) on Wikipedia
*   [Analysis of variance](https://en.wikipedia.org/wiki/Analysis_of_variance) on Wikipedia
*   [Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) on Wikipedia
*   [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) on Wikipedia
