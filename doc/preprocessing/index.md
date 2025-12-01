# Preprocessing Functions

The `chemtools.preprocessing` module provides a collection of essential functions for preparing your data for multivariate analysis. Proper preprocessing is a critical step to ensure that the results of your analysis are meaningful and not biased by artifacts in the data.

## Scaling and Normalization

These functions adjust the data to meet the assumptions of various statistical models, such as by balancing the influence of variables with different scales or removing variations unrelated to the chemical properties of the samples.

### `autoscaling(x)`
Performs autoscaling (standardization or Z-score normalization) on the data.
*   **Action**: Centers each column (variable) to have a mean of 0 and scales it to have a standard deviation of 1.
*   **Use Case**: This is the most common scaling method and is highly recommended when variables have different units or vastly different scales (e.g., major vs. trace elements in XRF).

### `pareto_scale(x)`
Performs Pareto scaling.
*   **Action**: Centers each column to have a mean of 0 and scales it by the square root of its standard deviation.
*   **Use Case**: A good compromise between autoscaling and mean centering. It down-weights large-fold changes, which is particularly useful in fields like metabolomics or for full-spectrum data where autoscaling might excessively amplify noise.

### `row_normalize_sum(x)`
Performs row-wise normalization to a constant sum.
*   **Action**: Divides each value in a row by the sum of all values in that row.
*   **Use Case**: Essential for spectroscopic data (like XRF, Raman, or IR) to correct for variations in signal intensity caused by factors like sample amount, path length, or measurement time.

## Baseline Correction

### `polynomial_correction(y, poly_order=2)`
Removes a curved baseline from a 1D spectrum.
*   **Action**: Fits a polynomial of a specified order to the spectrum and subtracts it.
*   **Use Case**: Useful for removing simple, non-linear background signals from spectral data. A low polynomial order (e.g., 2 or 3) is recommended to avoid distorting actual peaks.

## Matrix Operations

These functions are used for fundamental matrix calculations often required as intermediate steps in more complex analyses.

### `correlation_matrix(X)`
Calculates the correlation matrix for a given data matrix, assuming the input `X` is already autoscaled.

### `diagonalized_matrix(X)`
Performs eigenvalue decomposition of a square matrix, returning the eigenvalues and eigenvectors.

### `matrix_mean(x, mode)`
Calculates the mean of a matrix along either rows (`mode='row'`) or columns (`mode='column'`).

### `matrix_standard_deviation(X, mode)`
Calculates the standard deviation of a matrix along either rows (`mode='row'`) or columns (`mode='column'`).

### `matrix_variance(x, axis=0)`
Calculates the sample variance of a matrix along a specified axis (0 for columns, 1 for rows).
