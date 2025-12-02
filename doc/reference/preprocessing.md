# Preprocessing Module Reference (`chemtools.preprocessing`)

The `chemtools.preprocessing` module provides a collection of essential functions for preparing your data for multivariate analysis. These functions are designed to handle common data preparation tasks such as scaling, normalization, baseline correction, and fundamental matrix operations.

## Scaling and Normalization Functions

Functions to adjust data to balance variable influence and remove systemic variations.

### `autoscaling(x)`

Performs autoscaling (standardization or Z-score normalization) on the data.

*   **Signature:** `autoscaling(x: np.ndarray) -> np.ndarray`
*   **Parameters:**
    *   `x` (`np.ndarray`): The input data matrix.
*   **Returns:**
    *   `np.ndarray`: The autoscaled data matrix.
*   **Description:** Centers each column (variable) to have a mean of 0 and scales it to have a standard deviation of 1.

### `pareto_scale(x)`

Performs Pareto scaling on the data.

*   **Signature:** `pareto_scale(x: np.ndarray) -> np.ndarray`
*   **Parameters:**
    *   `x` (`np.ndarray`): The input data matrix.
*   **Returns:**
    *   `np.ndarray`: The Pareto-scaled data matrix.
*   **Description:** Centers each column to have a mean of 0 and scales it by the square root of its standard deviation.

### `row_normalize_sum(x)`

Performs row-wise normalization to a constant sum.

*   **Signature:** `row_normalize_sum(x: np.ndarray) -> np.ndarray`
*   **Parameters:**
    *   `x` (`np.ndarray`): The input data matrix.
*   **Returns:**
    *   `np.ndarray`: The row-normalized data matrix.
*   **Description:** Divides each value in a row by the sum of all values in that row.

## Baseline Correction Functions

Functions to remove unwanted background signals from spectral data.

### `polynomial_correction(y, poly_order=2)`

Removes a curved baseline from a 1D spectrum by fitting and subtracting a polynomial.

*   **Signature:** `polynomial_correction(y: np.ndarray, poly_order: int = 2) -> np.ndarray`
*   **Parameters:**
    *   `y` (`np.ndarray`): The 1D spectrum (data array).
    *   `poly_order` (`int`, optional): The order of the polynomial to fit. Defaults to `2`.
*   **Returns:**
    *   `np.ndarray`: The baseline-corrected spectrum.
*   **Description:** Fits a polynomial of the specified order to the spectrum and subtracts it to remove the baseline.

## Matrix Operation Functions

These functions are used for fundamental matrix calculations often required as intermediate steps in more complex analyses.

### `correlation_matrix(X)`

Calculates the correlation matrix for a given data matrix.

*   **Signature:** `correlation_matrix(X: np.ndarray) -> np.ndarray`
*   **Parameters:**
    *   `X` (`np.ndarray`): The input data matrix (assumed to be autoscaled if Pearson correlation is desired).
*   **Returns:**
    *   `np.ndarray`: The correlation matrix.

### `diagonalized_matrix(X)`

Performs eigenvalue decomposition of a square matrix.

*   **Signature:** `diagonalized_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
*   **Parameters:**
    *   `X` (`np.ndarray`): The square input matrix.
*   **Returns:**
    *   `Tuple[np.ndarray, np.ndarray]`: A tuple containing:
        *   Eigenvalues (`np.ndarray`)
        *   Eigenvectors (`np.ndarray`)

### `matrix_mean(x, mode)`

Calculates the mean of a matrix along either rows or columns.

*   **Signature:** `matrix_mean(x: np.ndarray, mode: Literal['row', 'column']) -> np.ndarray`
*   **Parameters:**
    *   `x` (`np.ndarray`): The input data matrix.
    *   `mode` (`Literal['row', 'column']`): Specifies whether to calculate the mean along 'row' or 'column'.
*   **Returns:**
    *   `np.ndarray`: The mean vector.

### `matrix_standard_deviation(X, mode)`

Calculates the standard deviation of a matrix along either rows or columns.

*   **Signature:** `matrix_standard_deviation(X: np.ndarray, mode: Literal['row', 'column']) -> np.ndarray`
*   **Parameters:**
    *   `X` (`np.ndarray`): The input data matrix.
    *   `mode` (`Literal['row', 'column']`): Specifies whether to calculate the standard deviation along 'row' or 'column'.
*   **Returns:**
    *   `np.ndarray`: The standard deviation vector.

### `matrix_variance(x, axis=0)`

Calculates the sample variance of a matrix along a specified axis.

*   **Signature:** `matrix_variance(x: np.ndarray, axis: int = 0) -> np.ndarray`
*   **Parameters:**
    *   `x` (`np.ndarray`): The input data matrix.
    *   `axis` (`int`, optional): The axis along which to calculate the variance. `0` for columns, `1` for rows. Defaults to `0`.
*   **Returns:**
    *   `np.ndarray`: The variance vector.
