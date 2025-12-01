import numpy as np

def polynomial_correction(y: np.ndarray, poly_order: int = 2) -> np.ndarray:
    """
    Performs baseline correction on a spectrum using polynomial fitting.

    This method fits a polynomial of a specified order to the spectral data
    and subtracts it, which can help in removing simple, curved baselines.

    Args:
        y (np.ndarray): A 1D array representing the spectrum (intensity values).
        poly_order (int): The order of the polynomial to fit for the baseline.
                          A lower order (e.g., 1, 2) is generally recommended to
                          avoid fitting actual peaks.

    Returns:
        np.ndarray: The baseline-corrected spectrum.
    """
    if y.ndim != 1:
        raise ValueError("Input 'y' must be a 1D array.")
    
    x = np.arange(len(y))
    
    # Fit the polynomial to the data
    coeffs = np.polyfit(x, y, poly_order)
    
    # Evaluate the polynomial to get the baseline
    baseline = np.polyval(coeffs, x)
    
    # Subtract the baseline
    return y - baseline
