import numpy as np
import pandas as pd
from chemtools.exploration import PrincipalComponentAnalysis
from chemtools.classification import SIMCA
from chemtools.preprocessing import autoscaling, polynomial_correction, row_normalize_sum
from chemtools.plots import DimensionalityReductionPlot, SIMCAPlot
import matplotlib.pyplot as plt

# --- Data Generation Functions ---

def generate_xrf_point_data():
    """Generates synthetic XRF point analysis data for PCA."""
    print("Generating synthetic XRF point data for PCA...")
    np.random.seed(0)
    # Three clusters representing different material sources
    # Source 1: High Fe, Low Ni
    source1 = np.random.rand(20, 2) * [20, 5] + [80, 10]
    # Source 2: Low Fe, High Ni
    source2 = np.random.rand(20, 2) * [15, 25] + [20, 70]
    # Source 3: Medium Fe, Medium Ni
    source3 = np.random.rand(20, 2) * [30, 30] + [40, 40]
    
    # Adding a third element, correlated with the first two
    source1 = np.hstack([source1, (source1[:, 0].reshape(-1, 1) * 0.2 + np.random.randn(20, 1) * 2)])
    source2 = np.hstack([source2, (source2[:, 1].reshape(-1, 1) * 0.3 + np.random.randn(20, 1) * 2)])
    source3 = np.hstack([source3, ((source3[:, 0] + source3[:, 1]).reshape(-1, 1) * 0.1 + np.random.randn(20, 1) * 2)])

    X = np.vstack([source1, source2, source3])
    # Add some noise
    X += np.random.randn(*X.shape) * 2

    variables = ['Fe', 'Ni', 'Cr']
    objects = [f'Sample_{i+1}' for i in range(X.shape[0])]
    
    df = pd.DataFrame(X, columns=variables, index=objects)
    print("Synthetic point data generated with 3 elements and 60 samples.")
    return df

def pseudo_voigt(x, x0, sigma, gamma):
    """Helper to create a pseudo-Voigt peak profile."""
    return (1 - gamma) * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + \
           gamma * (1 / np.pi) * (sigma / ((x - x0) ** 2 + sigma ** 2))

def generate_xrf_spectrum_data():
    """Generates synthetic full-spectrum XRF data for SIMCA with high separation."""
    print("\nGenerating synthetic full-spectrum XRF data for SIMCA...")
    np.random.seed(123)
    n_channels = 1024
    channels = np.arange(n_channels)
    
    # --- Class A ("Alloy A") ---
    X_A = []
    for _ in range(25): # More samples
        baseline = 20 + 0.01 * channels + np.random.normal(0, 1, n_channels)
        peak1 = 8000 * pseudo_voigt(channels, 200, 10, 0.7) # Sharper, stronger peak
        peak2 = 4000 * pseudo_voigt(channels, 400, 15, 0.5)
        spectrum = baseline + peak1 + peak2 + np.random.poisson(5, n_channels) # Less noise
        X_A.append(spectrum)
        
    # --- Class B ("Alloy B") ---
    X_B = []
    for _ in range(25): # More samples
        baseline = 25 + 0.01 * channels + np.random.normal(0, 1, n_channels)
        peak1 = 7000 * pseudo_voigt(channels, 700, 12, 0.6) # Shifted far away
        peak3 = 3000 * pseudo_voigt(channels, 850, 18, 0.4)
        spectrum = baseline + peak1 + peak3 + np.random.poisson(5, n_channels) # Less noise
        X_B.append(spectrum)
        
    # --- Unknown Samples ---
    X_new = []
    # Sample 1: Perfect Alloy A
    X_new.append(20 + 0.01 * channels + 8000 * pseudo_voigt(channels, 200, 10, 0.7) + 4000 * pseudo_voigt(channels, 400, 15, 0.5))
    # Sample 2: Perfect Alloy B
    X_new.append(25 + 0.01 * channels + 7000 * pseudo_voigt(channels, 700, 12, 0.6) + 3000 * pseudo_voigt(channels, 850, 18, 0.4))
    # Sample 3: Obvious Outlier (peak in the middle)
    X_new.append(22 + 0.01 * channels + 9000 * pseudo_voigt(channels, 550, 15, 0.5))

    X_train = np.array(X_A + X_B)
    y_train = np.array(['Alloy A'] * 25 + ['Alloy B'] * 25)
    X_new = np.array(X_new)

    print("Generated 50 training spectra and 3 new spectra for prediction.")
    return X_train, y_train, X_new, channels

# --- Main Workflow ---

def run_pca_example():
    """Demonstrates PCA on point-analysis XRF data."""
    print("\n--- Running PCA Example on Point Data ---")
    df = generate_xrf_point_data()
    X = df.values

    # Preprocess the data
    X_scaled = autoscaling(X)

    # Fit PCA model
    pca = PrincipalComponentAnalysis()
    pca.fit(X_scaled)
    pca.reduction(n_components=2)
    
    print("\nPCA Model Summary:")
    print(pca.summary)
    
    # Initialize plotter
    plotter = DimensionalityReductionPlot(pca)

    # Plot scores
    print("\nGenerating Scores Plot...")
    fig_scores = plotter.plot_scores(title="PCA Scores Plot")
    plt.show()

    # Plot loadings
    print("Generating Loadings Plot...")
    fig_loadings = plotter.plot_loadings(title="PCA Loadings Plot")
    plt.show()
    
    # Plot biplot
    print("Generating Biplot...")
    fig_biplot = plotter.plot_biplot(title="PCA Biplot")
    plt.show()

def run_simca_example():
    """Demonstrates SIMCA on full-spectrum XRF data."""
    print("\n--- Running SIMCA Example on Full Spectrum Data ---")
    X_train, y_train, X_new, channels = generate_xrf_spectrum_data()

    # --- Preprocessing ---
    print("\nPreprocessing spectra...")
    # 1. Baseline Correction
    X_train_corrected = np.array([polynomial_correction(s, poly_order=3) for s in X_train])
    X_new_corrected = np.array([polynomial_correction(s, poly_order=3) for s in X_new])
    # 2. Normalization
    X_train_normalized = row_normalize_sum(X_train_corrected)
    X_new_normalized = row_normalize_sum(X_new_corrected)
    
    # Optional: Visualize a corrected spectrum
    plt.figure(figsize=(8, 4))
    plt.title("Example of a Preprocessed Spectrum")
    plt.plot(channels, X_train_normalized[0], label="Alloy A, Sample 0 (Corrected)")
    plt.plot(channels, X_train_normalized[15], label="Alloy B, Sample 0 (Corrected)")
    plt.xlabel("Energy Channels")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- SIMCA Modeling ---
    print("\nFitting SIMCA model...")
    # Use 2 components for each class model and a stricter alpha
    simca_model = SIMCA(n_components=2, alpha=0.01)
    simca_model.fit(X_train_normalized, y_train)

    print("\nSIMCA Model Summary:")
    print(simca_model.summary)
    
    # --- Prediction ---
    print("\nPredicting classes for new spectra...")
    predictions = simca_model.predict(X_new_normalized)

    print("\n--- SIMCA Prediction Results ---")
    result_map = {0: "Alloy A", 1: "Alloy B", 2: "Outlier"}
    for i, p in enumerate(predictions):
        sample_info = f"New Sample {i+1} (expected: {result_map[i]}):"
        if not p:
            assignment = "Is an OUTLIER (does not belong to any class)."
        else:
            assignment = f"Belongs to class(es): {', '.join(p)}"
        print(f"{sample_info.ljust(45)} {assignment}")
        
    # --- Visualization ---
    print("\nGenerating SIMCA Scores Plot...")
    simca_plotter = SIMCAPlot(simca_model)
    fig_simca = simca_plotter.plot_scores(
        title="SIMCA Class Models with 95% Confidence Ellipses",
        confidence_level=0.95
    )
    plt.show()


if __name__ == "__main__":
    run_pca_example()
    print("\n" + "="*50 + "\n")
    run_simca_example()
