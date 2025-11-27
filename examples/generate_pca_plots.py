import os
import pandas as pd
from chemtools.exploration import PrincipalComponentAnalysis
from chemtools.plots.exploration import PCAplots
import matplotlib.pyplot as plt

def generate_pca_plots():
    """
    This script runs a PCA and generates all associated plots for all available themes.
    The plots are saved in the 'doc/img/exploration/pca' directory.
    """
    # --- 1. Setup ---
    output_dir = os.path.join("doc", "img", "exploration", "pca")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    themes = [
        "classic_professional_light", "classic_professional_dark",
        "amethyst_twilight_light", "amethyst_twilight_dark",
        "botanical_sage_light", "botanical_sage_dark",
        "emerald_grove_light", "emerald_grove_dark",
        "eminent_graphite_light", "eminent_graphite_dark",
        "golden_umber_light", "golden_umber_dark",
        "oceanic_slate_light", "oceanic_slate_dark",
        "terracotta_sun_light", "terracotta_sun_dark",
    ]

    # --- 2. Load Data ---
    try:
        data = pd.read_excel("data/pca.xlsx", index_col=0)
        X = data.values
        variables = data.columns.tolist()
        objects = data.index.tolist()
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: 'data/pca.xlsx' not found. Please ensure the sample data is present.")
        return

    # --- 3. Perform PCA ---
    pca = PrincipalComponentAnalysis()
    pca.fit(X, variables_names=variables, objects_names=objects)
    pca.reduction(n_components=5) # Reduce to 5 components for plotting
    pca.statistics() # Calculate statistics like T2 and Q
    print("PCA performed successfully.")

    # --- 4. Generate and Save Plots ---
    plot_functions = {
        "correlation_matrix": "plot_correlation_matrix",
        "eigenvalues": "plot_eigenvalues",
        "loadings": "plot_loadings",
        "scores": "plot_scores",
        "biplot": "plot_biplot",
        "hotteling_t2_vs_q": "plot_hotteling_t2_vs_q",
        "pci_contribution": "plot_pci_contribution",
    }

    for theme in themes:
        print(f"  Generating plots for theme: {theme}")
        try:
            # Most PCA plots are matplotlib-only for now
            plotter = PCAplots(pca, library="matplotlib", theme=theme)

            for plot_name, method_name in plot_functions.items():
                plot_method = getattr(plotter, method_name)
                
                # Some methods have specific arguments
                if plot_name in ["loadings", "scores", "biplot"]:
                    # Generate single plot for first 2 components
                    fig = plot_method(components=(0, 1))
                elif plot_name == "eigenvalues":
                    fig = plot_method(criteria=["greater_than_one", "average_eigenvalue", "broken_stick"])
                else:
                    fig = plot_method()

                if fig:
                    filename = f"pca_{plot_name}_{theme}.png"
                    filepath = os.path.join(output_dir, filename)
                    fig.savefig(filepath, bbox_inches='tight')
                    plt.close(fig) # Close figure to free memory
                    print(f"    - Saved {filename}")
                else:
                    print(f"    - Could not generate {plot_name} for theme {theme}")

        except Exception as e:
            print(f"Error generating plots for theme {theme}: {e}")

    print("\nAll PCA plots have been generated.")

if __name__ == "__main__":
    generate_pca_plots()