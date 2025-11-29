import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import plotter classes
from chemtools.plots.exploration.pca_plots import PCAplots

# Import model classes (for dummy data/models)
from chemtools.exploration import PrincipalComponentAnalysis

def generate_pca_plots():
    """
    This script generates various PCA-related plots for demonstration purposes.
    The plots are saved in the 'doc/img/exploration/pca' directory.
    """
    print("--- Generating PCA Plots ---")
    output_dir = os.path.join("doc", "img", "exploration", "pca")
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]

    # --- 1. Create Synthetic Data ---
    np.random.seed(42)
    num_samples = 50
    num_variables = 7
    # Create some correlated data for PCA
    data_raw = np.random.randn(num_samples, num_variables)
    data_raw[:, 0] = data_raw[:, 0] * 3 + data_raw[:, 1] # Introduce correlation
    data_raw[:, 2] = data_raw[:, 2] * 2 - data_raw[:, 3]
    
    pca_data_X = pd.DataFrame(data_raw, columns=[f'Var{i+1}' for i in range(num_variables)])
    pca_data_X['Obj'] = [f'Obj{i+1}' for i in range(num_samples)]
    
    # --- 2. Perform PCA ---
    pca_model = PrincipalComponentAnalysis()
    pca_model.fit(pca_data_X.drop('Obj', axis=1).values, 
                  variables_names=[f'Var{i+1}' for i in range(num_variables)],
                  objects_names=pca_data_X['Obj'].tolist())
    pca_model.reduction(n_components=5) # Reduce to 5 components for plotting
    pca_model.statistics() # Calculate statistics like T2 and Q
    print("\nPCA performed successfully with synthetic data.")

    # --- 3. Generate and Save Plots ---
    for theme in themes:
        print(f"\n  Generating plots for theme: {theme}")
        try:
            plotter = PCAplots(pca_model, library="matplotlib", theme=theme)
            
            # Correlation Matrix
            fig = plotter.plot_correlation_matrix(subplot_title=f"PCA Correlation Matrix ({theme})")
            fig.savefig(os.path.join(output_dir, f"pca_correlation_matrix_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Eigenvalues (Scree Plot)
            fig = plotter.plot_eigenvalues(criteria=["greater_than_one", "average_eigenvalue", "broken_stick"], subplot_title=f"PCA Eigenvalue Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"pca_eigenvalues_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Scores Plot
            fig = plotter.plot_scores(components=(0,1), subplot_title=f"PCA Scores Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"pca_scores_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Loadings Plot
            fig = plotter.plot_loadings(components=(0,1), subplot_title=f"PCA Loadings Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"pca_loadings_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Biplot
            fig = plotter.plot_biplot(components=(0,1), subplot_title=f"PCA Biplot ({theme})")
            fig.savefig(os.path.join(output_dir, f"pca_biplot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # Hotelling's T2 vs. Q Residuals
            fig = plotter.plot_hotteling_t2_vs_q(subplot_title=f"PCA Hotelling's T2 vs. Q ({theme})")
            fig.savefig(os.path.join(output_dir, f"pca_hotteling_t2_vs_q_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # PCI Contribution Plot
            fig = plotter.plot_pci_contribution(subplot_title=f"PCA PCI Contribution ({theme})")
            fig.savefig(os.path.join(output_dir, f"pca_pci_contribution_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            print(f"    - Saved all PCA plots for theme: {theme}")
        except Exception as e:
            print(f"    - Error generating PCA plots for theme {theme}: {e}")

    print("\n--- All PCA plots have been generated. ---")

if __name__ == "__main__":
    generate_pca_plots()