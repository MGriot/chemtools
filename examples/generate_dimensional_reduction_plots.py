import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import plotter classes
from chemtools.plots.dimensional_reduction.dimensional_reduction_plot import DimensionalityReductionPlot
from chemtools.plots.exploration.mca_plots import MCAPlots
from chemtools.plots.exploration.pca_plots import PCAplots

# Import model classes (for dummy data/models)
from chemtools.exploration import PrincipalComponentAnalysis
from chemtools.dimensional_reduction import FactorAnalysis, FactorAnalysisOfMixedData
from chemtools.exploration import MultipleCorrespondenceAnalysis

def generate_dimensional_reduction_plots():
    """
    This script generates plots for dimensional reduction techniques:
    PCA, FAMD (using DimensionalityReductionPlot), and MCA.
    """
    print("--- Generating Dimensional Reduction Plots ---")

    # Define themes to use for generating plots
    themes = ["classic_professional_light", "classic_professional_dark"]

    # --- Sample Data ---
    # PCA data
    pca_data_X = pd.DataFrame(np.random.rand(20, 5), columns=[f'Var{i+1}' for i in range(5)])
    pca_data_X['Obj'] = [f'Obj{i+1}' for i in range(20)]
    
    # MCA data
    mca_data = pd.DataFrame({
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Education': np.random.choice(['High', 'Medium', 'Low'], 100),
        'Job': np.random.choice(['Blue Collar', 'White Collar'], 100)
    })
    # For MCA, we typically need a contingency table or one-hot encoded data.
    # Let's create a simple contingency table for the example.
    mca_contingency_table = pd.crosstab(mca_data['Gender'], mca_data['Education'])
    

    # --- Plot Generation ---
    
    # 1. PCA Plots (using PCAplots)
    output_dir_pca = "doc/img/exploration/pca"
    os.makedirs(output_dir_pca, exist_ok=True)
    print("\nGenerating PCA Plots...")
    
    # Fit a dummy PCA model
    pca_model = PrincipalComponentAnalysis()
    pca_model.fit(pca_data_X[['Var1', 'Var2', 'Var3', 'Var4', 'Var5']].values, 
                  variables_names=[f'Var{i+1}' for i in range(5)],
                  objects_names=pca_data_X['Obj'].tolist())
    pca_model.reduction(n_components=2)
    pca_model.statistics() # Calculate stats for T2/Q plots
    
    for theme in themes:
        try:
            plotter = PCAplots(pca_model, theme=theme)
            
            # Correlation Matrix
            fig = plotter.plot_correlation_matrix(subplot_title=f"Correlation Matrix ({theme})")
            fig.savefig(os.path.join(output_dir_pca, f"pca_correlation_matrix_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Eigenvalues (Scree Plot)
            fig = plotter.plot_eigenvalues(criteria=None, subplot_title=f"Eigenvalue Plot ({theme})")
            fig.savefig(os.path.join(output_dir_pca, f"pca_eigenvalues_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Scores Plot
            fig = plotter.plot_scores(components=(0,1), subplot_title=f"Scores Plot ({theme})")
            fig.savefig(os.path.join(output_dir_pca, f"pca_scores_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Loadings Plot
            fig = plotter.plot_loadings(components=(0,1), subplot_title=f"Loadings Plot ({theme})")
            fig.savefig(os.path.join(output_dir_pca, f"pca_loadings_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Biplot
            fig = plotter.plot_biplot(components=(0,1), subplot_title=f"Biplot ({theme})")
            fig.savefig(os.path.join(output_dir_pca, f"pca_biplot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # Hotelling's T2 vs. Q Residuals
            fig = plotter.plot_hotteling_t2_vs_q(subplot_title=f"Hotelling's T2 vs. Q ({theme})")
            fig.savefig(os.path.join(output_dir_pca, f"pca_hotteling_t2_vs_q_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # PCI Contribution Plot
            fig = plotter.plot_pci_contribution(subplot_title=f"PCI Contribution ({theme})")
            fig.savefig(os.path.join(output_dir_pca, f"pca_pci_contribution_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            print(f"  - Saved PCA plots for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating PCA plots for theme {theme}: {e}")
            
    # 2. MCA Plots (using MCAPlots)
    output_dir_mca = "doc/img/exploration/mca" # Assuming new directory for MCA images
    os.makedirs(output_dir_mca, exist_ok=True)
    print("\nGenerating MCA Plots...")
    
    # Fit a dummy MCA model
    mca_model = MultipleCorrespondenceAnalysis()
    mca_model.fit(mca_contingency_table.values, 
                  variables_names=mca_contingency_table.columns.tolist(),
                  objects_names=mca_contingency_table.index.tolist())

    for theme in themes:
        try:
            plotter = MCAPlots(mca_model, theme=theme)
            
            # Eigenvalues Plot
            fig = plotter.plot_eigenvalues(subplot_title=f"MCA Eigenvalues ({theme})")
            fig.savefig(os.path.join(output_dir_mca, f"mca_eigenvalues_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            # Objects Plot
            fig = plotter.plot_objects(subplot_title=f"MCA Objects Plot ({theme})")
            fig.savefig(os.path.join(output_dir_mca, f"mca_objects_{theme}.png"), bbox_inches='tight')
            plt.close(fig)

            print(f"  - Saved MCA plots for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating MCA plots for theme {theme}: {e}")

    # 3. DimensionalityReductionPlot (using FAMD for example)
    output_dir_dim_red = "doc/img/dimensional_reduction/famd" # Assuming new directory for FAMD images
    os.makedirs(output_dir_dim_red, exist_ok=True)
    print("\nGenerating Dimensionality Reduction Plots (FAMD example)...")
    
    # Create sample mixed data for FAMD
    famd_data = pd.DataFrame({
        'quant1': np.random.rand(30),
        'quant2': np.random.rand(30) * 10,
        'qual1': np.random.choice(['A', 'B', 'C'], 30),
        'qual2': np.random.choice(['X', 'Y'], 30)
    })
    qualitative_vars = ['qual1', 'qual2']

    # Fit a dummy FAMD model
    famd_model = FactorAnalysisOfMixedData(n_components=2)
    famd_model.fit(famd_data, qualitative_variables=qualitative_vars)
    
    for theme in themes:
        try:
            plotter = DimensionalityReductionPlot(famd_model, theme=theme)
            
            # Scores Plot
            fig = plotter.plot_scores(subplot_title=f"FAMD Scores Plot ({theme})")
            fig.savefig(os.path.join(output_dir_dim_red, f"famd_scores_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # Loadings Plot
            fig = plotter.plot_loadings(subplot_title=f"FAMD Loadings Plot ({theme})")
            fig.savefig(os.path.join(output_dir_dim_red, f"famd_loadings_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            
            print(f"  - Saved FAMD plots for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating FAMD plots for theme {theme}: {e}")


    print("\n--- All Dimensional Reduction plots have been generated. ---")

if __name__ == "__main__":
    generate_dimensional_reduction_plots()
