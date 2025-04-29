import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from chemtools.dimensional_reduction.PrincipalComponentAnalysis import (
    PrincipalComponentAnalysis,
)
from chemtools.plots.exploration.pca_plots import PCAplots


def test_pca():
    # Create a database with object names and variable titles
    data = {
        "Nome Oggetto": ["Oggetto1", "Oggetto2", "Oggetto3", "Oggetto4"],
        "Variabile1": [1.2, 2.3, 3.1, 4.5],
        "Variabile2": [3.4, 1.2, 4.5, 2.3],
        "Variabile3": [5.6, 3.4, 2.1, 1.2],
        "Variabile4": [7.8, 6.5, 3.2, 4.1],
        "Variabile5": [2.3, 4.6, 1.8, 3.9],
        "Variabile6": [9.0, 8.1, 7.2, 6.3],
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame
    print("Created Database:")
    print(df)

    # Prepare the data for PCA (excluding the 'Nome Oggetto' column)
    X = df.drop("Nome Oggetto", axis=1)
    variables = X.columns.tolist()  # Get variable names
    objects = df["Nome Oggetto"].tolist()  # Get object names
    X = X.to_numpy()

    pca = PrincipalComponentAnalysis()
    pca.fit(X, variables_names=variables, objects_names=objects)

    # Initialize plots with explicit theme settings
    plots = PCAplots(pca, library="matplotlib", theme="light", style_preset="default")


    plots.plot_correlation_matrix()
    plots.plot_eigenvalues()

    pca.reduction(3)
    plots.plot_loadings()
    plots.plot_biplot()
    plots.plot_scores()



    print(pca.summary)


if __name__ == "__main__":
    test_pca()
