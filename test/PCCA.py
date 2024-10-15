import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from chemtools.exploration import PrincipalComponentAnalysis
from chemtools.plots.exploration import pca_plots


def main():
    """
    Demonstrates class separation using PCA.
    """

    # --- Data Preparation ---
    # Sample data with two classes (Class A and Class B)
    data = {
        "Nome Oggetto": ["A1", "A2", "A3", "A4", "B1", "B2", "B3"],
        "Variabile1": [1.2, 2.3, 3.1, 4.5, 5.8, 6.9, 7.6],
        "Variabile2": [3.4, 1.2, 4.5, 2.3, 1.1, 2.5, 3.8],
        "Variabile3": [5.6, 3.4, 2.1, 1.2, 4.3, 5.7, 6.2],
        "Variabile4": [7.8, 6.5, 3.2, 4.1, 2.9, 4.6, 5.1],
        "Variabile5": [2.3, 4.6, 1.8, 3.9, 6.2, 7.1, 8.4],
        "Variabile6": [9.0, 8.1, 7.2, 6.3, 3.7, 5.2, 6.9],
    }
    df = pd.DataFrame(data)
    print("Created Database:")
    print(df)

    # --- Separate data by class ---
    df_class_A = df[df["Nome Oggetto"].str.startswith("A")]
    df_class_B = df[df["Nome Oggetto"].str.startswith("B")]

    X_A = df_class_A.drop("Nome Oggetto", axis=1).to_numpy()
    X_B = df_class_B.drop("Nome Oggetto", axis=1).to_numpy()

    # --- Get variable names from the DataFrame ---
    variables = df.drop("Nome Oggetto", axis=1).columns.tolist()

    # --- PCA for each class ---
    pca_A = PrincipalComponentAnalysis()
    pca_A.fit(
        X_A,
        variables_names=variables,
        objects_names=df_class_A["Nome Oggetto"].tolist(),
    )
    pca_plots.plot_eigenvalue(pca_A)
    pca_A.reduction(2)  # Retain 2 PCs

    pca_B = PrincipalComponentAnalysis()
    pca_B.fit(
        X_B,
        variables_names=variables,
        objects_names=df_class_B["Nome Oggetto"].tolist(),
    )
    pca_plots.plot_eigenvalue(pca_B)
    pca_B.reduction(2)  # Retain 2 PCs

    # --- Visualization ---
    # (You'll need to adjust plotting to handle two PCA models)
    pca_plots.plot_scores(pca_A)  # Plot scores for Class A
    pca_plots.plot_scores(pca_B)  # Plot scores for Class B

    # ... (Add more plots to compare the PCA models)

    # --- Example: Classifying a new data point ---
    new_data_point = np.array([[2.5, 2.8, 4.2, 5.7, 3.1, 7.9]])

    # Project the new data point onto the PC space of each class
    new_data_projection_A = pca_A.transform(new_data_point)
    new_data_projection_B = pca_B.transform(new_data_point)

    # Calculate the distance to the center of each class in PC space
    distance_to_A = np.linalg.norm(new_data_projection_A - pca_A.T.mean(axis=0))
    distance_to_B = np.linalg.norm(new_data_projection_B - pca_B.T.mean(axis=0))

    # --- Visualization with Class Separation ---
    pca_models = [pca_A, pca_B]  # List of PCA models
    class_labels = ["Class A", "Class B"]  # Corresponding class labels
    pca_plots.plot_classes_pca(pca_models, class_labels, new_data_point=new_data_point)

    print(f"Distance to Class A in PC space: {distance_to_A:.2f}")
    print(f"Distance to Class B in PC space: {distance_to_B:.2f}")

    # Simple classification based on distance
    if distance_to_A < distance_to_B:
        print("New data point is closer to Class A")
    else:
        print("New data point is closer to Class B")


if __name__ == "__main__":
    main()
