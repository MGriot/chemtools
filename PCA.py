import pandas as pd
import numpy as np
from chemtools.exploration import PrincipalComponentAnalysis

# Creazione di un database con nomi degli oggetti e titoli delle variabili
data = {
    "Nome Oggetto": ["Oggetto1", "Oggetto2", "Oggetto3", "Oggetto4"],
    "Variabile1": [1.2, 2.3, 3.1, 4.5],
    "Variabile2": [3.4, 1.2, 4.5, 2.3],
    "Variabile3": [5.6, 3.4, 2.1, 1.2],
    "Variabile4": [7.8, 6.5, 3.2, 4.1],
    "Variabile5": [2.3, 4.6, 1.8, 3.9],
    "Variabile6": [9.0, 8.1, 7.2, 6.3],
}

# Creazione del DataFrame
df = pd.DataFrame(data)

# Visualizzazione del DataFrame
print("Database creato:")
print(df)

# Preparazione dei dati per la PCA (escludendo la colonna 'Nome Oggetto')
X = df.drop("Nome Oggetto", axis=1)
X = X.to_numpy()
pca = PrincipalComponentAnalysis(X)
pca.fit()
# pca.plot_correlation_matrix()
pca.plot_correlation_matrix()
pca.plot_eigenvalue()
pca.reduction(int(input("how many PC?")))
pca.plot_loadings()
pca.plot_biplot()
pca.plot_scores()
