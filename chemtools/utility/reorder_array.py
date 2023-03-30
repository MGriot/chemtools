import numpy as np


def reorder_array(X):
    """
    Ordina un array dal valore più alto al più basso e restituisce l'array ordinato e gli indici originali.

    Args:
        X (np.ndarray): Array di input da ordinare.

    Returns:
        tuple: Una tupla contenente l'array ordinato e gli indici originali.
    """
    Y = np.arange(0, X.size, 1)
    sorted_indices = np.argsort(X)
    X = X[sorted_indices]
    Y = Y[sorted_indices]
    return np.flip(X), np.flip(Y)


def sort_arrays(x, y):
    """
    Ordina due array in base ai valori del primo.
    
    Parametri:
    x : array
        Array da utilizzare come base per l'ordinamento
    y : array
        Secondo array da ordinare
    
    Restituisce:
    sorted_x : array
        Primo array ordinato
    sorted_y : array
        Secondo array ordinato in base all'ordinamento del primo
    """
    # Creare un array di indici ordinati in base ai valori di x
    sorted_indices = np.argsort(x)
    # Utilizzare gli indici ordinati per riordinare entrambi gli array
    sorted_x = x[sorted_indices]
    sorted_y = y[sorted_indices]
    return sorted_x, sorted_y