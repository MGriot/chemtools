import numpy as np


def reorder_array(X):
    """Function for ordered an array from highest to lowest values.

    Args:
        X (_type_, optional): _description_. Defaults to V.
        Y (_type_, optional): _description_. Defaults to np.arange(0, V.size, 1).

    Returns:
        _type_: _description_
    """
    Y = np.arange(0, X.size, 1)
    new_list = list(zip(X, Y))
    # new_list = []
    # for x, y in zip(X, Y):
    # new_list.append((x, y))

    new_list = sorted(new_list, key=lambda element: element[0])

    X, Y = [], []

    for x, y in new_list:
        X.append(x)
        Y.append(y)
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