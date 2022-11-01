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
