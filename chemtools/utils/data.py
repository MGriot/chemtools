"""
This module contains utility functions for setting and checking data.
"""

import pandas as pd


def check_variable_type(x):
    """
    Checks if a variable is a pandas DataFrame and prints the result.
    """
    if isinstance(x, pd.DataFrame):
        print(f"{type(x)} is a Pandas DataFrame")


def set_objects_names(objects_names, X):
    """
    Set the names of the objects.

    Args:
        objects_names (list or None): List of object names.
        X (np.ndarray or pd.DataFrame): Data matrix.

    Returns:
        list: List of object names.
    """
    if objects_names is None:
        if isinstance(X, pd.DataFrame):
            objects = X.index.tolist()
        else:
            objects = [f"Obj{i}" for i in range(X.shape[0])]
    else:
        objects = objects_names
    return objects


def set_variables_names(variables_names, X):
    """
    Set the names of the variables.

    Args:
        variables_names (list or None): List of variable names.
        X (np.ndarray or pd.DataFrame): Data matrix.

    Returns:
        list: List of variable names.
    """
    if variables_names is None:
        if isinstance(X, pd.DataFrame):
            variables = X.columns.tolist()
        else:
            variables = [f"X{i}" for i in range(X.shape[1])]
    else:
        variables = variables_names
    return variables


def initialize_names_and_counts(x, variables_names, objects_names):
    """
    Initializes variable and object names and counts.

    Args:
        x (np.ndarray or pd.DataFrame): Data matrix.
        variables_names (list or None): List of variable names.
        objects_names (list or None): List of object names.

    Returns:
        tuple: Tuple containing variables, objects, n_variables, and n_objects.
    """
    variables = set_variables_names(variables_names, x)
    objects = set_objects_names(objects_names, x)
    n_variables = x.shape[1]
    n_objects = x.shape[0]
    return variables, objects, n_variables, n_objects
