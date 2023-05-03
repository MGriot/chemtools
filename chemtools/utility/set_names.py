def set_variables_names(X, variables_names):
    """Set variable names for columns in a 2D array.

    Args:
        X: A 2D array.
        variables_names: A list of variable names or None.

    Returns:
        A list of variable names.
    """
    if variables_names is None:
        n = X.shape[1]
        variables = [str(i) for i in range(n)]
    else:
        variables = variables_names
    return variables

def set_objects_names(X, objects_names):
    """Set object names for rows in a 2D array.

    Args:
        X: A 2D array.
        objects_names: A list of object names or None.

    Returns:
        A list of object names.
    """
    if objects_names is None:
        n = X.shape[0]
        objects = [str(i) for i in range(n)]
    else:
        objects = objects_names
    return objects