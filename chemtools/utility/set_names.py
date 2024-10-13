def set_variables_names(variables_names, X):
    """Sets the names of the variables."""
    if variables_names is not None:
        assert (
            len(variables_names) == X.shape[1]
        ), "The length of the list of variables names is different from the number of variables."
        return variables_names
    else:
        return np.array([f"V{i+1}" for i in range(X.shape[1])])


def set_objects_names(objects_names, X):
    """Sets the names of the objects."""
    if objects_names is not None:
        assert (
            len(objects_names) == X.shape[0]
        ), "The length of the list of objects names is different from the number of objects."
        return objects_names
    else:
        return np.array([f"Obj{i+1}" for i in range(X.shape[0])])
