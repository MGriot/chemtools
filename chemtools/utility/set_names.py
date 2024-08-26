def set_variables_names(variables_names, X):
    if variables_names is None:
        return [f"Var{i+1}" for i in range(X.shape[1])]
    return variables_names


def set_objects_names(objects_names, X):
    if objects_names is None:
        return [f"Obj{i+1}" for i in range(X.shape[0])]
    return objects_names
