import numpy as np
import pandas as pd

# import polars as pl


def handle_nan_values(X, method="mean", axis=None, index=None):
    """
    Handle NaN values in the matrix by replacing or removing them.

    Args:
        X (numpy.ndarray, pd.DataFrame, pl.DataFrame): Input matrix or dataframe.
        method (str): Method to replace NaN values. Can be 'zero', 'mean', 'median', 'remove'.
        axis (int, optional): Axis along which to apply the method. 0 for columns, 1 for rows. If None, apply to all.
        index (int, optional): Specific row or column index to apply the method. If None, apply to all.

    Returns:
        numpy.ndarray, pd.DataFrame, pl.DataFrame: Matrix or dataframe with NaN values handled.

    Examples:
        # Apply to a specific column
        handle_nan_values(X, method="mean", axis=0, index=2)  # Apply to column 2

        # Apply to a specific row
        handle_nan_values(X, method="mean", axis=1, index=3)  # Apply to row 3

        # Apply to multiple columns
        handle_nan_values(X, method="mean", axis=0, index=[1, 3, 5])  # Apply to columns 1, 3, and 5

        # Apply to multiple rows
        handle_nan_values(X, method="mean", axis=1, index=[0, 2, 4])  # Apply to rows 0, 2, and 4

        # Apply to a range of columns
        handle_nan_values(X, method="mean", axis=0, index=slice(1, 4))  # Apply to columns 1, 2, and 3

        # Apply to a range of rows
        handle_nan_values(X, method="mean", axis=1, index=slice(0, 3))  # Apply to rows 0, 1, and 2

        # Apply using boolean conditions for columns
        col_mask = [True, False, True, False]  # Apply to columns 0 and 2
        handle_nan_values(X, method="mean", axis=0, index=col_mask)

        # Apply using boolean conditions for rows
        row_mask = [False, True, False, True]  # Apply to rows 1 and 3
        handle_nan_values(X, method="mean", axis=1, index=row_mask)
    """
    # is_dataframe = isinstance(X, (pd.DataFrame, pl.DataFrame))
    is_dataframe = isinstance(X, (pd.DataFrame))
    if is_dataframe:
        X_np = X.to_numpy()
    else:
        X_np = X

    X_handled = X_np.copy()

    if method == "zero":
        # Replace NaNs with 0
        if index is not None:
            if axis == 0:
                X_handled[:, index] = np.nan_to_num(X_np[:, index], nan=0.0)
            elif axis == 1:
                X_handled[index, :] = np.nan_to_num(X_np[index, :], nan=0.0)
        else:
            X_handled = np.nan_to_num(X_np, nan=0.0)

    elif method in ["mean", "median"]:
        # Replace NaNs with column/row mean or median
        if axis == 0:
            stat = (
                np.nanmean(X_np, axis=0)
                if method == "mean"
                else np.nanmedian(X_np, axis=0)
            )
            if index is not None:
                X_handled[:, index] = np.where(
                    np.isnan(X_np[:, index]), stat[index], X_np[:, index]
                )
            else:
                inds = np.where(np.isnan(X_np))
                X_handled[inds] = np.take(stat, inds[1])
        elif axis == 1:
            stat = (
                np.nanmean(X_np, axis=1)
                if method == "mean"
                else np.nanmedian(X_np, axis=1)
            )
            if index is not None:
                X_handled[index, :] = np.where(
                    np.isnan(X_np[index, :]), stat[index], X_np[index, :]
                )
            else:
                inds = np.where(np.isnan(X_np))
                X_handled[inds] = np.take(stat, inds[0])
        else:
            # Apply to all
            if method == "mean":
                col_stat = np.nanmean(X_np, axis=0)
                row_stat = np.nanmean(X_np, axis=1)
            else:
                col_stat = np.nanmedian(X_np, axis=0)
                row_stat = np.nanmedian(X_np, axis=1)
            inds = np.where(np.isnan(X_np))
            for i, j in zip(*inds):
                X_handled[i, j] = col_stat[j] if np.isnan(row_stat[i]) else row_stat[i]

    elif method == "remove":
        # Remove rows or columns with NaNs
        if axis is None or axis == 0:
            X_handled = X_np[~np.isnan(X_np).any(axis=1)]
        else:
            X_handled = X_np[:, ~np.isnan(X_np).any(axis=0)]

    else:
        raise ValueError(
            "Invalid method. Choose from 'zero', 'mean', 'median', 'remove'."
        )

    if is_dataframe:
        return (
            pd.DataFrame(X_handled, columns=X.columns)
            if isinstance(X, pd.DataFrame)
            else pl.DataFrame(X_handled)
        )
    else:
        return X_handled


def one_hot_encode(data, column, axis=0, labels=None):
    """
    One-hot encode a specific column or row in the dataframe or numpy array.

    Args:
        data (pd.DataFrame, pl.DataFrame, np.ndarray): Input dataframe or numpy array.
        column (str or int): Column name or index to one-hot encode.
        axis (int, optional): Axis along which to apply the encoding. 0 for columns, 1 for rows. Default is 0.
        labels (list, optional): List of labels for the new columns or rows. Default is None.

    Returns:
        np.ndarray: Array with one-hot encoded variables.
        list (optional): Updated labels for the new columns or rows if labels are provided.
    """
    # if isinstance(data, pl.DataFrame):
    #    data = data.to_pandas()

    if isinstance(data, pd.DataFrame):
        if axis == 0:
            # One-hot encode a column
            dummies = pd.get_dummies(data[column], prefix=column)
            data = pd.concat([data.drop(column, axis=1), dummies], axis=1)
            updated_labels = list(data.columns)
            return data.to_numpy(), updated_labels
        elif axis == 1:
            # One-hot encode a row
            data = data.T
            dummies = pd.get_dummies(data[column], prefix=column)
            data = pd.concat([data.drop(column, axis=1), dummies], axis=1).T
            updated_labels = list(data.index)
            return data.to_numpy(), updated_labels
        else:
            raise ValueError("Invalid axis. Choose 0 for columns or 1 for rows.")

    elif isinstance(data, np.ndarray):
        if axis == 0:
            unique_values = np.unique(data[:, column])
            encoded_columns = [
                (data[:, column] == value).astype(int).reshape(-1, 1)
                for value in unique_values
            ]
            data = np.delete(data, column, axis=1)
            data = np.hstack([data] + encoded_columns)
            if labels is not None:
                updated_labels = (
                    labels[:column]
                    + labels[column + 1 :]
                    + [f"{labels[column]}_{value}" for value in unique_values]
                )
                return data, updated_labels
            else:
                return data
        elif axis == 1:
            unique_values = np.unique(data[column, :])
            encoded_rows = [
                (data[column, :] == value).astype(int).reshape(1, -1)
                for value in unique_values
            ]
            data = np.delete(data, column, axis=0)
            data = np.vstack([data] + encoded_rows)
            if labels is not None:
                updated_labels = (
                    labels[:column]
                    + labels[column + 1 :]
                    + [f"{labels[column]}_{value}" for value in unique_values]
                )
                return data, updated_labels
            else:
                return data
        else:
            raise ValueError("Invalid axis. Choose 0 for columns or 1 for rows.")

    else:
        raise TypeError(
            "Input data must be a pandas DataFrame, polars DataFrame, or numpy array."
        )


if __name__ == "__main__":
    # Creazione di un DataFrame di esempio con valori NaN
    data = {"A": [1, 2, np.nan, 4], "B": [5, np.nan, np.nan, 8], "C": [9, 10, 11, 12]}
    df = pd.DataFrame(data)
    print("DataFrame originale:")
    print(type(df))
    print(df)

    # Test della funzione handle_nan_values
    df_handled = handle_nan_values(df, method="mean", axis=0)
    print("\nDataFrame dopo handle_nan_values (mean):")
    print(type(df_handled))
    print(df_handled)

    # Creazione di un DataFrame di esempio per one_hot_encode
    data2 = {"Category": ["A", "B", "A", "C"], "Value": [10, 20, 30, 40]}
    df2 = pd.DataFrame(data2)
    print("\nDataFrame originale per one_hot_encode:")
    print(type(df2))
    print(df2)

    # Test della funzione one_hot_encode
    df_encoded, updated_labels = one_hot_encode(df2, column="Category", axis=0)
    df_one_hot_encode = pd.DataFrame(df_encoded, columns=updated_labels)
    print("\nDataFrame dopo one_hot_encode:")
    print(type(df_one_hot_encode))
    print(df_one_hot_encode)

    print("\nNumpy")
    # Creazione di una matrice NumPy di esempio con valori NaN
    matrix = np.array([[1, 2, np.nan, 4], [5, np.nan, np.nan, 8], [9, 10, 11, 12]])
    print("Matrice originale:")
    print(matrix)

    # Test della funzione handle_nan_values
    matrix_handled = handle_nan_values(matrix, method="mean", axis=0)
    print("\nMatrice dopo handle_nan_values (mean):")
    print(matrix_handled)

    # Creazione di una matrice NumPy di esempio per one_hot_encode
    matrix2 = np.array([["A", 10], ["B", 20], ["A", 30], ["C", 40]])
    print("\nMatrice originale per one_hot_encode:")
    print(matrix2)

    # Test della funzione one_hot_encode
    matrix_encoded, updated_labels = one_hot_encode(
        matrix2, column=0, axis=0, labels=["CAT", "val"]
    )
    print("\nMatrice dopo one_hot_encode:")
    print(matrix_encoded)
    print("Etichette aggiornate:")
    print(updated_labels)
