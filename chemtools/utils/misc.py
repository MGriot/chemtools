import inspect
import warnings
from typing import Any, Optional
from datetime import date, datetime
import numpy as np


def get_variable_name(variable: Any) -> Optional[str]:
    """
    Attempts to find the name of a variable in the caller's scope.

    Warning: Relies on introspection. May not work reliably in all contexts.

    Args:
        variable: The variable instance whose name is sought.

    Returns:
        The name (str) of the variable if found, otherwise None.
    """
    try:
        stack = inspect.stack()
        if len(stack) > 2:
            caller_of_caller_frame_record = stack[2]
            caller_of_caller_frame = caller_of_caller_frame_record.frame
            caller_locals = caller_of_caller_frame.f_locals
            for name, value in caller_locals.items():
                if value is variable:
                    del stack
                    del caller_of_caller_frame_record
                    del caller_of_caller_frame
                    return name
        else:
            warnings.warn(
                "get_variable_name: Call stack not deep enough.", stacklevel=2
            )
    except Exception as e:
        warnings.warn(
            f"get_variable_name: Error during introspection: {e}", stacklevel=2
        )
        return None
    finally:
        try:
            del stack
        except NameError:
            pass
        try:
            del caller_of_caller_frame_record
        except NameError:
            pass
        try:
            del caller_of_caller_frame
        except NameError:
            pass
    return None


def make_standards(x_min, x_max, n_standard=7, decimal=1):
    """
    Generates an array of equidistant points for a calibration line.

    Args:
        x_min (float): Minimum value of the domain.
        x_max (float): Maximum value of the domain.
        n_standard (int, optional): Number of standards to create. Defaults to 7.
        decimal (int, optional): Number of decimal places to round to. Defaults to 1.

    Returns:
        np.ndarray: An array of standard concentrations.
    """
    return np.around(np.linspace(x_min, x_max, n_standard), decimal)


def when_date():
    """
    Gets the current date as a string in YYYY-MM-DD format.

    Returns:
        string: Today's date.
    """
    now = datetime.now()
    return str(now.strftime("%Y-%m-%d"))
