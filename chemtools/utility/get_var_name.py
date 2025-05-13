import inspect
import warnings
from typing import Any, Optional


def get_variable_name(variable: Any) -> Optional[str]:
    """
    Attempts to find the name of a variable in the caller's scope.

    This function uses introspection to look up the call stack and find a variable
    name in the calling function's parent frame (the frame that called the
    calling function) that points to the exact same object instance as `variable`.

    Warning: Relies on inspect module and stack frame analysis. May not work
             reliably in all execution contexts (e.g., optimized code, complex
             frameworks, decorated functions) or if the passed `variable` is
             the result of an expression. Returns None if the name cannot be
             reliably determined.

    Args:
        variable: The variable instance whose name is sought in the caller's scope.

    Returns:
        The name (str) of the variable in the caller's scope if found,
        otherwise None.
    """
    try:
        # Get the full stack. stack[0] is this function's frame.
        stack = inspect.stack()

        # stack[1] is the frame that called get_variable_name (e.g., fit).
        # stack[2] is the frame that called the function in stack[1]
        # (e.g., the main script or function calling fit).
        # We need to search the local variables of the frame at stack[2].
        if len(stack) > 2:
            # Get the frame record for the caller of our caller
            caller_of_caller_frame_record = stack[2]
            caller_of_caller_frame = caller_of_caller_frame_record.frame

            # Get the local variables dictionary of that frame
            caller_locals = caller_of_caller_frame.f_locals

            # Iterate through the caller's local variables
            for name, value in caller_locals.items():
                # Check if the object identity matches
                if value is variable:
                    # Found a name pointing to the same object
                    # Clean up frame references to potentially help GC
                    del stack
                    del caller_of_caller_frame_record
                    del caller_of_caller_frame
                    return name
        else:
            # Stack is not deep enough (e.g., called from global scope directly?)
            warnings.warn(
                "get_variable_name: Call stack not deep enough to determine "
                "caller's variable name.",
                stacklevel=2,
            )

    except Exception as e:
        # Catch potential errors during introspection
        warnings.warn(
            f"get_variable_name: Error during introspection: {e}", stacklevel=2
        )
        return None
    finally:
        # Ensure cleanup even if errors occur or name not found
        # Note: 'del stack' etc. in the try block handles the success case.
        # Using try/except NameError here in case they weren't assigned.
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

    # If the loop completes without finding a match
    # warnings.warn(
    #    f"get_variable_name: Could not find variable name for object "
    #    f"of type {type(variable).__name__}. It might be an expression result.",
    #    stacklevel=2
    # ) # Optional: Add warning if not found
    return None


# --- Example Usage ---
if __name__ == "__main__":

    def model_fit_method(x_data, y_data):
        """Simulates a method like OLSRegression.fit"""
        print("\n--- Inside model_fit_method ---")
        # Attempt to get the names of the variables passed to this function
        # from the scope where model_fit_method was called.
        retrieved_y_name = get_variable_name(y_data)
        retrieved_x_name = get_variable_name(x_data)

        print(f"Received y_data object: id={id(y_data)}")
        print(f"Attempted to retrieve original y name: '{retrieved_y_name}'")

        print(f"Received x_data object: id={id(x_data)}")
        print(f"Attempted to retrieve original x name: '{retrieved_x_name}'")
        print("-----------------------------")


    # --- Test Scenarios ---

    print("--- Scenario 1: Simple Call ---")
    y_actual = [1, 2, 3, 4]
    X_predictors = [[1], [2], [3], [4]]
    print(f"Calling with variable 'y_actual' (id={id(y_actual)})")
    print(f"Calling with variable 'X_predictors' (id={id(X_predictors)})")
    model_fit_method(X_predictors, y_actual)


    print("\n--- Scenario 2: Using an Expression ---")
    import numpy as np

    original_array = np.array([10, 20, 30])
    print(
        f"Calling with 'original_array * 2' (id={id(original_array * 2)})"
    )  # Result is a NEW array
    model_fit_method(
        X_predictors, original_array * 2
    )  # Name retrieval for y_data will likely fail


    print("\n--- Scenario 3: Reused Variable Name ---")
    y_actual = [5, 6, 7]  # Reassign the name 'y_actual'
    print(f"Calling again with 'y_actual' (id={id(y_actual)})")
    model_fit_method(X_predictors, y_actual)


    print("\n--- Scenario 4: Multiple Names for Same Object ---")
    data_source = [100, 200, 300]
    ref1 = data_source
    ref2 = data_source
    print(
        f"Calling with 'ref1' (id={id(ref1)}) which points to same object as 'data_source'"
    )
    # The function might find 'ref1' or 'data_source' depending on dictionary iteration order
    model_fit_method(X_predictors, ref1)
