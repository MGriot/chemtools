import os


def directory_creator(directory):
    """
    Function that given the name of a directory (or path to a directory)
    checks if it exists, and if not, creates it.

    Args:
        directory (string): name or path of the directory you want to create.
    """
    if not os.path.exists(directory):
        print(f"Creating the output directory: {directory}.")
        os.makedirs(directory)
    else:
        print(f"The output directory already exists: {directory}.")