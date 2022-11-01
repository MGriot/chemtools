import os


def directory_creator(directory):
    """Function that given the name of a directory (or path to a directory) allows you to check if it exists, alternatively it is created.

    Args:
        directory (string): name or path of the directory you want to create.
    """
    if not os.path.exists(directory):
        print(f"I am creating the output directory, and is {directory}.")
        os.makedirs(directory)
    else:
        print(f"The output directory already exists, and is: {directory}.")
