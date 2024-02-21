import os

def get_num_files(path):
    """
    Walks through a directory and prints its contents
    :param path: Path to directory.
    """

    for path, _, filenames in os.walk(path):
        print(f"There are {len(filenames)} images in '{path}'.")
