# Author: Leon Kleinschmidt

import os


def generate_folder(folders):
    """
    Creates multiple folders based on the provided list.

    Parameters:
    folders (list): A list of folder names to be created.

    """
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
