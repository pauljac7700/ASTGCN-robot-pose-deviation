"""Walk a directory and return its structure, for run bookkeeping."""

import os

def print_folder_structure(root_path: str, max_depth: int = None, prefix: str = ""):
    """
    Recursively prints the folder structure starting from the given root path.

    Args:
        root_path (str): The root directory to visualize.
        max_depth (int, optional): Max depth of recursion. If None, no limit.
        prefix (str): Internal parameter for indentation (used during recursion).
    """
    if not os.path.isdir(root_path):
        raise NotADirectoryError(f"{root_path} is not a valid directory.")

    def _walk(path, depth, prefix):
        if max_depth is not None and depth > max_depth:
            return

        items = sorted(os.listdir(path))
        for idx, name in enumerate(items):
            full_path = os.path.join(path, name)
            connector = "└── " if idx == len(items) - 1 else "├── "
            print(prefix + connector + name)

            if os.path.isdir(full_path):
                extension = "    " if idx == len(items) - 1 else "│   "
                _walk(full_path, depth + 1, prefix + extension)

    print(root_path)
    _walk(root_path, depth=1, prefix=prefix)

