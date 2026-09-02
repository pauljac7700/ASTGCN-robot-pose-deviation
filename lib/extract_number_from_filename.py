"""Pull the graph variant number out of an adjacency matrix filename.

The graph topology is selected by filename, and the evaluation code needs the
variant number to know which node holds the residual.
"""

import re

def extract_number_from_filename(filename):
    """
    Extracts the first number following 'matrix_' in the filename and raises an error if not found.

    For example:
        'adjacency_matrix_1.npz'      -> returns 1
        'adjacency_matrix_2_3D.npz'     -> returns 2

    Parameters:
        filename (str): The filename to process.

    Returns:
        int: The extracted number.

    Raises:
        ValueError: If no number is found after 'matrix_' in the filename.
    """
    match = re.search(r'(?<=matrix_)(\d+)', filename)
    if not match:
        raise ValueError(f"No number found in filename: {filename}")
    return int(match.group(1))
