"""Read the node count from the configured adjacency matrix.

The model's input dimensions follow from the graph, so this is resolved once at
startup rather than being duplicated in every config.
"""

import numpy as np

def get_adjacency_matrix_size(config) -> int:
    """
    Load an adjacency matrix from a file specified in a configuration dictionary
    and return the size (number of columns or rows) of the square matrix.

    Parameters:
        config (dict): A dictionary containing the key 'adjacency_matrix_file'
                       which is the path to the .npy file holding the adjacency matrix.

    Returns:
        int: The size (number of columns or rows) of the square adjacency matrix.

    Raises:
        ValueError: If the 'adjacency_matrix_file' key is missing in the config,
                    or if the loaded matrix is not square.
    """
    # Get the adjacency matrix file path from the config.
    adjacency_matrix_file = config.get("adjacency_matrix_file")
    if not adjacency_matrix_file:
        raise ValueError("The config file does not contain 'adjacency_matrix_file'.")

    # Load the adjacency matrix.
    matrix = np.load(adjacency_matrix_file)

    # Ensure the matrix is square (symmetric).
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The loaded matrix is not square.")
    
    # Return the number of columns (or rows).
    return matrix.shape[1]
