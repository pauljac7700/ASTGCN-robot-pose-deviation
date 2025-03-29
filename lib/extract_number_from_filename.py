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
