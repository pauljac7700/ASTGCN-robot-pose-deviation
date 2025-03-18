import re

def extract_number_from_filename(filename):
    """
    Extracts the number at the end of a filename before the extension.

    Parameters:
        filename (str): The filename to process (e.g., 'filename1.npz').

    Returns:
        int or None: The extracted number if found, otherwise None.
    """
    match = re.search(r'(\d+)(?=\.\w+$)', filename)
    return int(match.group(1)) if match else None

