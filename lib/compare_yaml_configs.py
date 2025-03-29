def compare_yaml_configs(config1, config2):
    """
    Compare two YAML config files and raise an error if they do not match.

    Parameters:
        file1 (str): Path to the first YAML file.
        file2 (str): Path to the second YAML file.

    Raises:
        ValueError: If there is a mismatch in dataset_dimension, dataset_name, or dataset_type.
    """

    keys_to_compare = ['dataset_dimension', 'dataset_name', 'dataset_type', 'prep_data_incl_past_residuals']
    
    mismatches = []
    
    for key in keys_to_compare:
        if config1.get(key) != config2.get(key):
            mismatches.append(f"Mismatch in '{key}': '{config1.get(key)}' != '{config2.get(key)}'")

    if mismatches:
        error_message = "\n".join(mismatches)
        raise ValueError(f"Configuration mismatch detected:\n{error_message}")

    print("Configs Match: ✅")