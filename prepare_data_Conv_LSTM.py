# prepare_data_Conv_LSTM.py

import numpy as np
import yaml
import os
import joblib
from lib.compare_yaml_configs import compare_yaml_configs
from lib.extract_number_from_filename import extract_number_from_filename

def prepare_data(config, config_ASTGCN):
    """
    Transforms ASTGCN-prepared data into a format suitable for ConvLSTM.

    Parameters:
    - config (dict): Configuration parameters loaded from the YAML file.
    - config_ASTGCN (dict): Configuration parameters loaded from the YAML file.
    """
    if config['prep_data_incl_past_residuals']: 
        prep_data_incl_past_residuals = 'wr'
        print("Including past residuals in input features.")
    else:   
        prep_data_incl_past_residuals = 'nr'
        print("Excluding past residuals from input features.")
    
    # Load previously prepared ASTGCN datasets
    graph_nr = extract_number_from_filename(config_ASTGCN['adjacency_matrix_file'])
    train_data = np.load('data/train_data_' + str(graph_nr) + '_' + prep_data_incl_past_residuals + '.npz')
    val_data = np.load('data/val_data_' + str(graph_nr) + '_' + prep_data_incl_past_residuals + '.npz')
    test_data = np.load('data/test_data_' + str(graph_nr) + '_' + prep_data_incl_past_residuals + '.npz')

    inputs_train = train_data['inputs']   # Shape: (N, 8, 6, T)
    residuals_train = train_data['residuals'] # Shape: (N, num_residuals)
    inputs_val = val_data['inputs']
    residuals_val = val_data['residuals']
    inputs_test = test_data['inputs']
    residuals_test = test_data['residuals']

    # The ASTGCN data shape: (N, num_nodes=8, in_channels=6, T)
    # For ConvLSTM we want: (N, T, C, H, W)
    # We can treat num_nodes=8 as height and set width=1 since it's essentially 1D.
    # So we do:
    # Transpose from (N, 8, 6, T) to (N, T, 6, 8) and then expand dims to (N, T, 6, 8, 1)

    def reshape_for_convlstm(inputs):
        inputs_transposed = np.transpose(inputs, (0, 3, 2, 1))  # (N, T, C=6, H=8)
        inputs_reshaped = np.expand_dims(inputs_transposed, axis=-1)  # (N, T, C=6, H=8, W=1)
        return inputs_reshaped

    inputs_train_conv = reshape_for_convlstm(inputs_train)
    inputs_val_conv = reshape_for_convlstm(inputs_val)
    inputs_test_conv = reshape_for_convlstm(inputs_test)

    # Targets remain the same shape (N, num_targets)

    # Save the prepared data for ConvLSTM
    np.savez_compressed(f'data/convlstm_train_{prep_data_incl_past_residuals}.npz', inputs=inputs_train_conv, residuals=residuals_train)
    np.savez_compressed(f'data/convlstm_val_{prep_data_incl_past_residuals}.npz', inputs=inputs_val_conv, residuals=residuals_val)
    np.savez_compressed(f'data/convlstm_test_{prep_data_incl_past_residuals}.npz',inputs=inputs_test_conv, residuals=residuals_test)
    print("ConvLSTM datasets saved.")


if __name__ == "__main__":
    # Load configuration
    with open('config_Conv_LSTM.yaml', 'r') as f:
        config = yaml.safe_load(f)

    with open('config_ASTGCN.yaml', 'r') as f:
        config_ASTGCN = yaml.safe_load(f)
    compare_yaml_configs(config, config_ASTGCN)

    prepare_data(config, config_ASTGCN)
