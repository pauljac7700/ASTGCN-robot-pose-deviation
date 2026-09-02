"""Build train, validation and test tensors for the multi-target ASTGCN.

Reads the robot pose CSVs named in the config, constructs the graph sequences over
the kinematic chain, fits the feature scalers on the training split only, and writes
the arrays and scalers under ``data/``. Run this before any training script.
"""

# prep_data_multi.py

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lib.extract_number_from_filename import extract_number_from_filename
from lib.get_adjacency_matrix_size import get_adjacency_matrix_size
from create_data_sequences.create_all_sequences import create_all_sequences
import yaml

def prepare_data(config):
    # Load data
    with open(config['locate_data_file'], 'r') as r:
        locate_data = yaml.safe_load(r)

    print("Dataset Dimension:", config['dataset_dimension'])
    print("Dataset Name:", config['dataset_name'])
    print("Dataset Type:", config['dataset_type'])
    num_joints = config['num_joints']
    
    if config['prep_data_incl_past_residuals']:
        prep_data_incl_past_residuals = 'wr'
        print("Including past residuals in input features.")
    else:
        prep_data_incl_past_residuals = 'nr'
        print("Excluding past residuals from input features.")

    graph_nr = extract_number_from_filename(config['adjacency_matrix_file'])
    print(f"Graph number: {graph_nr}")

    # Get the number of nodes from the adjacency matrix
    num_nodes = get_adjacency_matrix_size(config)
    print(f"Number of nodes: {num_nodes}")

    # Extract dataset file path from YAML
    grid_file = locate_data.get(config['dataset_dimension'], {}).get(config['dataset_name'], {}).get(config['dataset_type'], None)

    # Construct full dataset path
    datapath_to_dataset = f"data/{config['dataset_dimension']}_datasets/{grid_file}"
    df = pd.read_csv(datapath_to_dataset)
    df=df.loc[1:]

    # Drop unnecessary columns and check for missing values
    df = df.drop(columns=['step_order'], errors='ignore')
    if df.isnull().values.any():
        df = df.dropna()

    # Define input features and residual variables
    residual_variables = config.get('residual_variables', {}).get(config['dataset_dimension'], [])
    joint_features = [f'joint_{i}' for i in range(1, num_joints + 1)]
    target_pose_features = config.get('target_pose_variables', {}).get(config['dataset_dimension'], [])

    # Exclude or include residuals from input features
    if prep_data_incl_past_residuals == 'nr':
        input_features = joint_features + target_pose_features
    elif prep_data_incl_past_residuals == 'wr':
        input_features = joint_features + target_pose_features + residual_variables

    # Ensure all required columns are present
    required_columns = input_features + residual_variables
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"The following required columns are missing from the dataset: {missing_columns}")
        return

    # Extract data for inputs and residuals
    inputs_df = df[input_features]
    residual_df = df[residual_variables]

    # Split the data into train, validation, and test sets
    test_size = config['test_size']
    val_size = config['val_size']
    train_size = 1 - test_size - val_size

    # First split off the test set
    inputs_train_val_df, inputs_test_df, residual_train_val_df, residual_test_df = train_test_split(
        inputs_df, residual_df, test_size=test_size, random_state=config['random_seed'], shuffle=False)

    # Then split train and validation sets without shuffling to prevent data leakage
    val_size_adjusted = val_size / (train_size + val_size)
    inputs_train_df, inputs_val_df, residual_train_df, residual_val_df = train_test_split(
        inputs_train_val_df, residual_train_val_df, test_size=val_size_adjusted, random_state=config['random_seed'], shuffle=False)

    print(f"Training set size: {inputs_train_df.shape[0]}")
    print(f"Validation set size: {inputs_val_df.shape[0]}")
    print(f"Test set size: {inputs_test_df.shape[0]}")

    # Scale the data
    # Fit scalers on training data only
    input_scalers = {}
    for feature in input_features:
        scaler = StandardScaler()
        # Fit on NumPy array
        scaler.fit(inputs_train_df[[feature]].values)
        # Transform the data
        inputs_train_df[feature] = scaler.transform(inputs_train_df[[feature]].values)
        inputs_val_df[feature] = scaler.transform(inputs_val_df[[feature]].values)
        inputs_test_df[feature] = scaler.transform(inputs_test_df[[feature]].values)
        # Save the scaler
        input_scalers[feature] = scaler

    # Scale the pose residual variables
    residual_scalers = {}
    for residual_var in residual_variables:
        scaler = StandardScaler()
        scaler.fit(residual_train_df[[residual_var]].values)
        residual_train_df[residual_var] = scaler.transform(residual_train_df[[residual_var]].values)
        residual_val_df[residual_var] = scaler.transform(residual_val_df[[residual_var]].values)
        residual_test_df[residual_var] = scaler.transform(residual_test_df[[residual_var]].values)
        residual_scalers[residual_var] = scaler

    # Save the scalers
    scalers = {'input_scalers': input_scalers, 'residual_scalers': residual_scalers}
    joblib.dump(scalers, config['scalers_file'])
    print(f"Scalers saved to {config['scalers_file']}")

    # Create sequences for training, validation, and test sets
    inputs_train, residuals_train, inputs_val, residuals_val, inputs_test, residuals_test = create_all_sequences(
        graph_nr, num_nodes, inputs_train_df, residual_train_df, inputs_val_df, residual_val_df, inputs_test_df, residual_test_df, config, joint_features, target_pose_features, residual_variables)
    
    # Save the datasets
    np.savez_compressed(f'data/train_data_{graph_nr}_{prep_data_incl_past_residuals}.npz', inputs=inputs_train, residuals=residuals_train)
    np.savez_compressed(f'data/val_data_{graph_nr}_{prep_data_incl_past_residuals}.npz', inputs=inputs_val, residuals=residuals_val)
    np.savez_compressed(f'data/test_data_{graph_nr}_{prep_data_incl_past_residuals}.npz', inputs=inputs_test, residuals=residuals_test)
    print("Datasets saved to disk.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_ASTGCN.yaml",
                        help="Path to the YAML configuration file (default: %(default)s).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    prepare_data(config)
