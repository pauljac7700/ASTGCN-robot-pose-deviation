import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.preprocessing import StandardScaler
from lib.extract_number_from_filename import extract_number_from_filename
from lib.get_adjacency_matrix_size import get_adjacency_matrix_size
from create_data_sequences.create_sequences_1_3_6_7 import create_sequences_1_3_6_7

def prepare_different_test_data(config):
    # Check that test_dataset_type is provided and is different from dataset_type.
    if 'test_dataset_type' not in config or config['test_dataset_type'] == config['dataset_type']:
        raise ValueError("test_dataset_type must be provided and cannot be the same as dataset_type.")

    # Load the YAML file that maps dataset locations.
    with open(config['locate_data_file'], 'r') as r:
        locate_data = yaml.safe_load(r)

    print("Dataset Dimension:", config['dataset_dimension'])
    print("Dataset Name:", config['dataset_name'])
    print("Training Dataset Type:", config['dataset_type'])

    test_dataset_type = config['test_dataset_type']
    print("Test Dataset Type:", test_dataset_type)

    num_joints = config['num_joints']
    dataset_dimension = config['dataset_dimension']

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

    # Extract dataset file name from YAML using test_dataset_type.
    grid_file = locate_data.get(config['dataset_dimension'], {}) \
                           .get(config['dataset_name'], {}) \
                           .get(test_dataset_type, None)
    if grid_file is None:
        raise ValueError("Dataset file not found in locate_data for the given dimension, name, and test dataset type.")
    
    # Construct full dataset path.
    datapath_to_dataset = f"data/{config['dataset_dimension']}_datasets/{grid_file}"
    df = pd.read_csv(datapath_to_dataset)

    # Drop unnecessary columns and remove missing values.
    df = df.drop(columns=['step_order'], errors='ignore')
    if df.isnull().values.any():
        df = df.dropna()

    # Define input features and residual variables.
    residual_variables = config.get('residual_variables', {}).get(config['dataset_dimension'], [])
    joint_features = [f'joint_{i}' for i in range(1, num_joints + 1)]
    target_pose_features = config.get('target_pose_variables', {}).get(config['dataset_dimension'], [])

    # Determine input features based on past residual inclusion.
    if prep_data_incl_past_residuals == 'nr':
        input_features = joint_features + target_pose_features
    else:  # 'wr'
        input_features = joint_features + target_pose_features + residual_variables

    # Ensure required columns are present.
    required_columns = input_features + residual_variables
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns in dataset: {missing_columns}")
        return

    # Extract inputs and residuals.
    inputs_df = df[input_features]
    residual_df = df[residual_variables]

    # Load the pre-fitted scalers.
    scalers = joblib.load(config['scalers_file'])
    input_scalers = scalers['input_scalers']
    residual_scalers = scalers['residual_scalers']

    # Transform the input features using the saved scalers.
    for feature in input_features:
        if feature in input_scalers:
            scaler = input_scalers[feature]
            inputs_df.loc[:, feature] = scaler.transform(inputs_df[[feature]].values).astype(np.float64).squeeze()
        else:
            print(f"Scaler for feature {feature} not found.")

    # Transform the residual variables.
    for residual_var in residual_variables:
        if residual_var in residual_scalers:
            scaler = residual_scalers[residual_var]
            residual_df.loc[:, residual_var] = scaler.transform(residual_df[[residual_var]].values).astype(np.float64).squeeze()
        else:
            print(f"Scaler for residual variable {residual_var} not found.")

    # For a big test dataset, use the entire dataset (no train/val split).
    inputs_test_df = inputs_df.reset_index(drop=True)
    residual_test_df = residual_df.reset_index(drop=True)

    # Create sequences using the appropriate function.
    if graph_nr in [1, 3, 6, 7]:
        inputs_test, residuals_test = create_sequences_1_3_6_7(inputs_df, residual_df, config, dataset_dimension, num_nodes, joint_features, target_pose_features, residual_variables, num_joints)
    else:
        raise ValueError("Unsupported configuration for sequence creation.")

    # Save the big test dataset.
    output_file = f'data/different_test_data_{graph_nr}_{prep_data_incl_past_residuals}.npz'
    np.savez_compressed(output_file, inputs=inputs_test, residuals=residuals_test)
    print(f"Big test dataset saved to {output_file}")

if __name__ == "__main__":
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)
    prepare_different_test_data(config)
