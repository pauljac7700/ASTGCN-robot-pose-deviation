# prep_data_multi_with_target_input.py

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lib.extract_number_from_filename import extract_number_from_filename
import yaml

def prepare_data(config):
    # Load data
    with open(config['locate_data_file'], 'r') as r:
        locate_data = yaml.safe_load(r)

    print("Dataset Dimension:", config['dataset_dimension'])
    print("Dataset Name:", config['dataset_name'])
    print("Dataset Type:", config['dataset_type'])
    num_joints = config['num_joints']
    dataset_dimension = config['dataset_dimension']
    if config['prep_data_incl_past_residuals']:
        prep_data_incl_past_residuals = 'wr'
    else:
        prep_data_incl_past_residuals = 'nr'

    graph_number = extract_number_from_filename(config['adjacency_matrix_file'])
    print(f"Graph number: {graph_number}")

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

    # Define input features and target variables
    residual_variables = config.get('residual_variables', {}).get(config['dataset_dimension'], [])
    joint_features = [f'joint_{i}' for i in range(1, num_joints + 1)]
    target_pose_features = config.get('target_pose_variables', {}).get(config['dataset_dimension'], [])

    # Include residuals from input features
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

    # Prepare sequences for inputs and residuals
    def create_sequences(inputs_df, residual_df):
        inputs_raw = inputs_df.values
        residuals_raw = residual_df.values  # Shape: (num_samples, num_residuals)

        len_input = config['model']['len_input']  # Sequence length for inputs (e.g., 10)
        num_for_predict = config['model']['num_for_predict']  # Should be set to 1
        num_nodes = config['model']['num_of_vertices']  # Should be 8 nodes
        in_channels = config['model'][f'in_channels_{dataset_dimension}']  # Should remain at 6

        # Ensure there are enough samples
        total_samples = inputs_raw.shape[0] - (len_input + 1) + 1
        if total_samples <= 0:
            print("Not enough data to create sequences.")
            return None, None

        inputs = []
        residuals = []

        # Get indices for features
        joint_indices = [inputs_df.columns.get_loc(f) for f in joint_features]
        target_pose_indices = [inputs_df.columns.get_loc(f) for f in target_pose_features]
        dif_indices = [inputs_df.columns.get_loc(f) for f in residual_variables]

        for i in range(total_samples):
            # Initialize input sequence with zeros
            input_sequence = np.zeros((len_input + 1, num_nodes, in_channels))

            for t in range(len_input + 1):  # Includes time t + 1
                idx = i + t

                # Nodes 0-5: Joint positions
                joint_positions = inputs_raw[idx, joint_indices]  # Shape: (6,)
                # Node 6: End-effector target poses
                end_effector_target_pose = inputs_raw[idx, target_pose_indices]  # Shape: (6,)
                # Node 7: Error node (deviations)
                if t < len_input:
                    dif_values = inputs_raw[idx, dif_indices]  # Use past deviation values
                else:
                    dif_values = np.zeros(len(residual_variables))  # At time t + 1, set deviations to zero

                # Assign features
                # Nodes 0-5 (Joint nodes)
                for node_idx in range(6):
                    input_sequence[t, node_idx, 0] = joint_positions[node_idx]
                    # The remaining feature indices (1-5) are already zero

                # Node 6: End-effector input node
                input_sequence[t, num_joints, :num_joints] = end_effector_target_pose  # Assign all 6 features

                # Node 7: Error node (deviations)
                input_sequence[t, num_joints + 1, 0:len(residual_variables)] = dif_values # Assign deviation features
                # The remaining feature indices are already zero

            # Transpose to (num_nodes=8, in_channels=6, len_input + 1)
            input_sequence = input_sequence.transpose(1, 2, 0)
            inputs.append(input_sequence)

            # Residual values (deviations at time t + 1)
            residual_idx = i + len_input  # Adjusted to get residual at t + 1
            residual_values = residuals_raw[residual_idx]  # Shape: (num_residual,)
            residuals.append(residual_values)

        inputs = np.array(inputs)  # Shape: (num_samples, num_nodes=8, in_channels=6, len_input + 1)
        print('inputs shape: '+str(inputs.shape))
        residuals = np.array(residuals)  # Shape: (num_samples, num_residual)

        return inputs, residuals

    # Create sequences for training, validation, and test sets
    inputs_train, residuals_train = create_sequences(inputs_train_df.reset_index(drop=True), residual_train_df.reset_index(drop=True))
    inputs_val, residuals_val = create_sequences(inputs_val_df.reset_index(drop=True), residual_val_df.reset_index(drop=True))
    inputs_test, residuals_test = create_sequences(inputs_test_df.reset_index(drop=True), residual_test_df.reset_index(drop=True))

    # Save the datasets
    np.savez_compressed(f'data/train_data_{prep_data_incl_past_residuals}_{graph_number}.npz', inputs=inputs_train, residuals=residuals_train)
    np.savez_compressed(f'data/val_data_{prep_data_incl_past_residuals}_{graph_number}.npz', inputs=inputs_val, residuals=residuals_val)
    np.savez_compressed(f'data/test_data_{prep_data_incl_past_residuals}_{graph_number}.npz', inputs=inputs_test, residuals=residuals_test)
    print("Datasets saved to disk.")

if __name__ == "__main__":
    # Load configuration
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)

    prepare_data(config)
