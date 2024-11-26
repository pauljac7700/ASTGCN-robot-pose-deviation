# data_prep.py

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml

def prepare_data(config):
    # Load data
    df = pd.read_csv(config['data_file'])

    # Drop unnecessary columns and check for missing values
    df = df.drop(columns=['step_order'], errors='ignore')
    if df.isnull().values.any():
        df = df.dropna()

    # Define input features and target variable
    target_variable = config['single_target_variable']  # The target variable is configurable and specified in the config file
    joint_features = [f'joint_{i}' for i in range(1, 7)]  # Joint features (6 variables)
    setpoint_features = ['x_set', 'y_set', 'z_set', 'rx_set', 'ry_set', 'rz_set']  # Setpoint features (6 variables)

    # Include the selected target variable in input features for previous time steps
    input_features = joint_features + setpoint_features + [target_variable]

    # Ensure all required columns are present
    required_columns = input_features + [target_variable]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"The following required columns are missing from the dataset: {missing_columns}")
        return

    # Extract data for inputs and target
    inputs_df = df[input_features]
    target_df = df[[target_variable]]  # Target variable is dynamically selected based on the config

    # Split the data into train, validation, and test sets
    test_size = config['test_size']
    val_size = config['val_size']
    train_size = 1 - test_size - val_size

    # First split off the test set
    inputs_train_val_df, inputs_test_df, target_train_val_df, target_test_df = train_test_split(
        inputs_df, target_df, test_size=test_size, random_state=config['random_seed'], shuffle=False)

    # Then split train and validation sets without shuffling to prevent data leakage
    val_size_adjusted = val_size / (train_size + val_size)
    inputs_train_df, inputs_val_df, target_train_df, target_val_df = train_test_split(
        inputs_train_val_df, target_train_val_df, test_size=val_size_adjusted, random_state=config['random_seed'], shuffle=False)

    print(f"Training set size: {inputs_train_df.shape[0]}")
    print(f"Validation set size: {inputs_val_df.shape[0]}")
    print(f"Test set size: {inputs_test_df.shape[0]}")

    # Scale the data
    # Fit scalers on training data only
    input_scalers = {}
    for feature in input_features:
        scaler = StandardScaler()
        scaler.fit(inputs_train_df[[feature]].values)  # Fit scaler only on training data
        inputs_train_df[feature] = scaler.transform(inputs_train_df[[feature]].values)
        inputs_val_df[feature] = scaler.transform(inputs_val_df[[feature]].values)
        inputs_test_df[feature] = scaler.transform(inputs_test_df[[feature]].values)
        input_scalers[feature] = scaler  # Save the scaler for each feature

    # Scale the target variable
    target_scaler = StandardScaler()
    target_scaler.fit(target_train_df[[target_variable]].values)
    target_train_df[target_variable] = target_scaler.transform(target_train_df[[target_variable]].values)
    target_val_df[target_variable] = target_scaler.transform(target_val_df[[target_variable]].values)
    target_test_df[target_variable] = target_scaler.transform(target_test_df[[target_variable]].values)

    # Save the scalers
    scalers = {'input_scalers': input_scalers, 'target_scaler': target_scaler}
    joblib.dump(scalers, config['scalers_file'])
    print(f"Scalers saved to {config['scalers_file']}")

    # Prepare sequences for inputs and targets
    def create_sequences(inputs_df, target_df):
        inputs_raw = inputs_df.values
        targets_raw = target_df.values

        len_input = config['model']['len_input']  # Sequence length for inputs (e.g., 10)
        num_for_predict = config['model']['num_for_predict']  # Typically 1
        num_nodes = config['model']['num_of_vertices']  # Typically 8 nodes
        in_channels = config['model']['in_channels']  # Input channels (e.g., 6)

        # Ensure there are enough samples
        total_samples = inputs_raw.shape[0] - (len_input + 1) + 1
        if total_samples <= 0:
            print("Not enough data to create sequences.")
            return None, None

        inputs = []
        targets = []

        # Get indices for features
        joint_indices = [inputs_df.columns.get_loc(f) for f in joint_features]
        setpoint_indices = [inputs_df.columns.get_loc(f) for f in setpoint_features]
        target_variable_index = inputs_df.columns.get_loc(target_variable)  # Dynamically get index of target variable

        for i in range(total_samples):
            # Initialize input sequence with zeros
            input_sequence = np.zeros((len_input + 1, num_nodes, in_channels))

            for t in range(len_input + 1):  # Includes time t + 1
                idx = i + t

                # Nodes 0-5: Joint positions
                joint_positions = inputs_raw[idx, joint_indices]  # Shape: (6,)
                # Node 6: End-effector setpoints
                end_effector_setpoints = inputs_raw[idx, setpoint_indices]  # Shape: (6,)
                # Node 7: Error node (target variable from config)
                if t < len_input:
                    target_value = inputs_raw[idx, target_variable_index]  # Use past target variable values
                else:
                    target_value = 0  # At time t + 1, set target variable to zero to avoid data leakage

                # Assign features
                # Nodes 0-5 (Joint nodes)
                for node_idx in range(6):
                    input_sequence[t, node_idx, 0] = joint_positions[node_idx]
                # Node 6: End-effector input node
                input_sequence[t, 6, :6] = end_effector_setpoints
                # Node 7: Target node (configurable target variable)
                input_sequence[t, 7, 0] = target_value

            # Transpose to (num_nodes=8, in_channels=6, len_input + 1)
            input_sequence = input_sequence.transpose(1, 2, 0)
            inputs.append(input_sequence)

            # Target value at time t + 1
            target_idx = i + len_input
            target_value = targets_raw[target_idx]
            targets.append(target_value[0])

        inputs = np.array(inputs)  # Shape: (num_samples, num_nodes=8, in_channels=6, len_input + 1)
        targets = np.array(targets)  # Shape: (num_samples,)

        return inputs, targets

    # Create sequences for training, validation, and test sets
    inputs_train, targets_train = create_sequences(inputs_train_df.reset_index(drop=True), target_train_df.reset_index(drop=True))
    inputs_val, targets_val = create_sequences(inputs_val_df.reset_index(drop=True), target_val_df.reset_index(drop=True))
    inputs_test, targets_test = create_sequences(inputs_test_df.reset_index(drop=True), target_test_df.reset_index(drop=True))

    # Save the datasets
    np.savez_compressed(config['train_data_file'], inputs=inputs_train, targets=targets_train)
    np.savez_compressed(config['val_data_file'], inputs=inputs_val, targets=targets_val)
    np.savez_compressed(config['test_data_file'], inputs=inputs_test, targets=targets_test)
    print("Datasets saved to disk.")

if __name__ == "__main__":
    # Load configuration
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)

    prepare_data(config)
