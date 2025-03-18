import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml

# Load config
with open("config_ASTGCN_new_graph.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define the paths from the config file
data_file = config['data_file']
train_data_file = config['train_data_file']
val_data_file = config['val_data_file']
test_data_file = config['test_data_file']
scalers_file = config['scalers_file']

# Load the data
data = pd.read_csv(data_file)

# Data preparation
def create_sequences(inputs_df, target_df):
    inputs_raw = inputs_df.values  # Shape: (num_samples, num_input_features)
    targets_raw = target_df.values  # Shape: (num_samples, num_targets)

    len_input = config['model']['len_input']
    num_nodes = config['model']['num_of_vertices']  # Should be 18 nodes
    in_channels = config['model']['in_channels']  # Should remain at 1

    total_samples = inputs_raw.shape[0] - len_input
    if total_samples <= 0:
        print("Not enough data to create sequences.")
        return None, None

    inputs = []
    targets = []

    # Get indices for features
    joint_start_idx = inputs_df.columns.get_loc('joint_1')
    joint_end_idx = inputs_df.columns.get_loc('joint_6') + 1
    setpoint_start_idx = inputs_df.columns.get_loc('x_set')
    setpoint_end_idx = inputs_df.columns.get_loc('rz_set') + 1

    for i in range(total_samples):
        input_sequence = np.zeros((len_input + 1, num_nodes, in_channels))
        for t in range(len_input + 1):
            idx = i + t
            # Assign features to joint nodes (0-5)
            input_sequence[t, 0:6, 0] = inputs_raw[idx, joint_start_idx:joint_end_idx]
            # Assign features to error nodes (6-11)
            # Since we are not including error features in inputs, set them to zero
            input_sequence[t, 6:12, 0] = 0
            # Assign features to setpoint nodes (12-17)
            input_sequence[t, 12:18, 0] = inputs_raw[idx, setpoint_start_idx:setpoint_end_idx]

        inputs.append(input_sequence)
        # Target values at time t + len_input
        target_idx = i + len_input
        targets.append(targets_raw[target_idx])

    inputs = np.array(inputs)  # Shape: (num_samples, len_input + 1, num_nodes, in_channels)
    targets = np.array(targets)  # Shape: (num_samples, num_targets)

    # Transpose inputs to match ASTGCN expected input shape: (batch_size, num_nodes, in_channels, len_input + 1)
    inputs = inputs.transpose(0, 2, 3, 1)

    return inputs, targets

# Prepare data splitting
def prepare_data(data, target_columns, val_size=0.15, test_size=0.20, random_seed=42):
    # Exclude target variables from the features
    features = data.drop(columns=target_columns)
    target = data[target_columns]

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, target, test_size=val_size + test_size, random_state=random_seed, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (val_size + test_size), random_state=random_seed, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Apply scaling
def apply_scaling(X_train, X_val, X_test, y_train, y_val, y_test):
    # Initialize scalers dictionary
    scalers = {}

    # Fit scalers only on the training set for features
    for feature in X_train.columns:
        scaler = StandardScaler()
        X_train[feature] = scaler.fit_transform(X_train[[feature]])
        X_val[feature] = scaler.transform(X_val[[feature]])
        X_test[feature] = scaler.transform(X_test[[feature]])
        scalers[feature] = scaler

    # Fit scalers for target variables separately
    for target_col in y_train.columns:
        scaler = StandardScaler()
        y_train[target_col] = scaler.fit_transform(y_train[[target_col]])
        y_val[target_col] = scaler.transform(y_val[[target_col]])
        y_test[target_col] = scaler.transform(y_test[[target_col]])
        scalers[target_col] = scaler

    # Save scalers for later use
    joblib.dump(scalers, scalers_file)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Main data preparation flow
def preprocess():
    # Prepare the data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        data, target_columns=config['target_variables'],
        val_size=config['val_size'], test_size=config['test_size'])

    # Apply scaling
    X_train, X_val, X_test, y_train, y_val, y_test = apply_scaling(
        X_train, X_val, X_test, y_train, y_val, y_test)

    # Create sequences for ASTGCN
    inputs_train, targets_train = create_sequences(X_train, y_train)
    inputs_val, targets_val = create_sequences(X_val, y_val)
    inputs_test, targets_test = create_sequences(X_test, y_test)

    # Save preprocessed data
    np.savez(train_data_file, inputs=inputs_train, targets=targets_train)
    np.savez(val_data_file, inputs=inputs_val, targets=targets_val)
    np.savez(test_data_file, inputs=inputs_test, targets=targets_test)

    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess()
