# prep_data_STGAT.py

import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data_stgat(config):
    # 1. Load CSV
    df = pd.read_csv(config['data_file'])
    df.dropna(inplace=True)

    # The columns we'll use as input:
    #   1) 6 joint columns
    #   2) 6 setpoint columns
    #   3) 6 deviation columns (past deviation, to be placed on node 7)
    joint_features = [f'joint_{i}' for i in range(1, 7)]
    setpoint_features = ['x_set', 'y_set', 'z_set', 'rx_set', 'ry_set', 'rz_set']
    deviation_features = ['x_dif', 'y_dif', 'z_dif', 'rx_dif', 'ry_dif', 'rz_dif']

    # Combine them all as input_features
    input_features = joint_features + setpoint_features + deviation_features

    # The target variables (also from config):
    # e.g. ['x_dif','y_dif','z_dif','rx_dif','ry_dif','rz_dif']
    target_variables = config['target_variables']

    # Check columns in your CSV:
    # The following must exist in df.columns:
    #   joint_1..joint_6, x_set..rz_set, x_dif..rz_dif
    #   (and your target_variables)
    print("Available columns in CSV:", df.columns.tolist())

    # 2. Prepare data splits
    test_size = config['test_size']
    val_size = config['val_size']
    random_seed = config['random_seed']

    # Create DataFrame copies from selected columns
    inputs_df = df[input_features].copy()
    targets_df = df[target_variables].copy()

    train_size = 1 - test_size - val_size

    # Split into train+val, test
    inputs_train_val_df, inputs_test_df, targets_train_val_df, targets_test_df = train_test_split(
        inputs_df, targets_df,
        test_size=test_size,
        shuffle=False,
        random_state=random_seed
    )

    # Then split train, val
    val_size_adjusted = val_size / (train_size + val_size)
    inputs_train_df, inputs_val_df, targets_train_df, targets_val_df = train_test_split(
        inputs_train_val_df, targets_train_val_df,
        test_size=val_size_adjusted,
        shuffle=False,
        random_state=random_seed
    )

    print(f"Train: {len(inputs_train_df)}, Val: {len(inputs_val_df)}, Test: {len(inputs_test_df)}")

    # 3. Scale inputs and targets (fit only on training, then apply to val/test)
    input_scalers = {}
    for col in input_features:
        scaler = StandardScaler()
        scaler.fit(inputs_train_df[[col]])
        inputs_train_df[col] = scaler.transform(inputs_train_df[[col]])
        inputs_val_df[col]   = scaler.transform(inputs_val_df[[col]])
        inputs_test_df[col]  = scaler.transform(inputs_test_df[[col]])
        input_scalers[col]   = scaler

    target_scalers = {}
    for col in target_variables:
        scaler = StandardScaler()
        scaler.fit(targets_train_df[[col]])
        targets_train_df[col] = scaler.transform(targets_train_df[[col]])
        targets_val_df[col]   = scaler.transform(targets_val_df[[col]])
        targets_test_df[col]  = scaler.transform(targets_test_df[[col]])
        target_scalers[col]   = scaler

    stgat_scalers = {
        'input_scalers': input_scalers,
        'target_scalers': target_scalers
    }
    joblib.dump(stgat_scalers, config['scalers_file'])
    print(f"Scalers saved to {config['scalers_file']}")

    # 4. Build sequences for STGAT
    def build_sequences(inputs_df, targets_df, config):
        """
        We build sequences of length `len_input` from i..(i + len_input - 1),
        and the target is the row at i + len_input (the next step).
        Node 7 will contain the past (already scaled) deviation values.
        """

        inputs_arr = inputs_df.values  # shape: (num_samples, 18) if 6+6+6
        targets_arr = targets_df.values

        num_samples = len(inputs_df)
        len_input = config['model']['num_timesteps_input']  # e.g. 10
        B = num_samples - len_input

        if B <= 0:
            print("Not enough data to form sequences.")
            return None, None

        # STGAT expects shape (B, N=8, F=6, T=len_input)
        N = config['model']['num_of_vertices']   # 8
        F = config['model']['num_features']      # 6
        num_targets = len(target_variables)      # e.g. 6

        # Indices for each node
        joint_indices     = [inputs_df.columns.get_loc(f'joint_{i}') for i in range(1,7)]
        setpoint_indices  = [inputs_df.columns.get_loc(col) for col in setpoint_features]
        deviation_indices = [inputs_df.columns.get_loc(col) for col in deviation_features]

        seq_inputs  = []
        seq_targets = []

        for i in range(B):
            # input window: rows [i, i+1, ..., i + len_input - 1]
            # target row: i + len_input
            chunk = inputs_arr[i : i + len_input]

            # We'll hold (8, 6, len_input)
            arr_3d = np.zeros((N, F, len_input))

            for t in range(len_input):
                # Node 0..5 (joints)
                arr_3d[0:6, 0, t] = chunk[t, joint_indices]
                # Node 6 (setpoints)
                arr_3d[6, 0:6, t] = chunk[t, setpoint_indices]
                # Node 7 (past deviation)
                arr_3d[7, 0:6, t] = chunk[t, deviation_indices]

            target_idx = i + len_input
            if target_idx >= num_samples:
                break

            seq_inputs.append(arr_3d)
            seq_targets.append(targets_arr[target_idx])  # shape (6,) if 6 dev columns

        seq_inputs  = np.array(seq_inputs)   # (B, 8, 6, len_input)
        seq_targets = np.array(seq_targets)  # (B, 6)

        return seq_inputs, seq_targets

    # Build train/val/test sequences
    train_inputs, train_targets = build_sequences(inputs_train_df, targets_train_df, config)
    val_inputs, val_targets     = build_sequences(inputs_val_df,   targets_val_df, config)
    test_inputs, test_targets   = build_sequences(inputs_test_df,  targets_test_df, config)

    # Save to .npz
    if train_inputs is not None:
        np.savez_compressed(config['train_data_file'], inputs=train_inputs, targets=train_targets)
    if val_inputs is not None:
        np.savez_compressed(config['val_data_file'], inputs=val_inputs, targets=val_targets)
    if test_inputs is not None:
        np.savez_compressed(config['test_data_file'], inputs=test_inputs, targets=test_targets)

    print(f"STGAT datasets saved to {config['train_data_file']}, {config['val_data_file']}, {config['test_data_file']}")

if __name__ == "__main__":
    with open('config_STGAT.yaml', 'r') as f:
        config = yaml.safe_load(f)

    prepare_data_stgat(config)
