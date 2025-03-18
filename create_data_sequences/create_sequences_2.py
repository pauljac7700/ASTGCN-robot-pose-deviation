import numpy as np

def create_sequences_2_wr(inputs_df, residual_df, config, dataset_dimension, joint_features, target_pose_features, residual_variables, num_joints):
        
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
    joint_start_idx = inputs_df.columns.get_loc('joint_1')
    joint_end_idx = inputs_df.columns.get_loc('joint_6') + 1
    error_start_idx = inputs_df.columns.get_loc('x_dif')
    error_end_idx = inputs_df.columns.get_loc('rz_dif') + 1
    setpoint_start_idx = inputs_df.columns.get_loc('x_set')
    setpoint_end_idx = inputs_df.columns.get_loc('rz_set') + 1

    for i in range(total_samples):
        input_sequence = np.zeros((len_input + 1, num_nodes, in_channels))
        for t in range(len_input + 1):
            idx = i + t
            # Assign features to joint nodes (0-5)
            input_sequence[t, 0:6, 0] = inputs_raw[idx, joint_start_idx:joint_end_idx]
            # Assign features to error nodes (6-11)
            if t < len_input:
                # Use past error values
                input_sequence[t, 6:12, 0] = inputs_raw[idx, error_start_idx:error_end_idx]
            else:
                # For the last time step, set error node values to zero
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

#------------------------------------------------------------

def create_sequences_2_nr(inputs_df, residual_df, config, dataset_dimension, joint_features, target_pose_features, residual_variables, num_joints):
        
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