"""Build input sequences for graphs 4 and 5, position and orientation split.

The residual occupies two nodes, three position features and three orientation
features. With a single prediction step the time dimension is squeezed and the two
nodes are flattened into one six-feature vector, which is what the evaluation code
expects.
"""

import os
import numpy as np

def create_sequences_4_5(
    inputs_df, residual_df, config, num_nodes, joint_features, target_pose_features, residual_variables
):
    """
    Creates sequences for a graph that splits pose target and pose residual
    into separate position and orientation nodes, with fixed in_channels = 3.
    The node structure is as follows:
      - Nodes 0 .. (num_joints-1): Joint nodes (each stores one joint feature in channel 0)
      - Node num_joints:         Target position node (x_t, y_t, z_t)
      - Node num_joints+1:       Target orientation node (rx_t, ry_t, rz_t)
      - Node num_joints+2:       Residual position node (x_dif, y_dif, z_dif)
      - Node num_joints+3:       Residual orientation node (rx_dif, ry_dif, rz_dif)

    Parameters
    ----------
    inputs_df : pd.DataFrame
        DataFrame containing input features.
    residual_df : pd.DataFrame
        DataFrame containing residual target values.
    config : dict
        Configuration dictionary containing keys:
          - 'dataset_dimension'
          - 'adjacency_matrix_file' (optional)
          - 'model' with keys 'len_input', 'num_for_predict'
          - 'prep_data_incl_past_residuals'
          - 'target_pose_variables' (dict by dimension)
          - 'residual_variables' (dict by dimension)
    num_nodes : int
        Total number of nodes in the graph.
    joint_features : list of str
        Column names for the joint features.

    Returns
    -------
    inputs : np.ndarray
        Array of shape (num_samples, num_nodes, 3, len_input+1)
    residuals : np.ndarray
        Array of shape (num_samples, num_residuals)
    """

    # Verify dataset_dimension consistency with the adjacency matrix file name, if provided.
    adjacency_matrix_file = config.get('adjacency_matrix_file')
    if adjacency_matrix_file is not None:
        basename = os.path.basename(adjacency_matrix_file)
        basename_no_ext = os.path.splitext(basename)[0]
        # Expecting the file name to end with the dimension string, e.g., '2_3D'
        expected_dimension = basename_no_ext.split('_')[-1]
        if config['dataset_dimension'] != expected_dimension:
            raise ValueError(f"Mismatch: config['dataset_dimension'] ({config['dataset_dimension']}) does not match "
                             f"the expected dimension from adjacency matrix file ({expected_dimension}).")

    dataset_dimension = config['dataset_dimension']
    len_input = config['model']['len_input']
    num_for_predict = config['model']['num_for_predict']  # typically 1
    # Fixed in_channels = 3
    in_channels = 3
    num_joints = config['num_joints']

    # Convert DataFrames to numpy arrays
    inputs_raw = inputs_df.values
    residuals_raw = residual_df.values  # (num_samples, num_residuals)

    # Determine how many sequences can be formed
    total_samples = inputs_raw.shape[0] - (len_input + 1) + 1
    if total_samples <= 0:
        print("Not enough data to create sequences.")
        return None, None

    # Get target and residual variable names from config
    target_pose_features = config.get('target_pose_variables', {}).get(dataset_dimension, [])
    residual_variables = config.get('residual_variables', {}).get(dataset_dimension, [])

    # Split target_pose_features into position and orientation groups.
    target_pos_feats = [f for f in target_pose_features if f in ['x_t', 'y_t', 'z_t']]
    target_rot_feats = [f for f in target_pose_features if f in ['rx_t', 'ry_t', 'rz_t']]

    # Similarly, split residual_variables.
    residual_pos_vars = [f for f in residual_variables if f in ['x_dif', 'y_dif', 'z_dif']]
    residual_rot_vars = [f for f in residual_variables if f in ['rx_dif', 'ry_dif', 'rz_dif']]

    # Convert column names to indices.
    joint_indices = [inputs_df.columns.get_loc(f) for f in joint_features]
    target_pos_indices = [inputs_df.columns.get_loc(f) for f in target_pos_feats]
    target_rot_indices = [inputs_df.columns.get_loc(f) for f in target_rot_feats]

    if config.get('prep_data_incl_past_residuals', False):
        residual_pos_indices = [inputs_df.columns.get_loc(f) for f in residual_pos_vars]
        residual_rot_indices = [inputs_df.columns.get_loc(f) for f in residual_rot_vars]

    # Define node indices based on the new graph layout.
    # Joint nodes: 0 .. num_joints-1.
    node_idx_target_pos = num_joints           # Target position node
    node_idx_target_rot = num_joints + 1         # Target orientation node
    node_idx_resid_pos = num_joints + 2          # Residual position node
    node_idx_resid_rot = num_joints + 3          # Residual orientation node

    inputs = []
    residuals = []

    for i in range(total_samples):
        # Initialize sequence array: (len_input+1, num_nodes, 3)
        input_sequence = np.zeros((len_input + 1, num_nodes, in_channels))

        for t in range(len_input + 1):
            idx = i + t

            # 1) Fill joint nodes
            joint_vals = inputs_raw[idx, joint_indices]  # shape: (num_joints,)
            for jn in range(num_joints):
                input_sequence[t, jn, 0] = joint_vals[jn]

            # 2) Fill target position node
            if target_pos_indices:
                pos_vals = inputs_raw[idx, target_pos_indices]  # shape: (number of target pos features)
                input_sequence[t, node_idx_target_pos, :len(pos_vals)] = pos_vals

            # 3) Fill target orientation node
            if target_rot_indices:
                rot_vals = inputs_raw[idx, target_rot_indices]  # shape: (number of target rot features)
                input_sequence[t, node_idx_target_rot, :len(rot_vals)] = rot_vals

            # 4) Fill residual nodes
            if config.get('prep_data_incl_past_residuals', False):
                if t < len_input:
                    resid_pos = inputs_raw[idx, residual_pos_indices] if residual_pos_vars else np.array([])
                    resid_rot = inputs_raw[idx, residual_rot_indices] if residual_rot_vars else np.array([])
                else:
                    resid_pos = np.zeros(len(residual_pos_vars))
                    resid_rot = np.zeros(len(residual_rot_vars))
            else:
                resid_pos = np.zeros(len(residual_pos_vars))
                resid_rot = np.zeros(len(residual_rot_vars))

            if residual_pos_vars:
                input_sequence[t, node_idx_resid_pos, :len(resid_pos)] = resid_pos
            if residual_rot_vars:
                input_sequence[t, node_idx_resid_rot, :len(resid_rot)] = resid_rot

        # Transpose sequence to (num_nodes, 3, len_input+1)
        input_sequence = input_sequence.transpose(1, 2, 0)
        inputs.append(input_sequence)

        # Use residual values from time step i + len_input as targets.
        residual_idx = i + len_input
        residual_values = residuals_raw[residual_idx]
        residuals.append(residual_values)

    inputs = np.array(inputs)      # Shape: (num_samples, num_nodes, 3, len_input+1)
    residuals = np.array(residuals)  # Shape: (num_samples, num_residuals)

    print(f"inputs shape: {inputs.shape}")
    return inputs, residuals
