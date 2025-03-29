import numpy as np
import os

def create_sequences_2(
    inputs_df, residual_df, config, num_nodes, joint_features, target_pose_features, residual_variables
):
    """
    Creates sequences for a node graph with in_channels=1, including joint nodes,
    pose residual nodes, and end-effector target nodes. Pose residual nodes are filled with
    past residual values for all time steps except the last one in each sequence (where they are set to zero)
    if config['prep_data_incl_past_residuals'] is True.

    The node graph is structured as follows:
      - Nodes 0 .. num_joints-1:
            Joint nodes. Each node stores one joint dimension in channel 0.
      - Nodes num_joints .. num_joints+num_residual_nodes-1:
            Pose residual nodes. If config['prep_data_incl_past_residuals'] is True, these nodes are filled
            with past residual values at each time step except the last one in each sequence.
      - Nodes num_joints+num_residual_nodes .. num_joints+2*num_residual_nodes-1:
            End-effector target nodes. Each node stores one target dimension in channel 0.

    Additionally, the function checks that the dataset_dimension in the config matches the expected
    dimension indicated at the end of the adjacency matrix file name (e.g., for 'adjacency_matrix_2_3D.npz', the
    expected dimension is '2_3D'). If not, an error is thrown.

    Parameters:
        inputs_df (DataFrame): DataFrame containing input data.
        residual_df (DataFrame): DataFrame containing residual data.
        config (dict): Configuration dictionary with keys:
                       - 'dataset_dimension'
                       - 'model': containing 'len_input' and 'num_for_predict'
                       - 'num_joints'
                       - 'prep_data_incl_past_residuals'
                       - 'adjacency_matrix_file' (optional): used to verify dataset_dimension.
        num_nodes (int): Total number of nodes.
        joint_features (list): List of joint feature names.
        target_pose_features (list): List of target pose feature names.
        residual_variables (list): List of residual variable names.

    Returns:
        tuple: (inputs, residuals) where:
               - inputs is a numpy array of shape (num_samples, num_nodes, in_channels, len_input+1)
               - residuals is a numpy array of shape (num_samples, num_residuals)

        If there isn't enough data to create sequences, returns (None, None).

    Raises:
        ValueError: If the dataset_dimension in config does not match the expected dimension
                    from the adjacency matrix file name.
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

    inputs_raw = inputs_df.values
    residuals_raw = residual_df.values  # shape: (num_samples, num_residuals)

    len_input = config['model']['len_input']            # e.g., 10
    num_for_predict = config['model']['num_for_predict']  # typically 1
    in_channels = 1                                     # Hardcoded to 1 for this graph
    num_joints = config['num_joints']
    num_residual_nodes = (num_nodes - num_joints) // 2   # Ensure integer division

    total_samples = inputs_raw.shape[0] - (len_input + 1) + 1
    if total_samples <= 0:
        print("Not enough data to create sequences.")
        return None, None

    inputs = []
    residuals = []

    # Get column indices from the DataFrame for each feature group
    joint_indices = [inputs_df.columns.get_loc(f) for f in joint_features]        
    target_pose_indices = [inputs_df.columns.get_loc(f) for f in target_pose_features] 
    if config.get('prep_data_incl_past_residuals', False):
        dif_indices = [inputs_df.columns.get_loc(f) for f in residual_variables]

    for i in range(total_samples):
        # Create a sequence array with shape: (len_input+1, num_nodes, in_channels)
        input_sequence = np.zeros((len_input + 1, num_nodes, in_channels))

        for t in range(len_input + 1):
            idx = i + t

            # 1) Joint nodes (0 .. num_joints-1)
            joint_positions = inputs_raw[idx, joint_indices]  # shape: (num_joints,)
            for node_idx in range(num_joints):
                input_sequence[t, node_idx, 0] = joint_positions[node_idx]

            # 2) Pose residual nodes (num_joints .. num_joints+num_residual_nodes-1)
            if config.get('prep_data_incl_past_residuals', False):
                # Use past residuals for all time steps except the last one
                if t < len_input:
                    dif_values = inputs_raw[idx, dif_indices]  # past residuals
                else:
                    dif_values = np.zeros(len(dif_indices))    # zero at t+1
                for node_idx in range(num_residual_nodes):
                    input_sequence[t, num_joints + node_idx, 0] = dif_values[node_idx]

            # 3) End-effector target nodes (num_joints+num_residual_nodes .. num_joints+2*num_residual_nodes-1)
            target_pose = inputs_raw[idx, target_pose_indices]  # shape: (num_joints,)
            for node_idx in range(num_residual_nodes):
                input_sequence[t, num_joints + num_residual_nodes + node_idx, 0] = target_pose[node_idx]

        # Transpose to (num_nodes, in_channels, len_input+1)
        input_sequence = input_sequence.transpose(1, 2, 0)
        inputs.append(input_sequence)

        # Residual at time i + len_input
        residual_idx = i + len_input
        residual_values = residuals_raw[residual_idx]  # shape: (num_residuals)
        residuals.append(residual_values)

    inputs = np.array(inputs)     # Final shape: (num_samples, num_nodes, in_channels, len_input+1)
    residuals = np.array(residuals)
    print(f"inputs shape: {inputs.shape}")
    return inputs, residuals
