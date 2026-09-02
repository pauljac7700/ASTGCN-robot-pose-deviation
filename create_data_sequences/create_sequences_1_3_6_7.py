"""Build input sequences for the single-residual-node graphs (1, 3, 6, 7).

Produces windows shaped (samples, nodes, features, timesteps) with the residual on
node ``num_joints + 1``.
"""

import numpy as np

def create_sequences_1_3_6_7(
        inputs_df, residual_df, config, num_nodes, joint_features, target_pose_features, residual_variables
):
        
        inputs_raw = inputs_df.values
        residuals_raw = residual_df.values  # Shape: (num_samples, num_residuals)

        dataset_dimension = config['dataset_dimension']
        len_input = config['model']['len_input']  # Sequence length for inputs (e.g., 10)
        num_for_predict = config['model']['num_for_predict']  # Should be set to 1
        in_channels = config['model'][f'in_channels_{dataset_dimension}']  # Should remain at 6
        num_joints = config['num_joints']

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
        if config['prep_data_incl_past_residuals']:
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
                # Node 7: Pose residual node (deviations)
                if config['prep_data_incl_past_residuals']:
                    if t < len_input:
                        dif_values = inputs_raw[idx, dif_indices]  # Use past deviation values
                    else:
                        dif_values = np.zeros(len(residual_variables))  # At time t + 1, set deviations to zero
                else:
                    dif_values = np.zeros(len(residual_variables))  # Since we're excluding deviations from inputs
                


                # Assign features
                # Nodes 0-5 (Joint nodes)
                for node_idx in range(num_joints):
                    input_sequence[t, node_idx, 0] = joint_positions[node_idx]
                    # The remaining feature indices (1-5) are already zero

                # Node 6: End-effector input node
                input_sequence[t, num_joints, :num_joints] = end_effector_target_pose  # Assign all 6 features

                # Node 7: Error node (deviations)
                input_sequence[t, num_joints + 1, 0:len(residual_variables)] = dif_values  # Assign zeros to deviation features
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