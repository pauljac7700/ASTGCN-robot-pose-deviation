"""Dispatch to the sequence builder matching the configured graph variant.

The graph topologies place the residual on different nodes, so each family needs
its own tensor layout. This picks the right builder from the adjacency matrix
filename, which is why the graph number is encoded there.
"""

from create_data_sequences.create_sequences_1_3_6_7 import create_sequences_1_3_6_7
from create_data_sequences.create_sequences_2 import create_sequences_2
from create_data_sequences.create_sequences_4_5 import create_sequences_4_5

def create_all_sequences(graph_nr, num_nodes, inputs_train_df, residual_train_df, inputs_val_df, residual_val_df, inputs_test_df, residual_test_df, config, joint_features, target_pose_features, residual_variables):
    """
    Create sequences for training, validation, and test sets based on the specified graph number and whether to include past residuals.   
    """

    if graph_nr in [1,3,6,7]:
        inputs_train, residuals_train = create_sequences_1_3_6_7(inputs_train_df.reset_index(drop=True), residual_train_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)
        inputs_val, residuals_val = create_sequences_1_3_6_7(inputs_val_df.reset_index(drop=True), residual_val_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)
        inputs_test, residuals_test = create_sequences_1_3_6_7(inputs_test_df.reset_index(drop=True), residual_test_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)
    elif graph_nr == 2:
        inputs_train, residuals_train = create_sequences_2(inputs_train_df.reset_index(drop=True), residual_train_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)
        inputs_val, residuals_val = create_sequences_2(inputs_val_df.reset_index(drop=True), residual_val_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)
        inputs_test, residuals_test = create_sequences_2(inputs_test_df.reset_index(drop=True), residual_test_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)
    elif graph_nr in [4,5]:
        inputs_train, residuals_train = create_sequences_4_5(inputs_train_df.reset_index(drop=True), residual_train_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)
        inputs_val, residuals_val = create_sequences_4_5(inputs_val_df.reset_index(drop=True), residual_val_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)
        inputs_test, residuals_test = create_sequences_4_5(inputs_test_df.reset_index(drop=True), residual_test_df.reset_index(drop=True), config, num_nodes, joint_features, target_pose_features, residual_variables)

    return inputs_train, residuals_train, inputs_val, residuals_val, inputs_test, residuals_test