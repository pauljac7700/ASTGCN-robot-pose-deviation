"""Graph 3: as graph 1, but joints are not connected to the target-pose node.

The only difference from graph 1 is the removed joint-to-target edges, so
comparing the two isolates what that connection contributes.
"""

import os
import numpy as np

def build_adjacency_matrix_3(num_joints: int) -> np.ndarray:
    """
    Constructs an adjacency matrix representing the connectivity of a robotic system.

    Nodes:
    - Nodes 0-(num_joints-1): Joint nodes (representing the robotic arm joints).
    - Node num_joints: End-effector target pose node (represents the desired target position of the robot).
    - Node num_joints+1: Pose residual node (captures the error between actual and desired poses).

    Connections:
    - Each joint node (0 to num_joints-1) is connected sequentially in a chain.
    - Joint nodes are NOT connected to the end-effector target pose node.
    - All nodes (0 to num_joints) are connected to the pose residual node.
    - The adjacency matrix is symmetric, meaning connections are bidirectional.
    - All connections have an equal weight of 1.

    Args:
        num_joints (int): Number of joint nodes in the system.

    Returns:
        np.ndarray: The adjacency matrix of shape (num_nodes, num_nodes).
    """

    if num_joints < 2:
        raise ValueError("num_joints must be at least 2 to form a valid chain.")

    num_nodes = num_joints + 2  # num_joints + 1 end-effector node + 1 error node
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Connect joint nodes in a sequential chain (0 to num_joints-1)
    for i in range(num_joints - 1):
        adjacency_matrix[i, i+1] = 1
        adjacency_matrix[i+1, i] = 1

    # Indexing variables
    end_effector_node = num_joints
    residual_node = num_joints + 1

    # Joint nodes are NOT connected to the end-effector node

    # Connect all nodes (0 to num_joints) to the pose residual node
    for node in range(num_joints + 1):  # Includes the end-effector node
        adjacency_matrix[node, residual_node] = 1
        adjacency_matrix[residual_node, node] = 1

    return adjacency_matrix

if __name__ == "__main__":
    num_joints = 6  # Change to any number of joints

    adj_matrix = build_adjacency_matrix_3(num_joints)

    filename = f"data/adjacency_matrix_3.npy"
    np.save(filename, adj_matrix)
    
    print(f"Adjacency matrix saved to '{filename}'")
    print("Adjacency Matrix:")
    print(adj_matrix)
