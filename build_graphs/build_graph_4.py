import os
import numpy as np

def build_adjacency_matrix_4(num_joints: int) -> np.ndarray:
    """
    Constructs an adjacency matrix for a 6D robotic system, where the target pose node and pose residual node
    are each split into two nodes:
    
    - One node for (x, y, z) position.
    - One node for (rx, ry, rz) rotation.

    Nodes:
    - Nodes 0-(num_joints-1): Joint nodes (representing the robotic arm joints).
    - Node num_joints: Target pose node for position (x, y, z).
    - Node num_joints+1: Target pose node for rotation (rx, ry, rz).
    - Node num_joints+2: Pose residual node for position (x, y, z).
    - Node num_joints+3: Pose residual node for rotation (rx, ry, rz).

    Connections:
    - Each joint node (0 to num_joints-1) is connected sequentially in a chain.
    - Each residual node (position & rotation) is connected to every other node in the graph.
    - The adjacency matrix is symmetric, meaning connections are bidirectional.
    - All connections have an equal weight of 1.

    Args:
        num_joints (int): Number of joint nodes in the system.

    Returns:
        np.ndarray: The adjacency matrix of shape (num_nodes, num_nodes).
    """

    if num_joints < 2:
        raise ValueError("num_joints must be at least 2 to form a valid chain.")

    num_nodes = num_joints + 4  # num_joints + 2 target pose nodes + 2 residual nodes
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Connect joint nodes in a sequential chain (0 to num_joints-1)
    for i in range(num_joints - 1):
        adjacency_matrix[i, i+1] = 1
        adjacency_matrix[i+1, i] = 1

    # Indexing variables
    target_position_node = num_joints      # Target pose for position (x, y, z)
    target_rotation_node = num_joints + 1  # Target pose for rotation (rx, ry, rz)
    residual_position_node = num_joints + 2  # Pose residual for position (x, y, z)
    residual_rotation_node = num_joints + 3  # Pose residual for rotation (rx, ry, rz)

    # Connect residual nodes to **every other node**
    for node in range(num_nodes):
        if node != residual_position_node:  # Avoid self-connection
            adjacency_matrix[residual_position_node, node] = 1
            adjacency_matrix[node, residual_position_node] = 1

        if node != residual_rotation_node:  # Avoid self-connection
            adjacency_matrix[residual_rotation_node, node] = 1
            adjacency_matrix[node, residual_rotation_node] = 1

    return adjacency_matrix

if __name__ == "__main__":
    num_joints = 6  # This method is only valid for 6D

    adj_matrix = build_adjacency_matrix_4(num_joints)

    filename = "data/adjacency_matrix_4_6D.npy"
    np.save(filename, adj_matrix)
    
    print(f"Adjacency matrix saved to '{filename}'")
    print("Adjacency Matrix:")
    print(adj_matrix)
