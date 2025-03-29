import numpy as np

def build_adjacency_matrix_7(num_joints: int) -> np.ndarray:
    """
    Constructs an adjacency matrix for a robotic system where joint nodes are independent
    (not connected to each other), but connected to the end-effector and residual nodes.

    Node Indexing:
    - Joint nodes:            indices 0 to num_joints-1
    - End-effector node:      index num_joints
    - Pose residual node:     index num_joints + 1

    Connections:
    - No direct connections between joint nodes.
    - Each joint node connects to the end-effector node.
    - All nodes connect to the residual node.
    - All edges are symmetric and unweighted (binary: 1 for connected).

    Args:
        num_joints (int): Number of joint nodes (≥ 2).

    Returns:
        np.ndarray: Adjacency matrix of shape (num_joints + 2, num_joints + 2).
    """
    if num_joints < 2:
        raise ValueError("num_joints must be at least 2.")

    num_nodes = num_joints + 2
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    end_effector_node = num_joints
    residual_node = num_joints + 1

    # Connect each joint to the end-effector
    for joint in range(num_joints):
        adjacency_matrix[joint, end_effector_node] = 1
        adjacency_matrix[end_effector_node, joint] = 1

    # Connect all joints + end-effector to the residual node
    for node in range(num_joints + 1):  # 0 to num_joints (inclusive)
        adjacency_matrix[node, residual_node] = 1
        adjacency_matrix[residual_node, node] = 1

    return adjacency_matrix

if __name__ == "__main__":
    num_joints = 6
    adj_matrix = build_adjacency_matrix_7(num_joints)

    filename = f"data/adjacency_matrix_7.npy"
    np.save(filename, adj_matrix)

    print(f"Adjacency matrix saved to '{filename}'")
    print("Adjacency Matrix:")
    print(adj_matrix)
