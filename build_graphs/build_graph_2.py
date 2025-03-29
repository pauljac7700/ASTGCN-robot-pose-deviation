import numpy as np

def build_adjacency_matrix_2(dataset_dimension: str, num_joints: int) -> np.ndarray:
    """
    Constructs an adjacency matrix representing the connectivity of a robotic system.

    Nodes:
    - Nodes 0-(num_joints-1): Joint nodes (representing the robotic arm joints).
    - Nodes num_joints-(num_joints + num_pose_dimensions - 1): Pose residual (error) nodes for each pose dimension.
    - Nodes (num_joints + num_pose_dimensions)-(num_joints + 2 * num_pose_dimensions - 1): Target pose nodes for each pose dimension.

    Connections:
    - Each joint node (0 to num_joints-1) is connected sequentially in a chain.
    - Each joint node is connected to **all** error nodes.
    - Each target pose node is connected to **all** error nodes.
    - All error nodes are fully connected to each other.
    - All target pose nodes are fully connected to each other.
    - The adjacency matrix is symmetric, meaning connections are bidirectional.
    - All connections have an equal weight of 1.

    Args:
        dataset_dimension (str): '3D' (only x, y, z) or '6D' (x, y, z + rotations).
        num_joints (int): Number of joint nodes in the system.

    Returns:
        np.ndarray: The adjacency matrix of shape (num_nodes, num_nodes).
    """

    if dataset_dimension not in {'3D', '6D'}:
        raise ValueError("Invalid dataset_dimension. Must be '3D' or '6D'.")

    if num_joints < 2:
        raise ValueError("num_joints must be at least 2 to form a valid chain.")

    num_pose_dimensions = 3 if dataset_dimension == '3D' else 6  # 3D: (x, y, z) | 6D: (x, y, z + rotations)
    num_error_nodes = num_pose_dimensions  # Error nodes (one per pose dimension)
    num_set_nodes = num_pose_dimensions  # Target pose nodes (one per pose dimension)
    num_nodes = num_joints + num_error_nodes + num_set_nodes  # Total nodes

    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Connect joint nodes in a sequential chain (0 to num_joints-1)
    for i in range(num_joints - 1):
        adjacency_matrix[i, i+1] = 1
        adjacency_matrix[i+1, i] = 1

    # Indexing variables
    error_start = num_joints
    set_start = num_joints + num_error_nodes

    # Connect each joint node to each error node
    for joint_node in range(num_joints):
        for error_node in range(num_error_nodes):
            adjacency_matrix[joint_node, error_start + error_node] = 1
            adjacency_matrix[error_start + error_node, joint_node] = 1

    # Connect each target pose node to each error node
    for error_node in range(num_error_nodes):
        for set_node in range(num_set_nodes):
            adjacency_matrix[error_start + error_node, set_start + set_node] = 1
            adjacency_matrix[set_start + set_node, error_start + error_node] = 1

    # Fully connect all error nodes with each other
    for i in range(num_error_nodes):
        for j in range(i+1, num_error_nodes):
            adjacency_matrix[error_start + i, error_start + j] = 1
            adjacency_matrix[error_start + j, error_start + i] = 1

    # Fully connect all set nodes with each other
    for i in range(num_set_nodes):
        for j in range(i+1, num_set_nodes):
            adjacency_matrix[set_start + i, set_start + j] = 1
            adjacency_matrix[set_start + j, set_start + i] = 1

    return adjacency_matrix

if __name__ == "__main__":
    dataset_dimension = '6D'  # Change to '3D' or '6D' as needed
    num_joints = 6  # Change to any number of joints

    adj_matrix = build_adjacency_matrix_2(dataset_dimension, num_joints)

    filename = f'data/adjacency_matrix_2_{dataset_dimension}.npy'
    np.save(filename, adj_matrix)
    
    print(f"Adjacency matrix saved to '{filename}'")
    print("Adjacency Matrix:")
    print(adj_matrix)
