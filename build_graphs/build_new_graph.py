import numpy as np
import yaml

def build_adjacency_matrix(config):
    """
    Builds the adjacency matrix for the graph representing the robotic system.
    
    Nodes:
    - Nodes 0-5: Joint nodes
    - Nodes 6-11: Error nodes for each pose dimension (x, y, z, and their corresponding rotations)
    - Nodes 12-17: Set nodes for each pose dimension (x, y, z, and their corresponding rotations)
    
    Connections:
    - Each joint node is connected to the next joint node like a chain.
    - Each joint node is connected to each error node for all dimensions.
    - Each set node is connected to each error node for all dimensions.
    - The error nodes are connected with each other.
    - The set nodes are connected with each other.
    - All connections have a weight/value of 1.
    """
    num_joints = 6
    num_pose_dimensions = 6  # x, y, z, and rotations (3 position and 3 orientation)
    num_error_nodes = num_pose_dimensions
    num_set_nodes = num_pose_dimensions
    num_nodes = num_joints + num_error_nodes + num_set_nodes
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Connect joint nodes in a chain (0-5)
    for i in range(num_joints - 1):
        adjacency_matrix[i, i+1] = 1
        adjacency_matrix[i+1, i] = 1

    # Connect each joint node (0-5) to each error node (6-11)
    for joint_node in range(num_joints):
        for error_node in range(num_error_nodes):
            adjacency_matrix[joint_node, num_joints + error_node] = 1
            adjacency_matrix[num_joints + error_node, joint_node] = 1

    # Connect each set node (12-17) to each error node (6-11)
    for error_node in range(num_error_nodes):
        for set_node in range(num_set_nodes):
            adjacency_matrix[num_joints + error_node, num_joints + num_error_nodes + set_node] = 1
            adjacency_matrix[num_joints + num_error_nodes + set_node, num_joints + error_node] = 1

    # Connect all error nodes with each other
    for i in range(num_error_nodes):
        for j in range(i+1, num_error_nodes):
            adjacency_matrix[num_joints + i, num_joints + j] = 1
            adjacency_matrix[num_joints + j, num_joints + i] = 1

    # Connect all set nodes with each other
    for i in range(num_set_nodes):
        for j in range(i+1, num_set_nodes):
            adjacency_matrix[num_joints + num_error_nodes + i, num_joints + num_error_nodes + j] = 1
            adjacency_matrix[num_joints + num_error_nodes + j, num_joints + num_error_nodes + i] = 1

    return adjacency_matrix

if __name__ == "__main__":
    # Load configuration
    with open('config_new_graph_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)
    
    adj_matrix = build_adjacency_matrix(config)
    np.save('data/adjacency_matrix_new_graph.npy', adj_matrix)
    print("Adjacency matrix saved to 'data/adjacency_matrix_new_graph.npy'")
    print("Adjacency Matrix:")
    print(adj_matrix)
