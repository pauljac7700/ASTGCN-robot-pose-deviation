import numpy as np
import yaml

def build_adjacency_matrix(config):
    """
    Builds the adjacency matrix for the graph representing the robotic system.
    
    Nodes:
    - Nodes 0-5: Joint nodes
    - Node 6: End-effector input node
    - Node 7: Error node (x_dif)

    Connections:
    - Each joint node is connected to the next joint node like a chain.
    - Each joint node is connected to the end-effector input node.
    - All nodes are connected to the error node.
    - All connections have a weight/value of 1.
    """
    num_nodes = config['model']['num_of_vertices'] # 6 joint nodes + 1 end-effector input node + 1 error node
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Connect joint nodes in a chain
    for i in range(5):
        adjacency_matrix[i, i+1] = 1
        adjacency_matrix[i+1, i] = 1

    # Connect each joint node (0-5) to the end-effector input node (6)
    for joint_node in range(6):
        adjacency_matrix[joint_node, 6] = 1
        adjacency_matrix[6, joint_node] = 1

    # Connect all nodes to the error node (7)
    for node in range(num_nodes - 1):  # Nodes 0-6
        adjacency_matrix[node, 7] = 1
        adjacency_matrix[7, node] = 1

    # Since the connections are bidirectional and have a weight of 1, the adjacency matrix is symmetric
    return adjacency_matrix

if __name__ == "__main__":
    # Load configuration
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)
    
    adj_matrix = build_adjacency_matrix(config)
    np.save('data/adjacency_matrix.npy', adj_matrix)
    print("Adjacency matrix saved to 'data/adjacency_matrix.npy'")
    print("Adjacency Matrix:")
    print(adj_matrix)
