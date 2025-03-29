import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_2(adjacency_matrix: np.ndarray, num_joints: int, dataset_dimension: str, save_path: str):
    """
    Visualizes and saves the graph from an adjacency matrix with high resolution.

    Uses curved edges for nodes on the same horizontal line unless they are direct neighbors.
    Saves the visualization as a high-quality PNG in the same folder as the script.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix representing the robotic system.
        num_joints (int): Number of joint nodes.
        dataset_dimension (str): '3D' or '6D' dataset configuration.
        save_path (str): Path to save the visualization.
    """
    num_pose_dimensions = 3 if dataset_dimension == '3D' else 6
    num_error_nodes = num_pose_dimensions
    num_set_nodes = num_pose_dimensions

    # Create graph
    G = nx.Graph()
    num_nodes = adjacency_matrix.shape[0]
    
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)

    # Define node categories
    joint_nodes = list(range(num_joints))
    error_nodes = list(range(num_joints, num_joints + num_error_nodes))
    set_nodes = list(range(num_joints + num_error_nodes, num_nodes))

    # Set positions for better layout
    pos = {}
    
    # Joint nodes at y=2
    for i, node in enumerate(joint_nodes):
        pos[node] = (i, 2)
    
    # Error nodes at y=1
    for i, node in enumerate(error_nodes):
        pos[node] = (i * (num_joints / num_pose_dimensions), 1)
    
    # Target pose nodes at y=0
    for i, node in enumerate(set_nodes):
        pos[node] = (i * (num_joints / num_pose_dimensions), 0)

    # Define node colors
    node_colors = [
        'lightblue' if n in joint_nodes else
        'red' if n in error_nodes else
        'green'
        for n in G.nodes
    ]

    # Separate straight and curved edges
    straight_edges = []
    curved_edges = []
    for u, v in G.edges():
        if pos[u][1] == pos[v][1]:  # Nodes on same horizontal level
            if abs(u - v) == 1:
                straight_edges.append((u, v))  # Keep direct neighbors straight
            else:
                curved_edges.append((u, v))  # Curved for non-neighbors
        else:
            straight_edges.append((u, v))

    # Create figure with high DPI, transparent background, and no axis
    plt.figure(figsize=(8, 6), dpi=300)
    plt.axis("off")  # Remove black box around the graph

    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, edge_color="gray")
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=curved_edges,
        edge_color="gray",
        arrows=True,
        arrowstyle='-',
        connectionstyle="arc3, rad=0.3"
    )

    # Legend (Box is kept)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Joint Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Pose Residual Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='End-Effector Target Nodes')
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)  # Box is ON

    plt.title(f"Graph 2 Visualization ({dataset_dimension}, {num_joints} Joints)")
    plt.tight_layout()

    # Save the visualization with transparent background
    image_filename = f"graph_visualization_2_{dataset_dimension}_{num_joints}joints.png"
    image_path = os.path.join(save_path, image_filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight') 
    plt.close()

    print(f"Graph visualization saved as: {image_path})")

if __name__ == "__main__":
    # Define dataset and joints
    dataset_dimension = '6D'
    num_joints = 6
    
    # Load adjacency matrix from the 'data' folder
    adjacency_matrix = np.load(f'data/adjacency_matrix_2_{dataset_dimension}.npy')

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Visualize and save
    visualize_graph_2(adjacency_matrix, num_joints, dataset_dimension, save_path=script_dir)
