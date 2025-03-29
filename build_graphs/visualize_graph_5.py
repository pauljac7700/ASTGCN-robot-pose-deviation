import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_5(adjacency_matrix: np.ndarray, num_joints: int, save_path: str):
    """
    Visualizes and saves the graph from an adjacency matrix for a 6D robotic system.

    - Joint nodes are positioned in a straight horizontal line at the top.
    - Residual and target pose nodes are positioned in a straight horizontal line below.
    - Uses curved edges for non-neighboring joint nodes.
    - Differentiates between position and rotation nodes with distinct colors.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix representing the robotic system.
        num_joints (int): Number of joint nodes.
        save_path (str): Path to save the visualization.
    """

    num_nodes = adjacency_matrix.shape[0]

    # Create a graph from the adjacency matrix
    G = nx.Graph()

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)

    # Define node categories
    joint_nodes = list(range(num_joints))
    target_position_node = num_joints
    target_rotation_node = num_joints + 1
    residual_position_node = num_joints + 2
    residual_rotation_node = num_joints + 3

    # Set positions for better layout
    pos = {}

    # Arrange joint nodes in a horizontal line at the top
    for i, node in enumerate(joint_nodes):
        pos[node] = (i, 2)

    # Position all other nodes in a second line below the joint nodes
    pos[target_position_node] = (num_joints / 2 - 1.5, 0)
    pos[target_rotation_node] = (num_joints / 2 - 0.5, 0)
    pos[residual_position_node] = (num_joints / 2 + 0.5, 0)
    pos[residual_rotation_node] = (num_joints / 2 + 1.5, 0)

    # Define node colors to differentiate position vs. rotation nodes
    node_colors = [
        'lightblue' if n in joint_nodes else 
        '#FF9999' if n == residual_position_node else '#FF6666' if n == residual_rotation_node else 
        '#99FF99' if n == target_position_node else '#66CC66'
        for n in G.nodes
    ]

    # Separate straight and curved edges
    straight_edges = []
    curved_edges = []
    for u, v in G.edges():
        if pos[u][1] == pos[v][1]:  # Nodes on the same horizontal level
            if abs(u - v) == 1:
                straight_edges.append((u, v))  # Keep direct neighbors straight
            else:
                curved_edges.append((u, v))  # Curved for non-neighbors
        else:
            straight_edges.append((u, v))

    # Create figure with high DPI, transparent background, and no axis
    plt.figure(figsize=(8, 6), dpi=300)
    plt.axis("off")

    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, edge_color="gray")
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=curved_edges,
        edge_color="gray",
        arrows=True,
        arrowstyle='-',
        connectionstyle="arc3, rad=0.4"
    )

    # Legend with separate colors for position and orientation nodes
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Joint Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9999', markersize=10, label='Pose Residual (Position)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6666', markersize=10, label='Pose Residual (Rotation)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99FF99', markersize=10, label='Target Pose (Position)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#66CC66', markersize=10, label='Target Pose (Rotation)')
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    plt.title(f"Graph 5 Visualization (6D, {num_joints} Joints)")
    plt.tight_layout()

    # Save the visualization with transparent background
    image_filename = f"graph_visualization_5_{num_joints}joints.png"
    image_path = os.path.join(save_path, image_filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graph visualization saved as: {image_path}")

if __name__ == "__main__":
    num_joints = 6  # This method is only valid for 6D

    # Load adjacency matrix from the 'data' folder
    matrix_filename = "data/adjacency_matrix_5.npy"
    adjacency_matrix = np.load(matrix_filename)

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Visualize and save with high resolution
    visualize_graph_5(adjacency_matrix, num_joints, save_path=script_dir)
