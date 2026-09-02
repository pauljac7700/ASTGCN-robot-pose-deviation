"""Render graph 4, the position/orientation split topology, as a figure."""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_4(adjacency_matrix: np.ndarray, num_joints: int, save_path: str):
    """
    Visualizes and saves the graph from an adjacency matrix for a 6D robotic system.

    - Joint nodes are arranged at the top.
    - Residual nodes are in the middle (separated into position and rotation).
    - Target pose nodes are at the bottom (separated into position and rotation).

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

    # Position residual nodes in the middle
    pos[residual_position_node] = (num_joints / 2 - 0.5, 1)
    pos[residual_rotation_node] = (num_joints / 2 + 0.5, 1)

    # Position target pose nodes at the bottom
    pos[target_position_node] = (num_joints / 2 - 0.5, 0)
    pos[target_rotation_node] = (num_joints / 2 + 0.5, 0)

    # Define node colors to distinguish position vs. rotation
    node_colors = [
        'lightblue' if n in joint_nodes else 
        '#FF9999' if n == residual_position_node else '#FF6666' if n == residual_rotation_node else 
        '#99FF99' if n == target_position_node else '#66CC66'
        for n in G.nodes
    ]

    # Create figure with high DPI, transparent background, and no axis
    plt.figure(figsize=(8, 6), dpi=300)
    plt.axis("off")

    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color="gray")

    # Legend with separate colors for position and orientation nodes
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Joint Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9999', markersize=10, label='Pose Residual (Position)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6666', markersize=10, label='Pose Residual (Rotation)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99FF99', markersize=10, label='Target Pose (Position)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#66CC66', markersize=10, label='Target Pose (Rotation)')
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    plt.title(f"Graph 4 Visualization (6D, {num_joints} Joints)")
    plt.tight_layout()

    # Save the visualization with transparent background
    image_filename = f"graph_visualization_4_{num_joints}joints.png"
    image_path = os.path.join(save_path, image_filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graph visualization saved as: {image_path}")

if __name__ == "__main__":
    num_joints = 6  # This method is only valid for 6D

    # Load adjacency matrix from the 'data' folder
    matrix_filename = "data/adjacency_matrix_4.npy"
    adjacency_matrix = np.load(matrix_filename)

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Visualize and save with high resolution
    visualize_graph_4(adjacency_matrix, num_joints, save_path=script_dir)
