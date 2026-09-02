"""Render the single-residual-node topologies (graphs 1, 3, 6 and 7) as a figure."""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Fix: Add parent directory to sys.path so Python finds 'lib'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.extract_number_from_filename import extract_number_from_filename

def visualize_graph_1_or_3_or_6_or_7(adjacency_matrix: np.ndarray, graph_number: int, num_joints: int, save_path: str, joint_order: list[int] = None):
    """
    Visualizes and saves the graph from an adjacency matrix with high resolution.

    Joint nodes are arranged horizontally at the top, while the end-effector and residual nodes 
    are positioned at the bottom.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix representing the robotic system.
        graph_number (int): Identifier for the graph (e.g., 1 or 3 or 6 or 7).
        num_joints (int): Number of joint nodes.
        save_path (str): Path to save the visualization.
        joint_order (list[int], optional): List of node indices in desired left-to-right order.
    """
    
    num_nodes = adjacency_matrix.shape[0]
    G = nx.Graph()

    # Add nodes and edges
    for i in range(num_nodes):
        G.add_node(i)
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)

    # Default or custom joint order
    joint_nodes = list(range(num_joints)) if joint_order is None else joint_order
    end_effector_node = num_joints
    residual_node = num_joints + 1

    # Set node positions
    pos = {}
    for x, node in enumerate(joint_nodes):
        pos[node] = (x, 2)

    pos[end_effector_node] = (num_joints / 2 - 0.5, 0)
    pos[residual_node] = (num_joints / 2 + 0.5, 0)

    # Set node colors
    node_colors = ['lightblue' if n in joint_nodes else
                   'green' if n == end_effector_node else
                   'red' for n in G.nodes]

    # Draw graph
    plt.figure(figsize=(8, 6), dpi=300)
    plt.axis("off")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color="gray")

    # Add legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Joint Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='End-Effector Target Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Pose Residual Node')
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    plt.title(f"Graph {graph_number} Visualization (3D and 6D, {num_joints} Joints)")
    plt.tight_layout()

    # Save image
    image_filename = f"graph_visualization_{graph_number}_{num_joints}joints.png"
    image_path = os.path.join(save_path, image_filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')  
    plt.close()

    print(f"Graph visualization saved as: {image_path}")

if __name__ == "__main__":
    num_joints = 6
    matrix_filename = "data/adjacency_matrix_7.npy" 
    adjacency_matrix = np.load(matrix_filename)

    graph_number = extract_number_from_filename(matrix_filename) 
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Optional: load joint order if shuffled
    joint_order_path = "data/shuffle_order_for_graph_6.npy"
    graph_joint_order_number = extract_number_from_filename(joint_order_path) 
    if os.path.exists(joint_order_path) and graph_joint_order_number == graph_number:
        joint_order = np.load(joint_order_path).tolist()
        print("Loaded shuffled joint order:", joint_order)
    else:
        joint_order = None

    visualize_graph_1_or_3_or_6_or_7(adjacency_matrix, graph_number, num_joints, save_path=script_dir, joint_order=joint_order)
