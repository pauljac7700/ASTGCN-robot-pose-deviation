"""Graph 6: control experiment with the joint chain randomly shuffled.

Identical to graph 1 except that the order of joints along the chain is permuted,
destroying the true kinematic ordering while leaving the number of nodes and edges
unchanged. If the model performs as well on this as on graph 1, then it is not
exploiting the kinematic structure, and the graph is decoration. Returns the
shuffled ordering alongside the matrix so a run can be reproduced.
"""

import numpy as np

def build_adjacency_matrix_6(num_joints: int, seed: int = None) -> tuple[np.ndarray, list[int]]:
    """
    Constructs an adjacency matrix representing a robot joint structure with shuffled joint node order.

    Node Indexing:
    - Joint nodes:          indices 0 to num_joints-1 (shuffled)
    - End-effector node:    index num_joints
    - Residual (error) node: index num_joints + 1

    Graph Structure:
    - Joint nodes are connected in a chain, but the order is randomly shuffled.
    - Each joint node is also connected to the end-effector node.
    - All nodes are connected to the residual node.
    - All edges are bidirectional and unweighted (represented as 1 in the adjacency matrix).

    Args:
        num_joints (int): Number of joint nodes (must be ≥ 2).
        seed (int, optional): Seed for reproducible shuffling. If None, randomness is not seeded.

    Returns:
        tuple:
            - np.ndarray: Adjacency matrix of shape (num_joints + 2, num_joints + 2)
            - list[int]: List of joint node indices in their shuffled order
                        (e.g. [3, 0, 2, 1, 4, 5] means node 3 is first in the chain)
    """
    if num_joints < 2:
        raise ValueError("num_joints must be at least 2 to form a valid chain.")

    if seed is not None:
        np.random.seed(seed)

    # Generate and shuffle joint indices
    original_joint_indices = list(range(num_joints))
    shuffled_joint_indices = original_joint_indices.copy()
    np.random.shuffle(shuffled_joint_indices)

    num_nodes = num_joints + 2  # joint nodes + end-effector + residual
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Connect joint nodes in shuffled chain
    for i in range(num_joints - 1):
        a = shuffled_joint_indices[i]
        b = shuffled_joint_indices[i + 1]
        adjacency_matrix[a, b] = 1
        adjacency_matrix[b, a] = 1

    end_effector_node = num_joints
    residual_node = num_joints + 1

    # Connect joints to end-effector
    for joint in shuffled_joint_indices:
        adjacency_matrix[joint, end_effector_node] = 1
        adjacency_matrix[end_effector_node, joint] = 1

    # Connect joints + end-effector to residual node
    for node in shuffled_joint_indices + [end_effector_node]:
        adjacency_matrix[node, residual_node] = 1
        adjacency_matrix[residual_node, node] = 1

    return adjacency_matrix, shuffled_joint_indices

if __name__ == "__main__":
    num_joints = 6
    adj_matrix, shuffle_order = build_adjacency_matrix_6(num_joints, seed=42)

    np.save('data/adjacency_matrix_6.npy', adj_matrix)
    np.save("data/shuffle_order_for_graph_6.npy", shuffle_order)

    print("Adjacency matrix with shuffled joints saved to 'data/adjacency_matrix_6.npy'")
    print("Shuffled joint order:", shuffle_order)
    print("Adjacency Matrix:\n", adj_matrix)