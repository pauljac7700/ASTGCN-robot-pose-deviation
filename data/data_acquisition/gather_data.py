"""Generate measurement poses and record them into the training CSVs.

Samples the poses to be commanded, whether on a grid or at random, drives them
through the UR5 model, and writes the commanded and measured values into the CSV
schema the preprocessing scripts read. Grid poses are used for calibration and
training, random poses for testing.
"""

import random
import math
import numpy as np
import csv
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from UR5_robot_model import UR5_robot_model, rotation_vector_to_matrix, build_transformation_matrix
from spatialmath import SE3
from tqdm import tqdm


def create_shape(
    min_height: float,
    max_height: float,
    min_radius: float,
    max_radius: float,
    min_angle: float,
    max_angle: float,
    num_points: int = 100,
    mode: str = "random",
    output_file: str = "points_with_orientation.csv",
    rx: float = 4.041,
    ry: float = 1.516,
    rz: float = -1.466
) -> None:
    '''
    Creates a 3D partial cylindrical shape and generates points inside the shape, including orientation.

    Parameters:
    - min_height (float): Minimum height of the cylinder.
    - max_height (float): Maximum height of the cylinder.
    - min_radius (float): Minimum radius of the cylinder.
    - max_radius (float): Maximum radius of the cylinder.
    - min_angle (float): Minimum angle (in degrees) of the sector.
    - max_angle (float): Maximum angle (in degrees) of the sector.
    - num_points (int): Number of points to generate inside the shape (default=100).
    - mode (str): "grid" or "random" to specify point generation mode (default="random").
    - output_file (str): Name of the output CSV file (default="points_with_orientation.csv").
    - rx (float): Fixed rotation value for x in radians (default=3.938).
    - ry (float): Fixed rotation value for y in radians (default=2.004).
    - rz (float): Fixed rotation value for z in radians (default=-1.385).

    Returns:
    - None
    '''
    if mode not in ["grid", "random"]:
        raise ValueError(f"Invalid mode '{mode}'. Use 'grid' or 'random'.")

    # Convert angles from degrees to radians
    min_angle_rad = math.radians(min_angle)
    max_angle_rad = math.radians(max_angle)

    points = []

    if mode == "grid":
        # Create grid-like points
        height_steps = int(np.cbrt(num_points))
        radius_steps = height_steps
        angle_steps = height_steps

        z_values = np.linspace(min_height, max_height, height_steps)
        r_values = np.linspace(min_radius, max_radius, radius_steps)
        theta_values = np.linspace(min_angle_rad, max_angle_rad, angle_steps)

        z_grid, r_grid, theta_grid = np.meshgrid(z_values, r_values, theta_values, indexing="ij")

        for z, r, theta in zip(z_grid.ravel(), r_grid.ravel(), theta_grid.ravel()):
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            # Append the point with orientation
            points.append((x, y, z, rx, ry, rz))
    elif mode == "random":
        # Generate random points
        for _ in range(num_points):
            z = random.uniform(min_height, max_height)
            r = random.uniform(min_radius, max_radius)
            theta = random.uniform(min_angle_rad, max_angle_rad)

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            # Append the point with orientation
            points.append((x, y, z, rx, ry, rz))

    # Write points to CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z", "rx", "ry", "rz"])  # Header
        writer.writerows(points)

    print(f"Points saved to {output_file}")


def visualize_points_from_csv(csv_file: str, mode: str) -> None:
    '''
    Visualizes 3D points with orientation from a CSV file.

    Parameters:
    - csv_file (str): Path to the CSV file containing the points.

    Returns:
    - None
    '''
    # Read points from CSV
    points = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            points.append([float(val) for val in row[:3]])  # Only take x, y, z

    points = np.array(points)  # Convert to NumPy array

    # Separate x, y, z for plotting
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Height (z)', rotation=270, labelpad=15)

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'3D Visualization of {mode} Points')

    # Show plot
    plt.show()


def get_joint_angles(point_pth, robot, output_file="joint_angles.csv"):
    """

    :param point_pth: Cartesian space pose
    :param robot: Robot model
    :param output_file: Output file name
    :return:
    """
    points = pd.read_csv(point_pth, header=0)
    points = points.to_numpy()
    q0_ = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
    q_max = np.array([2 * np.pi, 0, 5 * np.pi / 6, np.pi, np.pi, 2 * np.pi])
    q_min = np.array([-2 * np.pi, -17 * np.pi / 18, -5 * np.pi / 6, -np.pi, -np.pi, -2 * np.pi])

    joint_angles = []
    success_point = []
    false_point = []
    flag = False
    for i in tqdm(range(points.shape[0])):
        point = points[i, :]

        xyz = point[:3]
        rxyz = point[3:]
        R = rotation_vector_to_matrix(rxyz)
        T = build_transformation_matrix(R, xyz)
        Tep = SE3(T)

        if flag == False:
            q0 = q0_
        else:
            q0 = q

        q_ik = robot.ikine_LM(Tep, q0=q0)
        q = q_ik.q

        if judgment_of_joint_angles(q):
            joint_angles.append(q)
            flag = True
            success_point.append(i)
        else:
            flag = False
            count = 0
            while not judgment_of_joint_angles(q):
                q0 = np.random.uniform(q_min, q_max)
                q_ik = robot.ikine_LM(Tep, q0=q0)
                q = q_ik.q

                count += 1
                if count == 1000:
                    false_point.append(i)
                    print(f"point {i} is False")
                    break
            if count < 1000:
                flag = True
                success_point.append(i)
                joint_angles.append(q)

    joint_angles = np.array(joint_angles)
    joint_angles = pd.DataFrame(joint_angles)
    joint_angles.to_csv(output_file)
    print(f"joint_angles saved to {output_file}")
    print("Success points: ", success_point)
    print("False points: ", false_point)

    return joint_angles


def judgment_of_joint_angles(joint_angles):
    """
    Determine whether there will be a collision at a given joint Angle

    :param joint_angles: joint angles calculated by the inverse kinematics
    :return:
    """
    q_max = np.array([2 * np.pi, 0, 5 * np.pi / 6, np.pi, np.pi, 2 * np.pi])
    q_min = np.array([-2 * np.pi, -17 * np.pi / 18, -5 * np.pi / 6, -np.pi, -np.pi, -2 * np.pi])

    q23_max = 0
    q23_min = -np.pi

    # joint limit 1
    joint_limit1 = np.all((joint_angles >= q_min) & (joint_angles <= q_max))

    # joint limit 2
    scale = 0.39225 / (0.39225 + 0.425)
    q23 = joint_angles[1] + scale * joint_angles[2]
    joint_limit2 = (q23 >= q23_min) & (q23 <= q23_max)

    return joint_limit1 & joint_limit2


if __name__ == "__main__":
    # Example usage
    create_shape(
        min_height=300/1000,
        max_height=800/1000,
        min_radius=500/1000,
        max_radius=910/1000,
        min_angle=106,
        max_angle=208,
        num_points=1000,
        mode="grid",
        output_file="grid_points.csv"
    )

    create_shape(
        min_height=300/1000,
        max_height=800/1000,
        min_radius=500/1000,
        max_radius=910/1000,
        min_angle=106,
        max_angle=208,
        num_points=1000,
        mode="random",
        output_file="random_points.csv"
    )

    visualize_points_from_csv('random_points.csv', 'random')
    visualize_points_from_csv('grid_points.csv', 'grid')

    ur5_robot = UR5_robot_model()
    grid_joint_angles = get_joint_angles("grid_points.csv", ur5_robot, output_file="grid_joint_angles.csv")
    random_joint_angles = get_joint_angles("random_points.csv", ur5_robot, output_file="random_joint_angles.csv")