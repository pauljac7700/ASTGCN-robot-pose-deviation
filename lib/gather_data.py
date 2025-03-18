#gather_data
import random
import math
import numpy as np
import csv
from typing import List
import matplotlib.pyplot as plt

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
    rx: float = 3.938,
    ry: float = 2.004,
    rz: float = -1.385
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

# Example usage
create_shape(
    min_height=150, 
    max_height=800, 
    min_radius=400, 
    max_radius=910, 
    min_angle=106, 
    max_angle=208, 
    num_points=1000, 
    mode="grid", 
    output_file="grid_points.csv"
)

create_shape(
    min_height=150, 
    max_height=800, 
    min_radius=400, 
    max_radius=910, 
    min_angle=106, 
    max_angle=208, 
    num_points=1000, 
    mode="random", 
    output_file="random_points.csv"
)

visualize_points_from_csv('random_points.csv', 'random')
visualize_points_from_csv('grid_points.csv', 'grid')