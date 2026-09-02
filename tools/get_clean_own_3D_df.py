"""Standalone: clean the recorded 3D UR5 measurements into the training schema.

Renames the raw joint and pose columns, drops incomplete rows and writes a
``_cleaned`` CSV next to the input. The input path is set at the top of the file
and is meant to be edited before running. Run directly; nothing imports it.
"""

import pandas as pd
import os
import numpy as np

# Load the CSV file with explicit indexing
file_path = 'data/3D_datasets/3D_UR5_v3_random.csv'
data = pd.read_csv(file_path, delimiter=',', decimal='.', index_col=None)

# Trim column names to remove leading/trailing spaces
data.columns = data.columns.str.strip()

# Print column names before renaming to debug
print("Column names before renaming:", data.columns.tolist())

# Renaming joint columns
joint_column_names = ['j1_target','j2_target','j3_target','j4_target','j5_target','j6_target'] #['j1_t', 'j2_t', 'j3_t', 'j4_t', 'j5_t', 'j6_t'] 
new_joint_names = [f'joint_{i+1}' for i in range(len(joint_column_names))]
column_rename_mapping = dict(zip(joint_column_names, new_joint_names))

# Apply renaming only to existing columns
data = data.rename(columns=column_rename_mapping)

# Convert joint angles from radians to degrees
for col in new_joint_names:
    if col in data.columns:
        data[col] = np.degrees(data[col])  # Convert to degrees

# Print renamed columns to verify
print("Column names after renaming:", data.columns.tolist())

# Set the index name explicitly
data.index.name = 'step_order'

# Convert position columns from meters to millimeters
position_columns = ['x_t', 'y_t', 'z_t', 'x_m', 'y_m', 'z_m']
data[position_columns] *= 1000  # Convert to millimeters

# Compute pose errors safely
data['x_dif'] = data['x_m'] - data['x_t']
data['y_dif'] = data['y_m'] - data['y_t']
data['z_dif'] = data['z_m'] - data['z_t']

data = data.drop(columns=['x_m', 'y_m', 'z_m'])

# Reorder columns so joint columns are last
joint_columns = new_joint_names  # ['joint_1', 'joint_2', ..., 'joint_6']
other_columns = [col for col in data.columns if col not in joint_columns]
data = data[other_columns + joint_columns]  # Ensures joints are last

# Print final column order for verification
print("Final column order:", data.columns.tolist())

# Save cleaned CSV file
new_file_path = os.path.splitext(file_path)[0] + '_cleaned.csv'
data.to_csv(new_file_path)

# Confirm success
print(f"Renaming and reordering successful. Cleaned file saved as: {new_file_path}")
