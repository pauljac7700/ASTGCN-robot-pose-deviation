import pandas as pd
import os

# Load the CSV file
file_path = 'data/measurements_grid.csv'
data = pd.read_csv(file_path, delimiter=';', decimal=',')
'''
# Remove unnecessary columns
data = data.drop([
    'measurement_temperature', 'measurement_std_x', 'measurement_std_y',
    'measurement_std_z', 'measurement_std_rx', 'measurement_std_ry', 'measurement_std_rz'
], axis=1)
'''
# Renaming the columns
joint_column_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]
new_joint_names = [f'joint_{i+1}' for i in range(len(joint_column_names))]

# Creating the mapping for renaming columns
column_rename_mapping = [
    ('Goal x', 'x_dif'),
    ('Goal y', 'y_dif'),
    ('Goal z', 'z_dif'),
    ('Goal roll', 'rx_dif'),
    ('Goal pitch', 'ry_dif'),
    ('Goal yaw', 'rz_dif'),
    ('measurement_x', 'x_measure'),
    ('measurement_y', 'y_measure'),
    ('measurement_z', 'z_measure'),
    ('measurement_rx', 'rx_measure'),
    ('measurement_ry', 'ry_measure'),
    ('measurement_rz', 'rz_measure')
]
column_rename_mapping.extend(zip(joint_column_names, new_joint_names))

# Applying all column renames in one go
data = data.rename(columns=dict(column_rename_mapping))

# Check and sort by 'timestamp' column if it exists and varies
if 'Timestamp' in data.columns:
    if data['Timestamp'].nunique() > 1:
        data = data.sort_values(by='Timestamp')
    else:
        data = data.drop(['Timestamp'],axis=1)
data = data.reset_index(drop=True)
data.index.name = 'step_order'

# Compute the pose error (differences between measurements and set positions)
data['x_set'] = data['x_measure'] - data['x_dif']
data['y_set'] = data['y_measure'] - data['y_dif']
data['z_set'] = data['z_measure'] - data['z_dif']
data['rx_set'] = data['rx_measure'] - data['rx_dif']
data['ry_set'] = data['ry_measure'] - data['ry_dif']
data['rz_set'] = data['rz_measure'] - data['rz_dif']

# Drop measurement columns if no longer needed
data = data.drop(columns=['x_measure', 'y_measure', 'z_measure', 'rx_measure', 'ry_measure', 'rz_measure'])

# Save the cleaned CSV file with '_cleaned' appended to the original filename
new_file_path = os.path.splitext(file_path)[0] + '_cleaned.csv'
data.to_csv(new_file_path)