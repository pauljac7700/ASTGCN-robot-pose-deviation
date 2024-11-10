# baseline_linear_regression.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib


# Load datasets
train_data = np.load('data/train_data.npz')
test_data = np.load('data/test_data.npz')

inputs_train = train_data['inputs']  # Shape: (num_samples, num_nodes, num_features, len_input)
targets_train = train_data['targets']  # Shape: (num_samples, 1)
inputs_test = test_data['inputs']  # Shape: (num_samples, num_nodes, num_features, len_input)
targets_test = test_data['targets']  # Shape: (num_samples, 1)

# Flatten the inputs for linear regression
# We'll concatenate the features across nodes and time steps
# New shape: (num_samples, num_nodes * num_features * len_input)

num_samples_train = inputs_train.shape[0]
num_samples_test = inputs_test.shape[0]
num_nodes = inputs_train.shape[1]
num_features = inputs_train.shape[2]
len_input = inputs_train.shape[3]

# Reshape inputs
inputs_train_flat = inputs_train.reshape(num_samples_train, -1)
inputs_test_flat = inputs_test.reshape(num_samples_test, -1)

# The targets are already in shape (num_samples, 1), suitable for linear regression

# Load scalers
scalers = joblib.load('data/scalers.joblib')
target_scaler = scalers['target_scaler']

# Do not inverse transform the targets before training
# Keep the targets scaled during training

# Train the linear regression model on scaled data
lr_model = LinearRegression()
lr_model.fit(inputs_train_flat, targets_train)

# Make predictions on the test set
predictions_scaled = lr_model.predict(inputs_test_flat)

# Inverse transform the predictions and targets for evaluation
predictions = target_scaler.inverse_transform(predictions_scaled)
targets_test_inverse = target_scaler.inverse_transform(targets_test)

# Evaluate the model
mse = mean_squared_error(targets_test_inverse, predictions)
mape = mean_absolute_percentage_error(targets_test_inverse, predictions) * 100

print(f"Linear Regression Test MSE: {mse:.4f}")
print(f"Linear Regression Test MAPE: {mape:.4f}%")

# Detailed summary
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(targets_test_inverse, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted x_dif')
plt.xlabel('Sample Index')
plt.ylabel('x_dif')
plt.show()