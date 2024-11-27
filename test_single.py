# test_single.py

import os
import torch
import numpy as np
import yaml
from model.ASTGCN_single import make_model
import joblib
from lib.evaluation_metrics import masked_mape, masked_mse, masked_mae, masked_r2_score
import matplotlib.pyplot as plt

def test_model(config):
    # Load test data
    test_data = np.load(config['test_data_file'])
    inputs_test = test_data['inputs']
    targets_test = test_data['targets']

    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs_test).float()
    targets_tensor = torch.from_numpy(targets_test).float()

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])

    # Initialize model
    model = make_model(
        DEVICE=DEVICE,
        nb_block=config['model']['nb_block'],
        in_channels=config['model']['in_channels'],
        K=config['model']['K'],
        nb_chev_filter=config['model']['nb_chev_filter'],
        nb_time_filter=config['model']['nb_time_filter'],
        time_strides=config['model']['time_strides'],
        adj_mx=adj_mx,
        num_for_predict=config['model']['num_for_predict'],
        len_input=config['model']['len_input'] + 1,
        num_of_vertices=config['model']['num_of_vertices']
    )
    model.to(DEVICE)

    # Load the best saved model
    model_save_dir = os.path.join(config['logging']['model_save_dir'], 'ASTGCN_single')
    model_identifier = config['single_target_variable']  # Use the target variable as the model identifier

    # Filter model files by both the prefix and the identifier
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith(f'astgcn_single_best_{model_identifier}_') and f.endswith('.pth')]

    if not model_files:
        raise FileNotFoundError(f"No saved model found in the specified directory for target variable '{model_identifier}'.")
    else:
        model_files.sort()
        model_name = model_files[-1]  # Load the most recent model
    model_path = os.path.join(model_save_dir, model_name)

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")

    model.eval()

    # Make predictions
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(DEVICE)
        targets_tensor = targets_tensor.to(DEVICE)

        outputs = model(inputs_tensor)  # Shape: (num_samples, N, num_for_predict)
        # Extract outputs for the target node (node 7)
        outputs_target_node = outputs[:, 7, :].squeeze(-1)  # Shape: (num_samples,)

    # Inverse transform the predictions and targets
    scalers = joblib.load(config['scalers_file'])
    target_scaler = scalers['target_scaler']

    outputs_np = outputs_target_node.cpu().numpy().reshape(-1, 1)
    outputs_inverse = target_scaler.inverse_transform(outputs_np).reshape(-1)
    targets_np = targets_tensor.cpu().numpy().reshape(-1, 1)
    targets_inverse = target_scaler.inverse_transform(targets_np).reshape(-1)

    # Compute residuals
    residuals = targets_inverse - outputs_inverse

    # Compute metrics
    mse = masked_mse(outputs_inverse, targets_inverse, null_val=0)
    rmse = np.sqrt(mse)
    mape = masked_mape(outputs_inverse, targets_inverse, null_val=0)
    mae = masked_mae(outputs_inverse, targets_inverse, null_val=0)
    r2_score = masked_r2_score(outputs_inverse, targets_inverse, null_val=0)

    # Print metrics
    print("Test Results:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")
    print(f"R-squared (R²): {r2_score:.6f}")

    # Save results and plots
    results_dir = config['logging']['results_single_dir']
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    model_identifier = os.path.splitext(model_name)[0]
    model_results_dir = os.path.join(results_dir, model_identifier)
    if not os.path.isdir(model_results_dir):
        os.makedirs(model_results_dir)

    # Add target variable to file names
    target_variable = config['single_target_variable']

    # Time Series Plot
    plt.figure(figsize=(12, 6))
    plt.plot(targets_inverse, label='Actual')
    plt.plot(outputs_inverse, label='Predicted')
    plt.legend()
    plt.title(f'Actual vs. Predicted {target_variable}')
    plt.xlabel('Sample Index')
    plt.ylabel(target_variable)
    plt.tight_layout()
    plot_path = os.path.join(model_results_dir, f"{model_identifier}_timeseries_{target_variable}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Time series plot saved to {plot_path}")

    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(targets_inverse, outputs_inverse, alpha=0.5)
    plt.plot([targets_inverse.min(), targets_inverse.max()],
             [targets_inverse.min(), targets_inverse.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scatter Plot for {target_variable}')
    plt.tight_layout()
    scatter_plot_path = os.path.join(model_results_dir, f"{model_identifier}_scatter_{target_variable}.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Scatter plot saved to {scatter_plot_path}")

    # Residual Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(outputs_inverse, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=outputs_inverse.min(), xmax=outputs_inverse.max(), colors='r', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {target_variable}')
    plt.tight_layout()
    residual_plot_path = os.path.join(model_results_dir, f"{model_identifier}_residual_{target_variable}.png")
    plt.savefig(residual_plot_path)
    plt.close()
    print(f"Residual plot saved to {residual_plot_path}")

    # Error Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'Error Histogram for {target_variable}')
    plt.tight_layout()
    histogram_path = os.path.join(model_results_dir, f"{model_identifier}_histogram_{target_variable}.png")
    plt.savefig(histogram_path)
    plt.close()
    print(f"Error histogram saved to {histogram_path}")

    # Save metrics to a text file
    metrics_path = os.path.join(model_results_dir, f"{model_identifier}_metrics_{target_variable}.txt")
    with open(metrics_path, 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%\n")
        f.write(f"R-squared (R²): {r2_score:.6f}\n")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # Load configuration
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)

    test_model(config)
