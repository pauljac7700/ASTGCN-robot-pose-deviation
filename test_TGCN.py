# test_TGCN.py

import os
import torch
import numpy as np
import yaml
from model.TGCN import TGCNWithGlobalOutput  # Ensure this imports your TGCN module
import joblib
from lib.evaluation_metrics import masked_mape, masked_mse, masked_mae, masked_r2_score
import matplotlib.pyplot as plt

def test_model(config):
    # Load test data
    test_data = np.load(config['test_data_file'])
    inputs_test = test_data['inputs']  # Shape: (num_samples, num_nodes=8, in_channels=6, seq_len)
    targets_test = test_data['targets']  # Shape: (num_samples, num_targets)

    # Convert to PyTorch tensors
    # TGCN expects inputs of shape (batch_size, seq_len, num_nodes, in_channels)
    inputs_tensor = torch.from_numpy(inputs_test.transpose(0, 3, 1, 2)).float()
    targets_tensor = torch.from_numpy(targets_test).float()

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])

    # Initialize model
    in_channels = config['model']['in_channels']  # Should be 6
    hidden_dim = config['model']['hidden_dim']
    tgcn_model = TGCNWithGlobalOutput(adj=adj_mx, in_channels=in_channels, hidden_dim=hidden_dim)
    tgcn_model.to(DEVICE)

    # Load the best saved model
    model_save_dir = config['logging']['model_save_dir']
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith('tgcn_best_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No saved model found in the specified directory.")
    else:
        # Assuming the latest model is the best one
        model_files.sort()
        model_name = model_files[-1]
    model_path = os.path.join(model_save_dir, model_name)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    tgcn_model.load_state_dict(checkpoint['tgcn_state_dict'])
    print(f"Loaded model from {model_path}")

    tgcn_model.eval()

    # Make predictions
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(DEVICE)
        targets_tensor = targets_tensor.to(DEVICE)

        outputs = tgcn_model(inputs_tensor)  # Outputs shape: (num_samples, num_targets)

    # Inverse transform the predictions and targets
    scalers = joblib.load(config['scalers_file'])
    target_scalers = scalers['target_scalers']

    outputs_np = outputs.cpu().numpy()
    targets_np = targets_tensor.cpu().numpy()

    outputs_inverse = np.zeros_like(outputs_np)
    targets_inverse = np.zeros_like(targets_np)

    # Ensure target_variables are defined
    target_variables = ['x_dif', 'y_dif', 'z_dif', 'rx_dif', 'ry_dif', 'rz_dif']

    for idx, target_var in enumerate(target_variables):
        scaler = target_scalers[target_var]
        outputs_inverse[:, idx] = scaler.inverse_transform(outputs_np[:, idx].reshape(-1, 1)).reshape(-1)
        targets_inverse[:, idx] = scaler.inverse_transform(targets_np[:, idx].reshape(-1, 1)).reshape(-1)

    # Compute metrics for each target variable
    metrics = {}
    mse_list = []
    rmse_list = []
    mae_list = []
    mape_list = []
    r2_list = []
    residuals_dict = {}  # To store residuals for each target variable
    for idx, target_var in enumerate(target_variables):
        mse = masked_mse(outputs_inverse[:, idx], targets_inverse[:, idx], null_val=0)
        rmse = np.sqrt(mse)
        mape = masked_mape(outputs_inverse[:, idx], targets_inverse[:, idx], null_val=0)
        mae = masked_mae(outputs_inverse[:, idx], targets_inverse[:, idx], null_val=0)
        r2_score = masked_r2_score(outputs_inverse[:, idx], targets_inverse[:, idx], null_val=0)
        metrics[target_var] = {'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'MAE': mae, 'R2': r2_score}
        # Append to lists for overall metrics
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2_score)
        # Store residuals
        residuals = targets_inverse[:, idx] - outputs_inverse[:, idx]
        residuals_dict[target_var] = residuals

    # Compute mean metrics over all target variables
    mean_metrics = {
        'MSE': np.mean(mse_list),
        'RMSE': np.mean(rmse_list),
        'MAE': np.mean(mae_list),
        'MAPE': np.mean(mape_list),
        'R2': np.mean(r2_list)
    }

    # Print metrics
    print("\nTest Results:")
    for target_var, metric in metrics.items():
        print(f"Metrics for {target_var}:")
        print(f"  Mean Squared Error (MSE): {metric['MSE']:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {metric['RMSE']:.6f}")
        print(f"  Mean Absolute Error (MAE): {metric['MAE']:.6f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {metric['MAPE']:.6f}%")
        print(f"  R-squared (R²): {metric['R2']:.6f}")

    # Print mean metrics
    print("\nMean Metrics over all target variables:")
    print(f"  Mean Squared Error (MSE): {mean_metrics['MSE']:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {mean_metrics['RMSE']:.6f}")
    print(f"  Mean Absolute Error (MAE): {mean_metrics['MAE']:.6f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mean_metrics['MAPE']:.6f}%")
    print(f"  R-squared (R²): {mean_metrics['R2']:.6f}")

    # Save results and plots

    # Create results directory if it doesn't exist
    results_dir = config['logging']['results_dir']
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Generate a filename prefix based on the model used
    model_identifier = os.path.splitext(model_name)[0]  # Remove '.pth' extension

    # Create a subdirectory named after the model identifier
    model_results_dir = os.path.join(results_dir, model_identifier)
    if not os.path.isdir(model_results_dir):
        os.makedirs(model_results_dir)

    num_targets = len(target_variables)

    # Time Series Plot
    plt.figure(figsize=(12, 6 * num_targets))
    for idx, target_var in enumerate(target_variables):
        plt.subplot(num_targets, 1, idx + 1)
        plt.plot(targets_inverse[:, idx], label='Actual')
        plt.plot(outputs_inverse[:, idx], label='Predicted')
        plt.legend()
        plt.title(f'Actual vs. Predicted {target_var}')
        plt.xlabel('Sample Index')
        plt.ylabel(target_var)
    plt.tight_layout()
    # Save the figure
    plot_filename = f"{model_identifier}_timeseries.png"
    plot_path = os.path.join(model_results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Time series plots saved to {plot_path}")

    # Scatter Plots and Residual Plots
    for idx, target_var in enumerate(target_variables):
        # Scatter Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(targets_inverse[:, idx], outputs_inverse[:, idx], alpha=0.5)
        plt.plot([targets_inverse[:, idx].min(), targets_inverse[:, idx].max()],
                 [targets_inverse[:, idx].min(), targets_inverse[:, idx].max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Scatter Plot for {target_var}')
        plt.tight_layout()
        scatter_plot_filename = f"{model_identifier}_scatter_{target_var}.png"
        scatter_plot_path = os.path.join(model_results_dir, scatter_plot_filename)
        plt.savefig(scatter_plot_path)
        plt.close()
        print(f"Scatter plot for {target_var} saved to {scatter_plot_path}")

        # Residual Plot
        residuals = residuals_dict[target_var]
        plt.figure(figsize=(6, 6))
        plt.scatter(outputs_inverse[:, idx], residuals, alpha=0.5)
        plt.hlines(y=0, xmin=outputs_inverse[:, idx].min(), xmax=outputs_inverse[:, idx].max(), colors='r', linestyles='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for {target_var}')
        plt.tight_layout()
        residual_plot_filename = f"{model_identifier}_residual_{target_var}.png"
        residual_plot_path = os.path.join(model_results_dir, residual_plot_filename)
        plt.savefig(residual_plot_path)
        plt.close()
        print(f"Residual plot for {target_var} saved to {residual_plot_path}")

        # Error Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title(f'Error Histogram for {target_var}')
        plt.tight_layout()
        histogram_filename = f"{model_identifier}_histogram_{target_var}.png"
        histogram_path = os.path.join(model_results_dir, histogram_filename)
        plt.savefig(histogram_path)
        plt.close()
        print(f"Error histogram for {target_var} saved to {histogram_path}")

    # Save metrics to a text file
    metrics_filename = f"{model_identifier}_metrics.txt"
    metrics_path = os.path.join(model_results_dir, metrics_filename)
    with open(metrics_path, 'w') as f:
        f.write("Test Results:\n")
        for target_var, metric in metrics.items():
            f.write(f"Metrics for {target_var}:\n")
            f.write(f"  Mean Squared Error (MSE): {metric['MSE']:.6f}\n")
            f.write(f"  Root Mean Squared Error (RMSE): {metric['RMSE']:.6f}\n")
            f.write(f"  Mean Absolute Error (MAE): {metric['MAE']:.6f}\n")
            f.write(f"  Mean Absolute Percentage Error (MAPE): {metric['MAPE']:.6f}%\n")
            f.write(f"  R-squared (R²): {metric['R2']:.6f}\n\n")

        # Write mean metrics
        f.write("Mean Metrics over all target variables:\n")
        f.write(f"  Mean Squared Error (MSE): {mean_metrics['MSE']:.6f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {mean_metrics['RMSE']:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mean_metrics['MAE']:.6f}\n")
        f.write(f"  Mean Absolute Percentage Error (MAPE): {mean_metrics['MAPE']:.6f}%\n")
        f.write(f"  R-squared (R²): {mean_metrics['R2']:.6f}\n")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # Load configuration
    with open('config_TGCN.yaml') as f:
        config = yaml.safe_load(f)

    test_model(config)
