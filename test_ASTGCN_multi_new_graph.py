# test_ASTGCN_multi_new_graph.py

import os
import torch
import numpy as np
import yaml
from model.ASTGCN_single import make_model
import joblib
from lib.evaluation_metrics import (
    masked_mape,
    masked_mse,
    masked_mae,
    masked_r2_score,
    masked_smape,
    masked_mdape
)
import matplotlib.pyplot as plt
import pandas as pd

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
        len_input=config['model']['len_input'] + 1,  # Adjusted for extended input sequence
        num_of_vertices=config['model']['num_of_vertices']
    )
    model.to(DEVICE)

    # Load the best saved model
    model_save_dir = config['logging']['model_save_dir']
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith('astgcn_new_graph_best_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No saved model found in the specified directory.")
    else:
        model_files.sort()
        model_name = model_files[-1]
    model_path = os.path.join(model_save_dir, model_name)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")

    model.eval()

    # Make predictions
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(DEVICE)
        targets_tensor = targets_tensor.to(DEVICE)

        outputs = model(inputs_tensor)  # Shape: (num_samples, N, num_for_predict)

    # Extract outputs for the target pose dimensions (nodes 6 to 11)
    target_nodes = list(range(6, 12))  # Corresponding to pose dimensions
    outputs_target_nodes = outputs[:, target_nodes, :].squeeze(-1)  # Shape: (num_samples, 6)

    # Load scalers
    scalers = joblib.load(config['scalers_file'])  # Contains scalers for both inputs and targets

    # Prepare output and target arrays
    target_variables = config['target_variables']  # ['x_dif', 'y_dif', 'z_dif', 'rx_dif', 'ry_dif', 'rz_dif']
    num_samples = outputs_target_nodes.shape[0]
    num_targets = len(target_variables)

    outputs_inverse = np.zeros((num_samples, num_targets))
    targets_inverse = np.zeros((num_samples, num_targets))
    residuals_dict = {}

    # Initialize metric lists
    metrics = {}
    mse_list = []
    rmse_list = []
    mae_list = []
    mape_list = []
    smape_list = []
    mdape_list = []
    r2_list = []

    # Compute metrics for each target variable
    for idx, target_var in enumerate(target_variables):
        # Use the corresponding scaler for the target variable
        scaler = scalers.get(target_var)
        if scaler is None:
            raise KeyError(f"Scaler for '{target_var}' not found in scalers dictionary.")

        # Inverse transform the outputs and targets
        outputs_np = outputs_target_nodes[:, idx].cpu().numpy().reshape(-1, 1)
        outputs_inv = scaler.inverse_transform(outputs_np).reshape(-1)
        outputs_inverse[:, idx] = outputs_inv

        targets_np = targets_tensor[:, idx].cpu().numpy().reshape(-1, 1)
        targets_inv = scaler.inverse_transform(targets_np).reshape(-1)
        targets_inverse[:, idx] = targets_inv

        # Compute residuals
        residuals = targets_inv - outputs_inv
        residuals_dict[target_var] = residuals

        # Compute metrics
        mse = masked_mse(outputs_inv, targets_inv, null_val=0)
        rmse = np.sqrt(mse)
        mape = masked_mape(outputs_inv, targets_inv, null_val=0)
        smape = masked_smape(outputs_inv, targets_inv, null_val=0)
        mdape = masked_mdape(outputs_inv, targets_inv, null_val=0)
        mae = masked_mae(outputs_inv, targets_inv, null_val=0)
        r2_score = masked_r2_score(outputs_inv, targets_inv, null_val=0)

        metrics[target_var] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'sMAPE': smape, 'MdAPE': mdape, 'R2': r2_score}

        # Append to lists for overall metrics
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        smape_list.append(smape)
        mdape_list.append(mdape)
        r2_list.append(r2_score)

    # Compute mean metrics over all target variables
    mean_metrics = {
        'MSE': np.mean(mse_list),
        'RMSE': np.mean(rmse_list),
        'MAE': np.mean(mae_list),
        'MAPE': np.mean(mape_list),
        'sMAPE': np.mean(smape_list),
        'MdAPE': np.median(mdape_list),
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
        print(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {metric['sMAPE']:.6f}%")
        print(f"  Median Absolute Percentage Error (MdAPE): {metric['MdAPE']:.6f}%")
        print(f"  R-squared (R²): {metric['R2']:.6f}\n")

    # Print mean metrics
    print("Mean Metrics over all target variables:")
    print(f"  Mean Squared Error (MSE): {mean_metrics['MSE']:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {mean_metrics['RMSE']:.6f}")
    print(f"  Mean Absolute Error (MAE): {mean_metrics['MAE']:.6f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mean_metrics['MAPE']:.6f}%")
    print(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {mean_metrics['sMAPE']:.6f}%")
    print(f"  Median Absolute Percentage Error (MdAPE): {mean_metrics['MdAPE']:.6f}%")
    print(f"  R-squared (R²): {mean_metrics['R2']:.6f}")

    # Save results and plots
    # Create a directory under model_save_dir with the model name as identifier
    model_identifier = os.path.splitext(model_name)[0]  # Remove '.pth' extension
    model_results_dir = os.path.join(config['logging']['results_dir'], model_identifier)
    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)
        print(f"Created directory for results and plots: {model_results_dir}")

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
    for idx, target_var in enumerate(config['target_variables']):
        # Scatter Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(targets_inverse[:, idx], outputs_inverse[:, idx], alpha=0.5)
        min_val = min(targets_inverse[:, idx].min(), outputs_inverse[:, idx].min())
        max_val = max(targets_inverse[:, idx].max(), outputs_inverse[:, idx].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
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
            f.write(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {metric['sMAPE']:.6f}%\n")
            f.write(f"  Median Absolute Percentage Error (MdAPE): {metric['MdAPE']:.6f}%\n")
            f.write(f"  R-squared (R²): {metric['R2']:.6f}\n\n")

        # Write mean metrics
        f.write("Mean Metrics over all target variables:\n")
        f.write(f"  Mean Squared Error (MSE): {mean_metrics['MSE']:.6f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {mean_metrics['RMSE']:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mean_metrics['MAE']:.6f}\n")
        f.write(f"  Mean Absolute Percentage Error (MAPE): {mean_metrics['MAPE']:.6f}%\n")
        f.write(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {mean_metrics['sMAPE']:.6f}%\n")
        f.write(f"  Median Absolute Percentage Error (MdAPE): {mean_metrics['MdAPE']:.6f}%\n")
        f.write(f"  R-squared (R²): {mean_metrics['R2']:.6f}\n")
    print(f"Metrics saved to {metrics_path}")

    # Create Excel File with Actual, Predicted, Residuals
    df_data = {}
    df_data['Sample Index'] = np.arange(len(targets_inverse))

    for idx, target_var in enumerate(config['target_variables']):
        df_data[f"{target_var}_Actual"] = targets_inverse[:, idx]
        df_data[f"{target_var}_Predicted"] = outputs_inverse[:, idx]
        df_data[f"{target_var}_Residual"] = residuals_dict[target_var]

    df = pd.DataFrame(df_data)

    excel_filename = f"{model_identifier}_results.xlsx"
    excel_path = os.path.join(model_results_dir, excel_filename)

    df.to_excel(excel_path, index=False)
    print(f"Detailed results (Actual, Predicted, Residuals) saved to {excel_path}")

if __name__ == "__main__":
    # Load configuration
    with open('config_ASTGCN_new_graph.yaml') as f:
        config = yaml.safe_load(f)

    test_model(config)
