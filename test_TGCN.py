"""Evaluate the TGCN baseline on the held-out test split.

TGCN uses the same graph as the ASTGCN but without the attention mechanism, so the
gap between the two isolates what attention contributes.
"""

# test_TGCN.py

import argparse
import os
import torch
import numpy as np
import yaml
from model.TGCN import TGCNWithGlobalOutput
from lib.extract_number_from_filename import extract_number_from_filename
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
    dataset_dimension = config['dataset_dimension']
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    print("Dataset Dimension:", dataset_dimension)
    print("Dataset Name:", dataset_name)
    print("Dataset Type:", dataset_type)
    model_name = config['model_name']['multi']
    num_joints = config['num_joints']
    if config['prep_data_incl_past_residuals']:
        prep_data_incl_past_residuals = 'wr'
        print("Including past residuals in input features.")
    else:
        prep_data_incl_past_residuals = 'nr'
        print("Excluding past residuals from input features.")

    graph_nr = extract_number_from_filename(config['adjacency_matrix_file'])
    print(f"Graph number: {graph_nr}")

    test_data = np.load(f'data/test_data_{graph_nr}_{prep_data_incl_past_residuals}.npz')
    inputs_test = test_data['inputs']  # 
    residuals_test = test_data['residuals']  

    # Process inputs to be compatible with TGCN model
    # TGCN now expects inputs of shape (batch_size, seq_len, num_nodes, in_channels)
    inputs_test = inputs_test.transpose(0, 3, 1, 2)  # Shape: (num_samples, seq_len, num_nodes, in_channels)

    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs_test).float()
    residuals_tensor = torch.from_numpy(residuals_test).float()

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])

    # Number of residuals
    num_residuals = len(config['residual_variables'][dataset_dimension])

    # Initialize model
    in_channels = config['model'][f'in_channels_{dataset_dimension}']
    hidden_dim = config['model']['hidden_dim']
    num_residuals = len(config['residual_variables'].get(config['dataset_dimension'], []))
    tgcn_model = TGCNWithGlobalOutput(adj=adj_mx, in_channels=in_channels, hidden_dim=hidden_dim, num_residuals=num_residuals)
    tgcn_model.to(DEVICE)

    # Load the best saved model
    model_save_dir = os.path.join('saved_models',dataset_dimension,dataset_name,dataset_type,model_name)
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith(f'{model_name}_best_{graph_nr}_{prep_data_incl_past_residuals}') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No saved model found in the specified directory.")
    else:
        model_files.sort()
        actual_model_name = model_files[-1]
    model_path = os.path.join(model_save_dir, actual_model_name)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    tgcn_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")

    tgcn_model.eval()

    # Make predictions
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(DEVICE)
        residuals_tensor = residuals_tensor.to(DEVICE)
        print("Input tensor shape:", inputs_tensor.shape)
        outputs = tgcn_model(inputs_tensor)  # (num_samples, num_targets)

    # Inverse transform the predictions and targets
    scalers = joblib.load(config['scalers_file'])
    residual_scalers = scalers['residual_scalers']

    outputs_np = outputs.cpu().numpy()
    residuals_np = residuals_tensor.cpu().numpy()

    outputs_inverse = np.zeros_like(outputs_np)
    residuals_inverse = np.zeros_like(residuals_np)

    for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]):
        scaler = residual_scalers[residual_var]
        outputs_inverse[:, idx] = scaler.inverse_transform(outputs_np[:, idx].reshape(-1, 1)).reshape(-1)
        residuals_inverse[:, idx] = scaler.inverse_transform(residuals_np[:, idx].reshape(-1, 1)).reshape(-1)

    metrics = {}
    mse_list = []
    rmse_list = []
    mae_list = []
    mape_list = []
    smape_list = []
    mdape_list = []
    r2_list = []
    residuals_dict = {}

    # Compute metrics for each target variable
    for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]):
        mse = masked_mse(outputs_inverse[:, idx], residuals_inverse[:, idx], null_val=0)
        rmse = np.sqrt(mse)
        mape = masked_mape(outputs_inverse[:, idx], residuals_inverse[:, idx], null_val=0)
        smape = masked_smape(outputs_inverse[:, idx], residuals_inverse[:, idx], null_val=0)
        mdape = masked_mdape(outputs_inverse[:, idx], residuals_inverse[:, idx], null_val=0)
        mae = masked_mae(outputs_inverse[:, idx], residuals_inverse[:, idx], null_val=0)
        r2_score = masked_r2_score(outputs_inverse[:, idx], residuals_inverse[:, idx], null_val=0)

        metrics[residual_var] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'sMAPE': smape,
            'MdAPE': mdape,
            'R2': r2_score
        }

        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        smape_list.append(smape)
        mdape_list.append(mdape)
        r2_list.append(r2_score)

        residuals = residuals_inverse[:, idx] - outputs_inverse[:, idx]
        residuals_dict[residual_var] = residuals

    mean_metrics = {
        'MSE': np.mean(mse_list),
        'RMSE': np.mean(rmse_list),
        'MAE': np.mean(mae_list),
        'MAPE': np.mean(mape_list),
        'sMAPE': np.mean(smape_list),
        'MdAPE': np.median(mdape_list),
        'R2': np.mean(r2_list)
    }

    print("\nTest Results:")
    for residual_var, metric in metrics.items():
        print(f"Metrics for {residual_var}:")
        print(f"  Mean Squared Error (MSE): {metric['MSE']:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {metric['RMSE']:.6f}")
        print(f"  Mean Absolute Error (MAE): {metric['MAE']:.6f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {metric['MAPE']:.6f}%")
        print(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {metric['sMAPE']:.6f}%")
        print(f"  Median Absolute Percentage Error (MdAPE): {metric['MdAPE']:.6f}%")
        print(f"  R-squared (R²): {metric['R2']:.6f}\n")

    print("Mean Metrics over all residual variables:")
    print(f"  Mean Squared Error (MSE): {mean_metrics['MSE']:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {mean_metrics['RMSE']:.6f}")
    print(f"  Mean Absolute Error (MAE): {mean_metrics['MAE']:.6f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mean_metrics['MAPE']:.6f}%")
    print(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {mean_metrics['sMAPE']:.6f}%")
    print(f"  Median Absolute Percentage Error (MdAPE): {mean_metrics['MdAPE']:.6f}%")
    print(f"  R-squared (R²): {mean_metrics['R2']:.6f}")

    # Save results and plots
    # Create results directory if it doesn't exist
    results_dir = os.path.join('results',dataset_dimension,dataset_name,dataset_type,model_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Generate a filename prefix based on the model used
    model_identifier = os.path.splitext(actual_model_name)[0]  # Remove '.pth' extension

    # Create a subdirectory named after the model identifier
    model_results_dir = os.path.join(results_dir, model_identifier)
    if not os.path.isdir(model_results_dir):
        os.makedirs(model_results_dir)

    plt.figure(figsize=(12, 6 * num_residuals))
    for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]):
        plt.subplot(num_residuals, 1, idx + 1)
        plt.plot(residuals_inverse[:, idx], label='Actual')
        plt.plot(outputs_inverse[:, idx], label='Predicted')
        plt.legend()
        plt.title(f'Actual vs. Predicted {residual_var}')
        plt.xlabel('Sample Index')
        plt.ylabel(residual_var)
    plt.tight_layout()
    plot_filename = f"{model_identifier}_timeseries.png"
    plot_path = os.path.join(model_results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Time series plots saved to {plot_path}")

    # Scatter Plots and Residual Plots
    for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]):
        # Scatter Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(residuals_inverse[:, idx], outputs_inverse[:, idx], alpha=0.5)
        plt.plot([residuals_inverse[:, idx].min(), residuals_inverse[:, idx].max()],
                 [residuals_inverse[:, idx].min(), residuals_inverse[:, idx].max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Scatter Plot for {residual_var}')
        plt.tight_layout()
        scatter_plot_filename = f"{model_identifier}_scatter_{residual_var}.png"
        scatter_plot_path = os.path.join(model_results_dir, scatter_plot_filename)
        plt.savefig(scatter_plot_path)
        plt.close()

        # Residual Plot
        residuals = residuals_dict[residual_var]
        plt.figure(figsize=(6, 6))
        plt.scatter(outputs_inverse[:, idx], residuals, alpha=0.5)
        plt.hlines(y=0, xmin=outputs_inverse[:, idx].min(), xmax=outputs_inverse[:, idx].max(), colors='r', linestyles='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for {residual_var}')
        plt.tight_layout()
        residual_plot_filename = f"{model_identifier}_residual_{residual_var}.png"
        residual_plot_path = os.path.join(model_results_dir, residual_plot_filename)
        plt.savefig(residual_plot_path)
        plt.close()

        # Error Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title(f'Error Histogram for {residual_var}')
        plt.tight_layout()
        histogram_filename = f"{model_identifier}_histogram_{residual_var}.png"
        histogram_path = os.path.join(model_results_dir, histogram_filename)
        plt.savefig(histogram_path)
        plt.close()

    metrics_filename = f"{model_identifier}_metrics.txt"
    metrics_path = os.path.join(model_results_dir, metrics_filename)
    with open(metrics_path, 'w') as f:
        f.write("Test Results:\n")
        for residual_var, metric in metrics.items():
            f.write(f"Metrics for {residual_var}:\n")
            f.write(f"  Mean Squared Error (MSE): {metric['MSE']:.6f}\n")
            f.write(f"  Root Mean Squared Error (RMSE): {metric['RMSE']:.6f}\n")
            f.write(f"  Mean Absolute Error (MAE): {metric['MAE']:.6f}\n")
            f.write(f"  Mean Absolute Percentage Error (MAPE): {metric['MAPE']:.6f}%\n")
            f.write(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {metric['sMAPE']:.6f}%\n")
            f.write(f"  Median Absolute Percentage Error (MdAPE): {metric['MdAPE']:.6f}%\n")
            f.write(f"  R-squared (R²): {metric['R2']:.6f}\n\n")

        f.write("Mean Metrics over all residual variables:\n")
        f.write(f"  Mean Squared Error (MSE): {mean_metrics['MSE']:.6f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {mean_metrics['RMSE']:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mean_metrics['MAE']:.6f}\n")
        f.write(f"  Mean Absolute Percentage Error (MAPE): {mean_metrics['MAPE']:.6f}%\n")
        f.write(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {mean_metrics['sMAPE']:.6f}%\n")
        f.write(f"  Median Absolute Percentage Error (MdAPE): {mean_metrics['MdAPE']:.6f}%\n")
        f.write(f"  R-squared (R²): {mean_metrics['R2']:.6f}\n")

    print(f"Metrics saved to {metrics_path}")

    df_data = {}
    df_data['Sample Index'] = np.arange(len(residuals_inverse))

    for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]):
        df_data[f"{residual_var}_Actual"] = residuals_inverse[:, idx]
        df_data[f"{residual_var}_Predicted"] = outputs_inverse[:, idx]
        df_data[f"{residual_var}_Difference"] = residuals_dict[residual_var]

    df = pd.DataFrame(df_data)

    excel_filename = f"{model_identifier}_results.xlsx"
    excel_path = os.path.join(model_results_dir, excel_filename)

    df.to_excel(excel_path, index=False)
    print(f"Detailed results (Actual, Predicted, Residuals) saved to {excel_path}")

    # Identify indices for x, y, z values
    xyz_indices = [idx for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]) if residual_var in ['x_dif', 'y_dif', 'z_dif']]

    # Calculate Euclidean distance for each sample using only x, y, z values
    euclidean_distances = np.linalg.norm(residuals_inverse[:, xyz_indices] - outputs_inverse[:, xyz_indices], axis=1)

    # Calculate Euclidean distance for actual residuals
    actual_euclidean_distances = np.linalg.norm(residuals_inverse[:, xyz_indices], axis=1)
    
    # Calculate mean values
    mean_euclidean = np.mean(euclidean_distances)
    mean_actual_euclidean = np.mean(actual_euclidean_distances)

    # Create a line plot for Euclidean distances with mean values indicated in the legend
    plt.figure(figsize=(12, 6))
    plt.plot(actual_euclidean_distances,
             label=f'Actual Residuals (Euclidean Distance) - Mean: {mean_actual_euclidean:.2f}',
             color='green')
    plt.plot(euclidean_distances,
             label=f'Difference (Actual - Predicted) - Mean: {mean_euclidean:.2f}',
             color='blue')
    plt.title('Euclidean Distance Over Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    euclidean_plot_filename = f"{model_identifier}_euclidean_distance_plot.png"
    euclidean_plot_path = os.path.join(model_results_dir, euclidean_plot_filename)
    plt.savefig(euclidean_plot_path)
    plt.close()
    print(f"Euclidean distance plot saved to {euclidean_plot_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_TGCN.yaml",
                        help="Path to the YAML configuration file (default: %(default)s).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    test_model(config)
