"""Evaluate the ConvLSTM baseline on the held-out test split.

Reported alongside the ASTGCN results to show what a purely convolutional
recurrent model achieves on the same data, without the graph structure.
"""

# test_Conv_LSTM.py

import argparse
import os
import torch
import numpy as np
import yaml
from model.ConvLSTM import ConvLSTMWithFC  # Ensure this imports your updated ConvLSTM module
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
    """
    Evaluates the ConvLSTM model on the test dataset and generates performance metrics and visualizations.

    Parameters:
    - config (dict): Configuration parameters loaded from the YAML file.
    """
    dataset_dimension = config['dataset_dimension']
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    model_name = config['model_name']['multi']
    num_joints = config['num_joints'] 
    if config['prep_data_incl_past_residuals']: 
        prep_data_incl_past_residuals = 'wr'
        print("Including past residuals in input features.")
    else:   
        prep_data_incl_past_residuals = 'nr'
        print("Excluding past residuals from input features.")

        # Load test data
    test_data = np.load(f'data/convlstm_test_{prep_data_incl_past_residuals}.npz')
    inputs_test = test_data['inputs']  # Shape: (num_samples, num_nodes, in_channels, len_input + 1)
    residuals_test = test_data['residuals']  # Shape: (num_samples, num_residuals)

    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs_test).float()   # Shape: (N, T, C, H, W)
    residuals_tensor = torch.from_numpy(residuals_test).float() # Shape: (N, num_residuals)
    
    # Device configuration
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ConvLSTMWithFC(
        input_dim=config['model'][dataset_dimension]['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        kernel_size=config['model']['kernel_size'],
        num_layers=config['model']['num_layers'],
        output_dim=config['model'][dataset_dimension]['output_dim'],
        bias=config['model']['bias'],
        return_all_layers=config['model']['return_all_layers']
    )
    model.to(device)
    print(model)
    
    # Load the best saved model
    model_save_dir = os.path.join('saved_models',dataset_dimension,dataset_name,dataset_type,model_name)
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith(f'convlstm_best_{prep_data_incl_past_residuals}') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No saved ConvLSTM model found in the specified directory.")
    else:
        model_files.sort()
        trained_model_name = model_files[-1]  # Select the latest saved model
    trained_model_path = os.path.join(model_save_dir, trained_model_name)
    checkpoint = torch.load(trained_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {trained_model_path}")
    
    model.eval()
    
    # Make predictions
    print("Making predictions on the test set...")
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(device)
        residuals_tensor = residuals_tensor.to(device)
        outputs = model(inputs_tensor)  # Shape: (N, num_residuals)
    
    # Inverse transform the predictions and targets
    print("Inverse transforming the predictions and residuals...")
    scalers = joblib.load(config['scalers_file'])
    residual_scalers = scalers['residual_scalers']
    
    outputs_np = outputs.cpu().numpy()       # Shape: (N, num_residuals)
    residuals_np = residuals_tensor.cpu().numpy()# Shape: (N, num_residuals)
    
    outputs_inverse = np.zeros_like(outputs_np)
    residuals_inverse = np.zeros_like(residuals_np)
    
    residual_variables = config.get('residual_variables', {}).get(config['dataset_dimension'], [])
    
    for idx, residual_var in enumerate(residual_variables):
        scaler = residual_scalers[residual_var]
        outputs_inverse[:, idx] = scaler.inverse_transform(outputs_np[:, idx].reshape(-1, 1)).reshape(-1)
        residuals_inverse[:, idx] = scaler.inverse_transform(residuals_np[:, idx].reshape(-1, 1)).reshape(-1)
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    metrics = {}
    mse_list = []
    rmse_list = []
    mae_list = []
    mape_list = []
    smape_list = []
    mdape_list = []
    r2_list = []
    residuals_dict = {}
    
    for idx, residual_var in enumerate(residual_variables):
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
    
    # Display metrics
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
    
    # Save results
    # Create results directory if it doesn't exist
    results_dir = os.path.join('results',dataset_dimension,dataset_name,dataset_type,model_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    model_identifier = os.path.splitext(trained_model_name)[0]
    model_results_dir = os.path.join(results_dir, model_identifier)
    if not os.path.isdir(model_results_dir):
        os.makedirs(model_results_dir)
    
    num_residuals = len(residual_variables)
    
    # Plot Actual vs Predicted Time Series for each target variable
    print("Generating time series plots...")
    plt.figure(figsize=(12, 6 * num_residuals))
    for idx, target_var in enumerate(residual_variables):
        plt.subplot(num_residuals, 1, idx + 1)
        plt.plot(residuals_inverse[:, idx], label='Actual')
        plt.plot(outputs_inverse[:, idx], label='Predicted')
        plt.legend()
        plt.title(f'Actual vs. Predicted {target_var}')
        plt.xlabel('Sample Index')
        plt.ylabel(target_var)
    plt.tight_layout()
    plot_filename = f"{model_identifier}_timeseries.png"
    plot_path = os.path.join(model_results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Time series plots saved to {plot_path}")
    
    # Plot Scatter Plots for each target variable
    print("Generating scatter plots...")
    for idx, residual_var in enumerate(residual_variables):
        plt.figure(figsize=(6, 6))
        plt.scatter(residuals_inverse[:, idx], outputs_inverse[:, idx], alpha=0.5)
        plt.plot([residuals_inverse[:, idx].min(), residuals_inverse[:, idx].max()],
                 [residuals_inverse[:, idx].min(), residuals_inverse[:, idx].max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Scatter Plot for {target_var}')
        plt.tight_layout()
        scatter_plot_filename = f"{model_identifier}_scatter_{residual_var}.png"
        scatter_plot_path = os.path.join(model_results_dir, scatter_plot_filename)
        plt.savefig(scatter_plot_path)
        plt.close()
    
    # Plot Residual Plots for each target variable
    print("Generating residual plots...")
    for idx, residual_var in enumerate(residual_variables):
        plt.figure(figsize=(6, 6))
        plt.scatter(outputs_inverse[:, idx], residuals_dict[residual_var], alpha=0.5)
        plt.hlines(y=0, xmin=outputs_inverse[:, idx].min(), xmax=outputs_inverse[:, idx].max(), colors='r', linestyles='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for {residual_var}')
        plt.tight_layout()
        residual_plot_filename = f"{model_identifier}_residual_{residual_var}.png"
        residual_plot_path = os.path.join(model_results_dir, residual_plot_filename)
        plt.savefig(residual_plot_path)
        plt.close()
    
    # Plot Error Histograms for each target variable
    print("Generating error histograms...")
    for idx, residual_var in enumerate(residual_variables):
        plt.figure(figsize=(6, 4))
        plt.hist(residuals_dict[residual_var], bins=50, alpha=0.7)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title(f'Error Histogram for {residual_var}')
        plt.tight_layout()
        histogram_filename = f"{model_identifier}_histogram_{residual_var}.png"
        histogram_path = os.path.join(model_results_dir, histogram_filename)
        plt.savefig(histogram_path)
        plt.close()
    
    # Save metrics to a text file
    metrics_filename = f"{model_identifier}_metrics.txt"
    metrics_path = os.path.join(model_results_dir, metrics_filename)
    print(f"Saving metrics to {metrics_path}...")
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
    
    # Save detailed results to Excel
    print("Saving detailed results to Excel...")
    df_data = {}
    df_data = {'Sample Index': np.arange(len(residuals_inverse))}

    for idx, residual_var in enumerate(residual_variables):  # Iterate over variable names
        df_data[f"{residual_var}_Actual"] = residuals_inverse[:, idx]
        df_data[f"{residual_var}_Predicted"] = outputs_inverse[:, idx]
        df_data[f"{residual_var}_Difference"] = residuals_dict[residual_var]  # Correct dictionary access
    
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
    parser.add_argument("--config", default="config_Conv_LSTM.yaml",
                        help="Path to the YAML configuration file (default: %(default)s).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    test_model(config)
