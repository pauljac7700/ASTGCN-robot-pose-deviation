"""Evaluate a trained single-target ASTGCN on the held-out test split.

Writes per-axis metrics and a comparison plot into ``results/results_single/``.
"""

# test_ASTGCN_single.py

import argparse
import os
import torch
import numpy as np
import yaml
from model.ASTGCN_single import make_model
from lib.extract_number_from_filename import extract_number_from_filename
import joblib
from lib.evaluation_metrics import masked_mape, masked_mse, masked_mae, masked_r2_score, masked_smape, masked_mdape
import matplotlib.pyplot as plt
import pandas as pd

def test_model(config):
    # Load test data
    dataset_dimension = config['dataset_dimension']
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    single_residual_variable = config['single_residual_variable']
    print("Dataset Dimension:", dataset_dimension)
    print("Dataset Name:", dataset_name)
    print("Dataset Type:", dataset_type)
    print("Single Residual Variable:", single_residual_variable)
    model_name = config['model_name']['single']
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
    inputs_test = test_data['inputs']  # Shape: (num_samples, num_nodes, in_channels, len_input + 1)
    residuals_test = test_data['residuals']  # Shape: (num_samples, num_residuals)

    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs_test).float()
    residuals_tensor = torch.from_numpy(residuals_test).float()

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])

    # Initialize model
    model = make_model(
        DEVICE=DEVICE,
        nb_block=config['model']['nb_block'],
        in_channels=config['model'][f'in_channels_{dataset_dimension}'],
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
    model_save_dir = os.path.join('saved_models',dataset_dimension,dataset_name,dataset_type,model_name,single_residual_variable)
    model_identifier = single_residual_variable

    # Filter model files by both the prefix and the identifier
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith(f'{model_name}_best_{model_identifier}_{graph_nr}_{prep_data_incl_past_residuals}') and f.endswith('.pth')]

    if not model_files:
        raise FileNotFoundError(f"No saved model found in the specified directory for residual variable '{model_identifier}'.")
    else:
        model_files.sort()
        actual_model_name = model_files[-1]  # Load the most recent model
    model_path = os.path.join(model_save_dir, actual_model_name)

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")

    model.eval()

    # Make predictions
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(DEVICE)
        residuals_tensor = residuals_tensor.to(DEVICE)

        outputs = model(inputs_tensor)  # Shape: (num_samples, N, num_for_predict)
        # Extract outputs for the residual node (node 7)
        outputs_residual_node = outputs[:, num_joints+1, :].squeeze(-1)  # Shape: (num_samples,)

    # Inverse transform the predictions and targets
    scalers = joblib.load(config['scalers_file'])
    residual_scaler = scalers['residual_scalers']

    outputs_np = outputs_residual_node.cpu().numpy().reshape(-1, 1)
    outputs_inverse = residual_scaler.inverse_transform(outputs_np).reshape(-1)
    residuals_np = residuals_tensor.cpu().numpy().reshape(-1, 1)
    residuals_inverse = residual_scaler.inverse_transform(residuals_np).reshape(-1)

    # Compute residuals
    residuals = residuals_inverse - outputs_inverse

    # Compute metrics
    mse = masked_mse(outputs_inverse, residuals_inverse, null_val=0)
    rmse = np.sqrt(mse)
    mae = masked_mae(outputs_inverse, residuals_inverse, null_val=0)
    mape = masked_mape(outputs_inverse, residuals_inverse, null_val=0)
    smape = masked_smape(outputs_inverse, residuals_inverse, null_val=0)
    mdape = masked_mdape(outputs_inverse, residuals_inverse, null_val=0)
    r2_score = masked_r2_score(outputs_inverse, residuals_inverse, null_val=0)
    

    # Print metrics
    print("Test Results:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")
    print(f"Symmetric Mean Absolute Percentage Error (sMAPE): {smape:.6f}%")
    print(f"Median Absolute Percentage Error (MdAPE): {mdape:.6f}%")
    print(f"R-squared (R²): {r2_score:.6f}")

    # Save results and plots
    results_dir = os.path.join('results',dataset_dimension,dataset_name,dataset_type,model_name,single_residual_variable)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Generate a filename prefix based on the model used
    model_identifier = os.path.splitext(actual_model_name)[0]  # Remove '.pth' extension

    # Create a subdirectory named after the model identifier
    model_results_dir = os.path.join(results_dir, model_identifier)
    if not os.path.isdir(model_results_dir):
        os.makedirs(model_results_dir)

    # Time Series Plot
    plt.figure(figsize=(12, 6))
    plt.plot(residuals_inverse, label='Actual')
    plt.plot(outputs_inverse, label='Predicted')
    plt.legend()
    plt.title(f'Actual vs. Predicted {single_residual_variable}')
    plt.xlabel('Sample Index')
    plt.ylabel(single_residual_variable)
    plt.tight_layout()
    plot_path = os.path.join(model_results_dir, f"{model_identifier}_timeseries.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Time series plot saved to {plot_path}")

    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(residuals_inverse, outputs_inverse, alpha=0.5)
    plt.plot([residuals_inverse.min(), residuals_inverse.max()],
             [residuals_inverse.min(), residuals_inverse.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scatter Plot for {single_residual_variable}')
    plt.tight_layout()
    scatter_plot_path = os.path.join(model_results_dir, f"{model_identifier}_scatter.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Scatter plot saved to {scatter_plot_path}")

    # Residual Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(outputs_inverse, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=outputs_inverse.min(), xmax=outputs_inverse.max(), colors='r', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {single_residual_variable}')
    plt.tight_layout()
    residual_plot_path = os.path.join(model_results_dir, f"{model_identifier}_residual.png")
    plt.savefig(residual_plot_path)
    plt.close()
    print(f"Residual plot saved to {residual_plot_path}")

    # Error Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'Error Histogram for {single_residual_variable}')
    plt.tight_layout()
    histogram_path = os.path.join(model_results_dir, f"{model_identifier}_histogram.png")
    plt.savefig(histogram_path)
    plt.close()
    print(f"Error histogram saved to {histogram_path}")

    # Save metrics to a text file
    metrics_path = os.path.join(model_results_dir, f"{model_identifier}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%\n")
        f.write(f"Symmetric Mean Absolute Percentage Error (sMAPE): {smape:.6f}%\n")
        f.write(f"Median Absolute Percentage Error (MdAPE): {mdape:.6f}%\n")
        f.write(f"R-squared (R²): {r2_score:.6f}\n")
    print(f"Metrics saved to {metrics_path}")

    # After computing the metrics and saving figures, add the following code to create and save the Excel file:
    df = pd.DataFrame({
        'Residual Variable': single_residual_variable,
        'Actual': residuals_inverse,
        'Predicted': outputs_inverse,
        'Residual': residuals
    })
    excel_path = os.path.join(model_results_dir, f"{model_identifier}_predictions.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Predictions and actual values saved to {excel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_ASTGCN.yaml",
                        help="Path to the YAML configuration file (default: %(default)s).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    test_model(config)
