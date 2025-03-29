import os
import torch
import numpy as np
import yaml
from model.ASTGCN_no_temporal import make_model_no_temporal
from model.ASTGCN_no_spatial import make_model_no_spatial
from model.ASTGCN_no_attention import make_model_no_attention
from lib.extract_number_from_filename import extract_number_from_filename
from lib.get_adjacency_matrix_size import get_adjacency_matrix_size
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
    model_name = config['model_name'].get(config['ablation_model'], None)
    num_joints = config['num_joints']
    if config['prep_data_incl_past_residuals']:
        prep_data_incl_past_residuals = 'wr'
        print("Including past residuals in input features.")
    else:
        prep_data_incl_past_residuals = 'nr'
        print("Excluding past residuals from input features.")

    graph_nr = extract_number_from_filename(config['adjacency_matrix_file'])
    print(f"Graph number: {graph_nr}")

    # Get the number of nodes from the adjacency matrix
    num_nodes = get_adjacency_matrix_size(config)
    print(f"Number of nodes: {num_nodes}")

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

    # Number of residuals
    num_residuals = len(config['residual_variables'][dataset_dimension])

    # Set the in_channels and residual_dim depending on the underlying graph
    if graph_nr in [1, 3, 6, 7]:
        in_channels = config['model'][f'in_channels_{dataset_dimension}']
        residual_dim = num_residuals
    elif graph_nr == 2:
        in_channels = 1
        residual_dim = 1
    elif graph_nr in [4, 5]:
        in_channels = 3
        residual_dim = 3 

    # Initialize model based on ablation type
    if config['ablation_model'] == 'multi_no_temporal':
        model = make_model_no_temporal(
            DEVICE=DEVICE,
            nb_block=config['model']['nb_block'],
            in_channels=in_channels,
            K=config['model']['K'],
            nb_chev_filter=config['model']['nb_chev_filter'],
            nb_time_filter=config['model']['nb_time_filter'],
            time_strides=config['model']['time_strides'],
            adj_mx=adj_mx,
            num_for_predict=config['model']['num_for_predict'],
            len_input=config['model']['len_input'] + 1,
            num_of_vertices=num_nodes,
            residual_dim=residual_dim
        )
    elif config['ablation_model'] == 'multi_no_spatial':
        model = make_model_no_spatial(
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
            num_of_vertices=num_nodes,
            residual_dim=residual_dim
        )
    elif config['ablation_model'] == 'multi_no_attention':
        model = make_model_no_attention(
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
            num_of_vertices=num_nodes,
            residual_dim=residual_dim
        )
        model.to(DEVICE)

    # Load the best saved model
    model_save_dir = os.path.join('saved_models', dataset_dimension, dataset_name, dataset_type, model_name)
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith(f'{model_name}_best_{graph_nr}_{prep_data_incl_past_residuals}') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No saved model found in the specified directory.")
    else:
        model_files.sort()
        actual_model_name = model_files[-1]
    model_path = os.path.join(model_save_dir, actual_model_name)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")

    model.eval()

    # Make predictions
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(DEVICE)
        residuals_tensor = residuals_tensor.to(DEVICE)

        outputs = model(inputs_tensor)  # Expected shape:
                                       # For graphs 1,3,6,7: (B, num_nodes, num_for_predict, residual_dim)
                                       # For graph 2: (B, num_nodes, num_for_predict, 1)
                                       # For graphs 4,5: (B, num_nodes, num_for_predict, residual_dim) with residual_dim=3
        if graph_nr in [1, 3, 6, 7]:
            # For these graphs, assume the residual node is at index num_joints+1.
            outputs_residual_node = outputs[:, num_joints+1, :, :]
            outputs_residual_node = outputs_residual_node.squeeze(1)  # Shape: (B, residual_dim)
        elif graph_nr == 2:
            # For graph 2, extract the multiple residual nodes.
            num_residual_nodes = (num_nodes - num_joints) // 2
            outputs_residual_node = outputs[:, num_joints:num_joints+num_residual_nodes, :, :]
            if config['model']['num_for_predict'] == 1:
                outputs_residual_node = outputs_residual_node.squeeze(2).squeeze(-1)  # Shape: (B, num_residual_nodes)
        elif graph_nr in [4, 5]:
            # For graphs 4 and 5, residuals are split into two nodes (position and orientation)
            # which are at indices num_joints+2 and num_joints+3.
            outputs_residual_node = outputs[:, num_joints+2:num_joints+4, :, :]
            if config['model']['num_for_predict'] == 1:
                outputs_residual_node = outputs_residual_node.squeeze(2).squeeze(-1)  # Shape: (B, 2, residual_dim)
                # Flatten the two residual nodes into one vector per sample:
                outputs_residual_node = outputs_residual_node.reshape(outputs_residual_node.shape[0], -1)
                # For residual_dim=3, the output shape becomes (B, 6)

    # Inverse transform the predictions and targets
    scalers = joblib.load(config['scalers_file'])
    residual_scalers = scalers['residual_scalers']

    outputs_np = outputs_residual_node.cpu().numpy()
    residuals_np = residuals_tensor.cpu().numpy()

    outputs_inverse = np.zeros_like(outputs_np)
    residuals_inverse = np.zeros_like(residuals_np)

    for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]):
        scaler = residual_scalers[residual_var]
        outputs_inverse[:, idx] = scaler.inverse_transform(outputs_np[:, idx].reshape(-1, 1)).reshape(-1)
        residuals_inverse[:, idx] = scaler.inverse_transform(residuals_np[:, idx].reshape(-1, 1)).reshape(-1)

    # Initialize metric lists
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
        
        residuals_diff = residuals_inverse[:, idx] - outputs_inverse[:, idx]
        residuals_dict[residual_var] = residuals_diff

    # Compute mean metrics over all residual variables
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
    results_dir = os.path.join('results', dataset_dimension, dataset_name, dataset_type, model_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    model_identifier = os.path.splitext(actual_model_name)[0]

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

    for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]):
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

    # Create Excel File with Actual, Predicted, Difference
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
    print(f"Detailed results (Actual, Predicted, Difference) saved to {excel_path}")

    # Identify indices for x, y, z values
    xyz_indices = [idx for idx, residual_var in enumerate(config['residual_variables'][dataset_dimension]) if residual_var in ['x_dif', 'y_dif', 'z_dif']]
    euclidean_distances = np.linalg.norm(residuals_inverse[:, xyz_indices] - outputs_inverse[:, xyz_indices], axis=1)
    actual_euclidean_distances = np.linalg.norm(residuals_inverse[:, xyz_indices], axis=1)
    mean_euclidean = np.mean(euclidean_distances)
    mean_actual_euclidean = np.mean(actual_euclidean_distances)

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
    euclidean_plot_filename = f"{model_identifier}_euclidean_distance_plot.png"
    euclidean_plot_path = os.path.join(model_results_dir, euclidean_plot_filename)
    plt.savefig(euclidean_plot_path)
    plt.close()
    print(f"Euclidean distance plot saved to {euclidean_plot_path}")
    
if __name__ == "__main__":
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)
    test_model(config)
