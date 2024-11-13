# test_multi.py

import os
import torch
import numpy as np
import yaml
from model.ASTGCN_multi import make_model
import joblib
from lib.evaluation_metrics import masked_mape, masked_mse, masked_mae, masked_r2_score

def test_model(config):
    # Load test data
    test_data = np.load(config['test_data_file'])
    inputs_test = test_data['inputs']  # Shape: (num_samples, num_nodes, in_channels, len_input + 1)
    targets_test = test_data['targets']  # Shape: (num_samples, num_targets)

    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs_test).float()
    targets_tensor = torch.from_numpy(targets_test).float()

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])
    # Do NOT convert adj_mx to a PyTorch tensor here; keep it as a NumPy array

    # Number of targets
    num_targets = len(config['target_variables'])

    # Initialize model
    model = make_model(
        DEVICE=DEVICE,
        nb_block=config['model']['nb_block'],
        in_channels=config['model']['in_channels'],
        K=config['model']['K'],
        nb_chev_filter=config['model']['nb_chev_filter'],
        nb_time_filter=config['model']['nb_time_filter'],
        time_strides=config['model']['time_strides'],
        adj_mx=adj_mx,  # Pass the adjacency matrix as a NumPy array
        num_for_predict=config['model']['num_for_predict'],
        len_input=config['model']['len_input'] + 1,  # Adjusted for extended input sequence
        num_of_vertices=config['model']['num_of_vertices'],
        target_dim=num_targets  # Pass the number of target variables
    )
    model.to(DEVICE)

    # Load the best saved model
    model_save_dir = config['logging']['model_save_dir']
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith('astgcn_multi_best_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No saved model found in the specified directory.")
    else:
        # Assuming the latest model is the best one
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

        outputs = model(inputs_tensor)  # Shape: (num_samples, num_nodes, num_for_predict, target_dim)

        # Extract outputs for Node 7
        outputs_node7 = outputs[:, 7, :, :].squeeze(1)  # Shape: (num_samples, target_dim)

    # Inverse transform the predictions and targets
    scalers = joblib.load(config['scalers_file'])
    target_scalers = scalers['target_scalers']

    outputs_np = outputs_node7.cpu().numpy()
    targets_np = targets_tensor.cpu().numpy()

    outputs_inverse = np.zeros_like(outputs_np)
    targets_inverse = np.zeros_like(targets_np)

    for idx, target_var in enumerate(config['target_variables']):
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
    for idx, target_var in enumerate(config['target_variables']):
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
    import matplotlib.pyplot as plt

    # Create results directory if it doesn't exist
    results_dir = config['logging']['results_multi_dir']
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"The directory '{results_dir}' does not exist.")

    # Generate a filename prefix based on the model used
    model_identifier = os.path.splitext(model_name)[0]  # Remove '.pth' extension

    # Include 'multi' in the filenames
    identifier = 'multi'

    num_targets = len(config['target_variables'])
    plt.figure(figsize=(12, 6 * num_targets))
    for idx, target_var in enumerate(config['target_variables']):
        plt.subplot(num_targets, 1, idx + 1)
        plt.plot(targets_inverse[:, idx], label='Actual')
        plt.plot(outputs_inverse[:, idx], label='Predicted')
        plt.legend()
        plt.title(f'Actual vs. Predicted {target_var}')
        plt.xlabel('Sample Index')
        plt.ylabel(target_var)
    plt.tight_layout()

    # Save the figure
    plot_filename = f"{model_identifier}_{identifier}_results.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Results plot saved to {plot_path}")

    # Save metrics to a text file
    metrics_filename = f"{model_identifier}_{identifier}_metrics.txt"
    metrics_path = os.path.join(results_dir, metrics_filename)
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
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)

    # No need to add target variables here since they are already in the config
    test_model(config)
