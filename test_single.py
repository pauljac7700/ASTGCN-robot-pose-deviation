# test_single.py

import os
import torch
import numpy as np
import yaml
from model.ASTGCN_single import make_model
import joblib
from lib.evaluation_metrics import masked_mape, masked_mse, masked_mae, masked_r2_score

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
    # Keep adj_mx as a NumPy array for compatibility with utils.py

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
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith('astgcn_single_best_') and f.endswith('.pth')]
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

        outputs = model(inputs_tensor)  # Shape: (num_samples, N, num_for_predict)
        # Extract outputs for the error node (Node 7)
        outputs_x_dif = outputs[:, 7, :].squeeze(-1)  # Shape: (num_samples,)

    # Inverse transform the predictions and targets
    scalers = joblib.load(config['scalers_file'])
    target_scaler = scalers['target_scaler']

    outputs_np = outputs_x_dif.cpu().numpy().reshape(-1, 1)
    outputs_inverse = target_scaler.inverse_transform(outputs_np)

    targets_np = targets_tensor.cpu().numpy().reshape(-1, 1)
    targets_inverse = target_scaler.inverse_transform(targets_np)

    # Compute metrics
    mse = masked_mse(outputs_inverse, targets_inverse, null_val=0)
    rmse = np.sqrt(mse)
    mape = masked_mape(outputs_inverse, targets_inverse, null_val=0)
    mae = masked_mae(outputs_inverse, targets_inverse, null_val=0)
    r2_score = masked_r2_score(outputs_inverse, targets_inverse, null_val=0)

    # Print metrics
    print("Test Results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"R-squared (R²): {r2_score:.4f}")

    # Save results and plots
    import matplotlib.pyplot as plt

    # Create results directory if it doesn't exist
    results_dir = config['logging']['results_single_dir']
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"The directory '{results_dir}' does not exist.")

    # Generate a filename prefix based on the model used
    model_identifier = os.path.splitext(model_name)[0]  # Remove '.pth' extension

    # Include 'single' in the filenames
    identifier = 'single'

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(targets_inverse, label='Actual')
    plt.plot(outputs_inverse, label='Predicted')
    plt.legend()
    plt.title('Actual vs. Predicted x_dif')
    plt.xlabel('Sample Index')
    plt.ylabel('x_dif')
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
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%\n")
        f.write(f"R-squared (R²): {r2_score:.4f}\n")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # Load configuration
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)

    test_model(config)
