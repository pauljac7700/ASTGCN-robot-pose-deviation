# test.py

import os
import torch
import numpy as np
import yaml
from model.adjusted_ASTGCN import make_model
import joblib
from lib.evaluation_metrics import masked_mape, masked_mse

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
        len_input=config['model']['len_input'],
        num_of_vertices=config['model']['num_of_vertices'],
        target_dim=config['model']['target_dim']
    )
    model.to(DEVICE)

    # Load the best saved model
    model_save_dir = config['logging']['model_save_dir']
    model_files = [f for f in os.listdir(model_save_dir) if f.startswith('astgcn_best_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No saved model found in the specified directory.")
    else:
        # Assuming the latest model is the best one
        model_files.sort()
        model_name = model_files[-1]
    model_path = os.path.join(model_save_dir, model_name)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # Make predictions
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(DEVICE)
        targets_tensor = targets_tensor.to(DEVICE)

        outputs = model(inputs_tensor)  # Shape: (num_samples, N, num_for_predict, target_dim)
        outputs_end_effector = outputs[:, 6, :, :]  # Shape: (num_samples, num_for_predict, target_dim)

    # Inverse transform the predictions and targets
    scalers = joblib.load(config['scalers_file'])
    target_scaler = scalers['target_scaler']

    outputs_np = outputs_end_effector.cpu().numpy().reshape(-1, config['model']['target_dim'])
    outputs_inverse = target_scaler.inverse_transform(outputs_np)

    targets_np = targets_tensor.cpu().numpy().reshape(-1, config['model']['target_dim'])
    targets_inverse = target_scaler.inverse_transform(targets_np)

    # Compute metrics
    mse = masked_mse(outputs_inverse, targets_inverse, null_val=0)
    mape = masked_mape(outputs_inverse, targets_inverse, null_val=0)

    print("Test Results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

    # Detailed summary
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(targets_inverse, label='Actual')
    plt.plot(outputs_inverse, label='Predicted')
    plt.legend()
    plt.title('Actual vs. Predicted x_dif')
    plt.xlabel('Sample Index')
    plt.ylabel('x_dif')
    plt.show()

if __name__ == "__main__":
    # Load configuration
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)

    test_model(config)
