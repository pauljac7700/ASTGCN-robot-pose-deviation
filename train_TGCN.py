import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import yaml
import joblib
from model.TGCN import TGCNWithGlobalOutput
from lib.extract_number_from_filename import extract_number_from_filename
from lib.compare_yaml_configs import compare_yaml_configs
import joblib


def train_model(config):

    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Load datasets and infos
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

    train_data = np.load(f'data/train_data_{graph_nr}_{prep_data_incl_past_residuals}.npz')
    val_data = np.load(f'data/val_data_{graph_nr}_{prep_data_incl_past_residuals}.npz')
    inputs_train = train_data['inputs']  # Shape: (num_samples, num_nodes, in_channels, len_input + 1)
    residuals_train = train_data['residuals']  # Shape: (num_samples, num_residuals)
    inputs_val = val_data['inputs']
    residuals_val = val_data['residuals']

    # Load scalers (if needed)
    scalers = joblib.load(config['scalers_file'])
    # No need to apply scalers here since data is already scaled

    # Process inputs to be compatible with TGCN model
    # TGCN now expects inputs of shape (batch_size, seq_len, num_nodes, in_channels)
    inputs_train = inputs_train.transpose(0, 3, 1, 2)  # Shape: (num_samples, seq_len, num_nodes, in_channels)
    inputs_val = inputs_val.transpose(0, 3, 1, 2)

    # Convert to PyTorch tensors
    inputs_train_tensor = torch.from_numpy(inputs_train).float()
    residuals_train_tensor = torch.from_numpy(residuals_train).float()
    inputs_val_tensor = torch.from_numpy(inputs_val).float()
    residuals_val_tensor = torch.from_numpy(residuals_val).float()

    # Create datasets and data loaders
    train_dataset = TensorDataset(inputs_train_tensor, residuals_train_tensor)
    val_dataset = TensorDataset(inputs_val_tensor, residuals_val_tensor)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Number of residuals
    num_residuals = len(config['residual_variables'][dataset_dimension])

    # Set the in_channels and residual_dim depending on the underlying graph
    if graph_nr in [1, 3, 6, 7]:
        in_channels = config['model'][f'in_channels_{dataset_dimension}']
    elif graph_nr == 2:
        in_channels = 1

    # Initialize model
    hidden_dim = config['model']['hidden_dim']
    tgcn_model = TGCNWithGlobalOutput(adj=adj_mx, in_channels=in_channels, hidden_dim=hidden_dim, num_residuals=num_residuals)
    tgcn_model.to(DEVICE)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(tgcn_model.parameters(),
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])

    # TensorBoard setup
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', dataset_dimension, dataset_name, dataset_type, model_name,
                           f'{model_name}_{current_time}_{dataset_dimension}_{dataset_name}_{dataset_type}_{graph_nr}_{prep_data_incl_past_residuals}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model_save_dir = os.path.join('saved_models', dataset_dimension, dataset_name, dataset_type, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # Log hyperparameters
    hparams = {**config['model'], **config['training']}

    # Early stopping parameters
    patience = config['training']['patience']
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        tgcn_model.train()
        train_loss = 0.0
        for batch_idx, (inputs_batch, residuals_batch) in enumerate(train_loader):
            inputs_batch = inputs_batch.to(DEVICE)  # Shape: (batch_size, seq_len, num_nodes, in_channels)
            residuals_batch = residuals_batch.to(DEVICE)  # Shape: (batch_size, num_targets)

            optimizer.zero_grad()

            # Forward pass
            outputs = tgcn_model(inputs_batch)  # Outputs shape: (batch_size, num_targets)

            # Compute loss
            loss = criterion(outputs, residuals_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss += loss.item() * inputs_batch.size(0)

        # Calculate average training loss for the epoch
        train_loss /= len(train_loader.dataset)

        # Validation loop
        tgcn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_batch, residuals_batch in val_loader:
                inputs_batch = inputs_batch.to(DEVICE)
                residuals_batch = residuals_batch.to(DEVICE)

                outputs = tgcn_model(inputs_batch)

                loss = criterion(outputs, residuals_batch)
                val_loss += loss.item() * inputs_batch.size(0)
        val_loss /= len(val_loader.dataset)

        # Log average losses to TensorBoard
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save the best model
            saved_model_name = f"{model_name}_best_{graph_nr}_{prep_data_incl_past_residuals}_{current_time}.pth"
            save_path = os.path.join(model_save_dir, saved_model_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': tgcn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'hyperparameters': hparams
            }, save_path)

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f} - Saving Best Model")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f} - No Improvement")

            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    # Finalize TensorBoard logging
    metrics = {'hparam/val_loss': best_val_loss}
    writer.add_hparams(hparams, metrics)
    writer.close()


if __name__ == "__main__":
    # Load configuration
    with open('config_TGCN.yaml') as f:
        config = yaml.safe_load(f)
    with open('config_ASTGCN.yaml', 'r') as f:
        config_ASTGCN = yaml.safe_load(f)
    compare_yaml_configs(config, config_ASTGCN)

    train_model(config)
