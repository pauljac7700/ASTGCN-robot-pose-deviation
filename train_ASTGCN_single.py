"""Train the single-target ASTGCN variant.

Carries the residual on a single node instead of spreading it across several, which
is the configuration used for the 3D UR5 experiments.
"""

# train_ASTGCN_single.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import yaml
from model.ASTGCN_single import make_model
from lib.extract_number_from_filename import extract_number_from_filename
import joblib

def train_model(config):

    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Load datasets and infos
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

    train_data = np.load(f'data/train_data_{graph_nr}_{prep_data_incl_past_residuals}.npz')
    val_data = np.load(f'data/val_data_{graph_nr}_{prep_data_incl_past_residuals}.npz')
    inputs_train = train_data['inputs']  # Shape: (num_samples, num_nodes, in_channels, len_input + 1)
    residuals_train = train_data['residuals']  # Shape: (num_samples, num_residuals)
    inputs_val = val_data['inputs']
    residuals_val = val_data['residuals']

    # Load scalers (if needed)
    scalers = joblib.load(config['scalers_file'])
    # No need to apply scalers here since data is already scaled

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
        len_input=config['model']['len_input'] + 1,  # Adjusted for extended input sequence
        num_of_vertices=config['model']['num_of_vertices']
    )
    model.to(DEVICE)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    # TensorBoard setup
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs',dataset_dimension,dataset_name,dataset_type,model_name,single_residual_variable, f'{model_name}_{current_time}_{dataset_dimension}_{dataset_name}_{dataset_type}_{graph_nr}_{prep_data_incl_past_residuals}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model_save_dir = os.path.join('saved_models',dataset_dimension,dataset_name,dataset_type,model_name,single_residual_variable)
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
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs_batch, residuals_batch) in enumerate(train_loader):
            inputs_batch = inputs_batch.to(DEVICE)
            residuals_batch = residuals_batch.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs_batch)  # Shape: (batch_size, N, num_for_predict)

            # Extract outputs for Residual Node
            outputs_residual_node = outputs[:, num_joints+1, :].squeeze(-1)  # Shape: (batch_size,)

            # Compute loss
            loss = criterion(outputs_residual_node, residuals_batch.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss += loss.item() * inputs_batch.size(0)
            
        # Calculate average training loss for the epoch
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_batch, residuals_batch in val_loader:
                inputs_batch = inputs_batch.to(DEVICE)
                residuals_batch = residuals_batch.to(DEVICE)

                outputs = model(inputs_batch)  # Shape: (batch_size, N, num_for_predict)
                outputs_residual_node = outputs[:, num_joints+1, :].squeeze(-1)  # Shape: (batch_size,)

                # Compute loss
                loss = criterion(outputs_residual_node, residuals_batch.view(-1))
                val_loss += loss.item() * inputs_batch.size(0)
        val_loss /= len(val_loader.dataset)

        # Log average losses to TensorBoard
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save the best model with the residual variable in the name
            saved_model_name = f"{model_name}_best_{single_residual_variable}_{graph_nr}_{prep_data_incl_past_residuals}_{current_time}.pth"
            save_path = os.path.join(model_save_dir, saved_model_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
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
    metrics = {'hparam/val_loss': val_loss}
    writer.add_hparams(hparams, metrics)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_ASTGCN.yaml",
                        help="Path to the YAML configuration file (default: %(default)s).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_model(config)
