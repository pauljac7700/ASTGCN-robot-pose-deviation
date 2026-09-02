"""Train the multi-target ASTGCN that predicts residual robot pose error.

This is the main model of the paper and the second stage of the hybrid method: it
learns the non-geometric error that remains after MDH geometric calibration, over a
graph whose structure follows the robot's serial kinematic chain. Logs to
TensorBoard and keeps the checkpoint with the best validation loss under early
stopping.
"""

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
from model.ASTGCN_multi import make_model
from lib.extract_number_from_filename import extract_number_from_filename
from lib.get_adjacency_matrix_size import get_adjacency_matrix_size
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
        prep_data_incl_residuals = 'nr'
        print("Excluding past residuals from input features.")
        prep_data_incl_past_residuals = 'nr'  # For consistency

    graph_nr = extract_number_from_filename(config['adjacency_matrix_file'])
    print(f"Graph number: {graph_nr}")

    # Get the number of nodes from the adjacency matrix
    num_nodes = get_adjacency_matrix_size(config)
    print(f"Number of nodes: {num_nodes}")

    train_data = np.load(f'data/train_data_{graph_nr}_{prep_data_incl_past_residuals}.npz')
    val_data = np.load(f'data/val_data_{graph_nr}_{prep_data_incl_past_residuals}.npz')
    inputs_train = train_data['inputs']  # Shape: (num_samples, num_nodes, in_channels, len_input + 1)
    residuals_train = train_data['residuals']  # Shape: (num_samples, num_residuals)
    inputs_val = val_data['inputs']
    residuals_val = val_data['residuals']

    # Load scalers (if needed)
    scalers = joblib.load(config['scalers_file'])
    # Data is already scaled, so no transformation is applied here

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

    # Load adjacency matrix (as NumPy array)
    adj_mx = np.load(config['adjacency_matrix_file'])

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Number of residuals from config (target variables)
    num_residuals = len(config['residual_variables'][dataset_dimension])

    # Set in_channels and residual_dim based on the graph structure
    if graph_nr in [1, 3, 6, 7]:
        in_channels = config['model'][f'in_channels_{dataset_dimension}']
        residual_dim = num_residuals
    elif graph_nr == 2:
        in_channels = 1
        residual_dim = 1
    elif graph_nr in [4, 5]:
        in_channels = 3
        residual_dim = 3  # Each residual node has 3 features; there are two such nodes => target shape (B, 6)

    # Initialize model
    model = make_model(
        DEVICE=DEVICE,
        nb_block=config['model']['nb_block'],
        in_channels=in_channels,
        K=config['model']['K'],
        nb_chev_filter=config['model']['nb_chev_filter'],
        nb_time_filter=config['model']['nb_time_filter'],
        time_strides=config['model']['time_strides'],
        adj_mx=adj_mx,
        num_for_predict=config['model']['num_for_predict'],
        len_input=config['model']['len_input'] + 1,  # Extended input sequence
        num_of_vertices=num_nodes,
        residual_dim=residual_dim
    )
    model.to(DEVICE)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

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
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs_batch, residuals_batch) in enumerate(train_loader):
            inputs_batch = inputs_batch.to(DEVICE)
            residuals_batch = residuals_batch.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass: outputs shape: (B, num_nodes, num_for_predict, residual_dim)
            outputs = model(inputs_batch)

            # Branch extraction based on graph structure
            if graph_nr in [1, 3, 6, 7]:
                # For these graphs, assume the residual node is at index num_joints+1.
                outputs_residual_node = outputs[:, num_joints+1, :, :]
                outputs_residual_node = outputs_residual_node.squeeze(1)  # Shape: (B, residual_dim)
            elif graph_nr == 2:
                # For graph 2, extract multiple residual nodes.
                num_residual_nodes = (num_nodes - num_joints) // 2
                outputs_residual_node = outputs[:, num_joints:num_joints+num_residual_nodes, :, :]
                if config['model']['num_for_predict'] == 1:
                    outputs_residual_node = outputs_residual_node.squeeze(2).squeeze(-1)  # Shape: (B, num_residual_nodes)
            elif graph_nr in [4, 5]:
                # For graphs 4 and 5, the graph splits residuals into two nodes:
                # one for position and one for orientation.
                # These are at indices num_joints+2 and num_joints+3.
                outputs_residual_node = outputs[:, num_joints+2:num_joints+4, :, :]
                if config['model']['num_for_predict'] == 1:
                    # Remove the time dimension and the last singleton dimension.
                    outputs_residual_node = outputs_residual_node.squeeze(2).squeeze(-1)  # Shape: (B, 2, residual_dim)
                    # Flatten the two residual nodes into a single vector per sample.
                    outputs_residual_node = outputs_residual_node.reshape(outputs_residual_node.shape[0], -1)
                    # For residual_dim = 3, this gives shape (B, 6)
            # Compute loss between predictions and targets
            loss = criterion(outputs_residual_node, residuals_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_batch, residuals_batch in val_loader:
                inputs_batch = inputs_batch.to(DEVICE)
                residuals_batch = residuals_batch.to(DEVICE)

                outputs = model(inputs_batch)
                if graph_nr in [1, 3, 6, 7]:
                    outputs_residual_node = outputs[:, num_joints+1, :, :].squeeze(1)
                elif graph_nr == 2:
                    num_residual_nodes = (num_nodes - num_joints) // 2
                    outputs_residual_node = outputs[:, num_joints:num_joints+num_residual_nodes, :, :]
                    if config['model']['num_for_predict'] == 1:
                        outputs_residual_node = outputs_residual_node.squeeze(2).squeeze(-1)
                elif graph_nr in [4, 5]:
                    outputs_residual_node = outputs[:, num_joints+2:num_joints+4, :, :]
                    if config['model']['num_for_predict'] == 1:
                        outputs_residual_node = outputs_residual_node.squeeze(2).squeeze(-1)
                        outputs_residual_node = outputs_residual_node.reshape(outputs_residual_node.shape[0], -1)
                loss = criterion(outputs_residual_node, residuals_batch)
                val_loss += loss.item() * inputs_batch.size(0)
        val_loss /= len(val_loader.dataset)

        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            saved_model_name = f"{model_name}_best_{graph_nr}_{prep_data_incl_past_residuals}_{current_time}.pth"
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
