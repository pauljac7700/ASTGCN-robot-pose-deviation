# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import yaml
from model.adjusted_ASTGCN import make_model
import joblib

def train_model(config):

    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Load datasets
    train_data = np.load(config['train_data_file'])
    val_data = np.load(config['val_data_file'])
    inputs_train = train_data['inputs']
    targets_train = train_data['targets']
    inputs_val = val_data['inputs']
    targets_val = val_data['targets']

    # Load scalers (if needed)
    scalers = joblib.load(config['scalers_file'])
    # No need to apply scalers here since data is already scaled

    # Convert to PyTorch tensors
    inputs_train_tensor = torch.from_numpy(inputs_train).float()
    targets_train_tensor = torch.from_numpy(targets_train).float()
    inputs_val_tensor = torch.from_numpy(inputs_val).float()
    targets_val_tensor = torch.from_numpy(targets_val).float()

    # Create datasets and data loaders
    train_dataset = TensorDataset(inputs_train_tensor, targets_train_tensor)
    val_dataset = TensorDataset(inputs_val_tensor, targets_val_tensor)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

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

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    # TensorBoard setup
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['logging']['log_dir'], current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model_save_dir = config['logging']['model_save_dir']
    os.makedirs(model_save_dir, exist_ok=True)

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
        for batch_idx, (inputs_batch, targets_batch) in enumerate(train_loader):
            inputs_batch = inputs_batch.to(DEVICE)
            targets_batch = targets_batch.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs_batch)  # Shape: (batch_size, N, num_for_predict, target_dim)

            # Extract outputs for the end-effector node (Node 6)
            outputs_end_effector = outputs[:, 6, :, :]  # Shape: (batch_size, num_for_predict, target_dim)

            # Flatten outputs and targets
            outputs_flat = outputs_end_effector.view(outputs_end_effector.size(0), -1)  # Shape: (batch_size, num_for_predict * target_dim)
            targets_flat = targets_batch.view(targets_batch.size(0), -1)  # Shape: (batch_size, num_for_predict * target_dim)

            # Compute loss
            loss = criterion(outputs_flat, targets_flat)

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
            for inputs_batch, targets_batch in val_loader:
                inputs_batch = inputs_batch.to(DEVICE)
                targets_batch = targets_batch.to(DEVICE)

                outputs = model(inputs_batch)
                outputs_end_effector = outputs[:, 6, :, :]  # Shape: (batch_size, num_for_predict, target_dim)

                # Flatten outputs and targets
                outputs_flat = outputs_end_effector.view(outputs_end_effector.size(0), -1)  # Shape: (batch_size, num_for_predict * target_dim)
                targets_flat = targets_batch.view(targets_batch.size(0), -1)  # Shape: (batch_size, num_for_predict * target_dim)

                # Compute loss
                loss = criterion(outputs_flat, targets_flat)
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
            model_name = f"astgcn_best_{current_time}.pth"
            save_path = os.path.join(model_save_dir, model_name)
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
    # Load configuration
    with open('config_ASTGCN.yaml') as f:
        config = yaml.safe_load(f)

    train_model(config)
