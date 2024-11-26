# train_TGCN.py

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
from model.TGCN import TGCNWithGlobalOutput  # Ensure this imports your updated TGCN module


def train_model(config):

    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Load datasets
    train_data = np.load(config['train_data_file'])
    val_data = np.load(config['val_data_file'])
    inputs_train = train_data['inputs']  # Shape: (num_samples, num_nodes=8, in_channels=6, seq_len)
    targets_train = train_data['targets']  # Shape: (num_samples, num_targets)
    inputs_val = val_data['inputs']
    targets_val = val_data['targets']

    # Load scalers (if needed)
    scalers = joblib.load(config['scalers_file'])
    # No need to apply scalers here since data is already scaled

    # Process inputs to be compatible with TGCN model
    # TGCN now expects inputs of shape (batch_size, seq_len, num_nodes, in_channels)
    inputs_train = inputs_train.transpose(0, 3, 1, 2)  # Shape: (num_samples, seq_len, num_nodes, in_channels)
    inputs_val = inputs_val.transpose(0, 3, 1, 2)

    # Convert to PyTorch tensors
    inputs_train_tensor = torch.from_numpy(inputs_train).float()
    targets_train_tensor = torch.from_numpy(targets_train).float()
    inputs_val_tensor = torch.from_numpy(inputs_val).float()
    targets_val_tensor = torch.from_numpy(targets_val).float()

    # Create datasets and data loaders
    train_dataset = TensorDataset(inputs_train_tensor, targets_train_tensor)
    val_dataset = TensorDataset(inputs_val_tensor, targets_val_tensor)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])

    # Device configuration
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Initialize model
    in_channels = config['model']['in_channels']  # Should be 6
    hidden_dim = config['model']['hidden_dim']
    tgcn_model = TGCNWithGlobalOutput(adj=adj_mx, in_channels=in_channels, hidden_dim=hidden_dim)
    tgcn_model.to(DEVICE)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(tgcn_model.parameters(),
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])

    # TensorBoard setup
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['logging']['log_dir'], current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model_save_dir = config['logging']['model_save_dir']
    os.makedirs(model_save_dir, exist_ok=True)

    # Combine hyperparameters for logging
    hparams = {**config['model'], **config['training']}

    # Early stopping parameters
    patience = config['training'].get('patience', 10)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        tgcn_model.train()
        train_loss = 0.0
        for batch_idx, (inputs_batch, targets_batch) in enumerate(train_loader):
            inputs_batch = inputs_batch.to(DEVICE)  # Shape: (batch_size, seq_len, num_nodes, in_channels)
            targets_batch = targets_batch.to(DEVICE)  # Shape: (batch_size, num_targets)

            optimizer.zero_grad()

            # Forward pass
            outputs = tgcn_model(inputs_batch)  # Outputs shape: (batch_size, num_targets)

            # Compute loss
            loss = criterion(outputs, targets_batch)

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
            for inputs_batch, targets_batch in val_loader:
                inputs_batch = inputs_batch.to(DEVICE)
                targets_batch = targets_batch.to(DEVICE)

                outputs = tgcn_model(inputs_batch)

                loss = criterion(outputs, targets_batch)
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
            model_name = f"tgcn_best_{current_time}.pth"
            save_path = os.path.join(model_save_dir, model_name)
            torch.save({
                'epoch': epoch + 1,
                'tgcn_state_dict': tgcn_model.state_dict(),
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

    train_model(config)
