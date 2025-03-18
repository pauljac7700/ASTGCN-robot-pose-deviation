#train_STGAT.py

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

from model.STGAT import STGAT  # Ensure this matches your directory structure

def train_model(config):

    # Set random seeds
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Load datasets
    train_data = np.load(config['train_data_file'])
    val_data = np.load(config['val_data_file'])
    inputs_train = train_data['inputs']   # (B, N, F, T_in)
    targets_train = train_data['targets'] # (B, num_targets)
    inputs_val = val_data['inputs']
    targets_val = val_data['targets']

    print('inputs_train shape: '+str(inputs_train.shape))

    # Load scalers
    scalers = joblib.load(config['scalers_file'])
    # Data is already scaled

    # Convert to tensors
    inputs_train_tensor = torch.from_numpy(inputs_train).float()
    targets_train_tensor = torch.from_numpy(targets_train).float()
    inputs_val_tensor = torch.from_numpy(inputs_val).float()
    targets_val_tensor = torch.from_numpy(targets_val).float()

    # Create Datasets and DataLoaders
    train_dataset = TensorDataset(inputs_train_tensor, targets_train_tensor)
    val_dataset = TensorDataset(inputs_val_tensor, targets_val_tensor)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load adjacency matrix
    adj_mx = np.load(config['adjacency_matrix_file'])
    adj_mx = torch.tensor(adj_mx, dtype=torch.float).to(DEVICE)

    
    num_targets = len(config['target_variables'])

    # Initialize STGAT model
    # Pass target_dim if you modified STGAT to handle multiple targets at once
    model = STGAT(
        cuda=(config['device'] != 'cpu'),
        num_nodes=config['model']['num_of_vertices'],
        num_features=config['model']['num_features'],
        num_timesteps_input=config['model']['num_timesteps_input'],
        num_timesteps_output=config['model']['num_timesteps_output'],
        nheads=config['model']['nheads'],
        nhid=config['model']['nhid'],
        layers=config['model']['layers']
    )
    # If you modified STGAT to accept target_dim:
    # model.target_dim = num_targets

    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['logging']['log_dir'], f'STGAT_{current_time}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model_save_dir = config['logging']['model_save_dir']
    os.makedirs(model_save_dir, exist_ok=True)

    hparams = {**config['model'], **config['training']}

    patience = config['training']['patience']
    best_val_loss = float('inf')
    epochs_no_improve = 0
    num_epochs = config['training']['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs_batch, targets_batch) in enumerate(train_loader):
            inputs_batch = inputs_batch.to(DEVICE)
            targets_batch = targets_batch.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass STGAT
            # STGAT output by default: (B, N, T_out, 1)
            # If modified for multiple targets: (B, N, T_out, num_targets)
            outputs = model(adj_mx, inputs_batch)  # (B, N, T_out, num_targets)

            # Extract node 7
            outputs_node7 = outputs[:, 7, :, :]  # (B, T_out, num_targets)
            # If T_out == 1, squeeze time dimension
            outputs_node7 = outputs_node7.squeeze(1) # (B, num_targets)

            loss = criterion(outputs_node7, targets_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_batch, targets_batch in val_loader:
                inputs_batch = inputs_batch.to(DEVICE)
                targets_batch = targets_batch.to(DEVICE)

                outputs = model(adj_mx, inputs_batch)
                outputs_node7 = outputs[:, 7, :, :].squeeze(1)  # (B, num_targets)
                loss = criterion(outputs_node7, targets_batch)
                val_loss += loss.item() * inputs_batch.size(0)
        val_loss /= len(val_loader.dataset)

        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_name = f"stgat_best_{current_time}.pth"
            save_path = os.path.join(model_save_dir, model_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'hyperparameters': hparams
            }, save_path)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} - Best Model Saved")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} - No Improvement")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    metrics = {'hparam/val_loss': val_loss}
    writer.add_hparams(hparams, metrics)
    writer.close()

if __name__ == "__main__":
    with open('config_STGAT.yaml') as f:
        config = yaml.safe_load(f)

    train_model(config)
