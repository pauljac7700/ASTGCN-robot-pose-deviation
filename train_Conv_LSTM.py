"""Train the ConvLSTM baseline on the prepared ASTGCN splits.

Requires ``prepare_data_Conv_LSTM.py`` to have run first. Serves as the
no-graph baseline in the comparison.
"""

# train_Conv_LSTM.py

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
import joblib
from model.ConvLSTM import ConvLSTMWithFC  # Ensure this imports your updated ConvLSTM module


def load_config(config_path='config_Conv_LSTM.yaml'):
    """
    Loads the YAML configuration file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - config (dict): Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data(data_file):
    """
    Loads the preprocessed data from a .npz file.

    Parameters:
    - data_file (str): Path to the .npz data file.

    Returns:
    - inputs (numpy.ndarray): Input data.
    - targets (numpy.ndarray): Target data.
    """
    data = np.load(data_file)
    inputs = data['inputs']   # Shape: (N, T, C=6, H=8, W=1)
    residuals = data['residuals'] # Shape: (N, num_residuals)
    return inputs, residuals


def create_dataloader(inputs, residuals, batch_size, shuffle=True):
    """
    Creates a PyTorch DataLoader from inputs and targets.

    Parameters:
    - inputs (numpy.ndarray): Input data.
    - residuals (numpy.ndarray): Residuals data.
    - batch_size (int): Batch size.
    - shuffle (bool): Whether to shuffle the data.

    Returns:
    - DataLoader: PyTorch DataLoader.
    """
    tensor_x = torch.tensor(inputs, dtype=torch.float32)  # Convert to torch tensor
    tensor_y = torch.tensor(residuals, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def initialize_model(config):
    """
    Initializes the ConvLSTM model based on the configuration.

    Parameters:
    - config (dict): Configuration parameters.

    Returns:
    - model (ConvLSTMWithFC): Initialized ConvLSTM model with FC layer.
    """
    model_config = config['model']
    model = ConvLSTMWithFC(
        input_dim=model_config[config.get('dataset_dimension', {})]['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        kernel_size=model_config['kernel_size'],
        num_layers=model_config['num_layers'],
        output_dim=model_config[config.get('dataset_dimension', {})]['output_dim'],
        bias=model_config['bias'],
        return_all_layers=config['model']['return_all_layers']
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Parameters:
    - model (ConvLSTMWithFC): The ConvLSTM model with FC layer.
    - dataloader (DataLoader): Training DataLoader.
    - criterion (nn.Module): Loss function.
    - optimizer (optim.Optimizer): Optimizer.
    - device (torch.device): Device to train on.

    Returns:
    - avg_loss (float): Average loss over the epoch.
    """
    model.train()
    epoch_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.to(device)    # Shape: (B, T, C=6, H=8, W=1)
        batch_targets = batch_targets.to(device)  # Shape: (B, num_targets)
        
        optimizer.zero_grad()
        
        preds = model(batch_inputs)  # Shape: (B, num_targets)
        
        loss = criterion(preds, batch_targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_inputs.size(0)
    
    avg_loss = epoch_loss / len(dataloader.dataset)
    return avg_loss


def validate_epoch(model, dataloader, criterion, device):
    """
    Validates the model for one epoch.

    Parameters:
    - model (ConvLSTMWithFC): The ConvLSTM model with FC layer.
    - dataloader (DataLoader): Validation DataLoader.
    - criterion (nn.Module): Loss function.
    - device (torch.device): Device to validate on.

    Returns:
    - avg_loss (float): Average validation loss.
    """
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            preds = model(batch_inputs)  # Shape: (B, num_targets)
            
            loss = criterion(preds, batch_targets)
            epoch_loss += loss.item() * batch_inputs.size(0)
    
    avg_loss = epoch_loss / len(dataloader.dataset)
    return avg_loss


def save_model(model, optimizer, epoch, loss, save_path):
    """
    Saves the model checkpoint.

    Parameters:
    - model (ConvLSTMWithFC): The ConvLSTM model with FC layer.
    - optimizer (optim.Optimizer): Optimizer.
    - epoch (int): Current epoch.
    - loss (float): Current loss.
    - save_path (str): Path to save the model.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def train_model(config):
    """
    Main training loop for the ConvLSTM model.

    Parameters:
    - config (dict): Configuration parameters loaded from the YAML file.
    """
    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Device configuration
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets and infos
    dataset_dimension = config['dataset_dimension']
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    model_name = config['model_name']['multi']
    num_joints = config['num_joints']

    if config['prep_data_incl_past_residuals']: 
        prep_data_incl_past_residuals = 'wr'
        print("Including past residuals in input features.")
    else:   
        prep_data_incl_past_residuals = 'nr'
        print("Excluding past residuals from input features.")

    # Load datasets
    print("Loading training data...")
    train_inputs, train_residuals = load_data(f'data/convlstm_train_{prep_data_incl_past_residuals}.npz')
    print("Loading validation data...")
    val_inputs, val_residuals = load_data(f'data/convlstm_val_{prep_data_incl_past_residuals}.npz')

    # Create DataLoaders
    train_loader = create_dataloader(train_inputs, train_residuals, 
                                     config['training']['batch_size'], shuffle=False)
    val_loader = create_dataloader(val_inputs, val_residuals, 
                                   config['training']['batch_size'], shuffle=False)

    # Initialize model
    model = initialize_model(config)
    model.to(device)
    print(model)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])

    # Setup logging with TensorBoard
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs',dataset_dimension,dataset_name,dataset_type,model_name, f'{model_name}_{current_time}_{dataset_dimension}_{dataset_name}_{dataset_type}_{prep_data_incl_past_residuals}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Create directory for saving models if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    model_save_dir = os.path.join('saved_models',dataset_dimension,dataset_name,dataset_type,model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # Combine hyperparameters for logging, converting lists to strings
    hparams = {
        'input_dim': config['model'][dataset_dimension]['input_dim'],
        'hidden_dim': str(config['model']['hidden_dim']),     # Convert list to string
        'kernel_size': str(config['model']['kernel_size']),   # Convert list of lists to string
        'num_layers': config['model']['num_layers'],
        'output_dim': config['model'][dataset_dimension]['output_dim'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'patience': config['training']['patience'],
        'num_epochs': config['training']['num_epochs']
    }

    # Early stopping parameters
    patience = config['training']['patience']
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.6f}")

        # Log losses to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save the best model
            model_name = f"convlstm_best_{prep_data_incl_past_residuals}_{current_time}.pth"
            save_path = os.path.join(model_save_dir, model_name)
            save_model(model, optimizer, epoch, val_loss, save_path)
            print(f"Epoch {epoch}: Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch}: No improvement in validation loss.")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    # Log hyperparameters and best validation loss to TensorBoard
    try:
        writer.add_hparams(hparams, {'hparam/val_loss': best_val_loss})
    except ValueError as e:
        print(f"Error logging hyperparameters: {e}")
        print("Ensure that all hyperparameter values are of type int, float, str, bool, or torch.Tensor.")
    
    # Finalize TensorBoard logging
    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_Conv_LSTM.yaml",
                        help="Path to the YAML configuration file (default: %(default)s).")
    args = parser.parse_args()

    config = load_config(args.config)
    train_model(config)
