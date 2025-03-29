import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import yaml
from sklearn.model_selection import train_test_split
import joblib  # For saving the model
from torch.utils.tensorboard import SummaryWriter  # TensorBoard SummaryWriter
from datetime import datetime  # For generating run identifiers

# Custom metric functions
def masked_mape(preds, labels, null_val=np.nan, epsilon=1e-8):
    '''
    Compute Mean Absolute Percentage Error with masking.
    If null_val is provided, positions with that value are masked out.
    Parameters:
        preds (array-like): Predicted values.
        labels (array-like): Actual values.
        null_val (float or np.nan): Value to mask out in labels.
        epsilon (float): Small threshold to exclude very small labels.
    Returns:
        float: MAPE percentage.
    '''
    # Create mask based on null_val
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = labels != null_val
    
    # Exclude zero or very small labels to prevent division by zero or inflated MAPE
    mask &= np.abs(labels) > epsilon
    
    if not np.any(mask):
        return np.nan  # or raise ValueError("All labels are masked.")
    
    # Apply mask
    masked_labels = labels[mask]
    masked_preds = preds[mask]
    
    # Compute MAPE
    mape = np.abs((masked_labels - masked_preds) / masked_labels) * 100
    
    return np.mean(mape)

def masked_smape(y_pred, y_true, null_val=0):
    mask = y_true != null_val
    denominator = np.abs(y_true[mask]) + np.abs(y_pred[mask])
    # To avoid division by zero, set denominator to 1 where it's zero
    denominator = np.where(denominator == 0, 1, denominator)
    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator) * 100

def masked_mdape(y_pred, y_true, null_val=0):
    mask = y_true != null_val
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def masked_mae(y_pred, y_true, null_val=0):
    mask = y_true != null_val
    return mean_absolute_error(y_true[mask], y_pred[mask])

def masked_r2_score(y_pred, y_true, null_val=0):
    mask = y_true != null_val
    return r2_score(y_true[mask], y_pred[mask])

def prepare_data(config):
    # Load CSV data
    df = pd.read_csv(config['data_file'])
    df.dropna(inplace=True)

    # Define input and target features
    joint_features = [f'joint_{i}' for i in range(1, 7)]
    setpoint_features = ['x_set', 'y_set', 'z_set', 'rx_set', 'ry_set', 'rz_set']

    input_features = joint_features + setpoint_features
    target_variable = config['target_variable']  # Single target

    # Extract features and target
    X = df[input_features].values
    y = df[target_variable].values  # y is now a 1D array

    return X, y

def train_model(config, X_train, y_train, writer, n_epochs):
    # Initialize the MLPRegressor
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(config['hidden_layer_sizes']),
        activation=config['activation'],
        solver=config['solver'],
        shuffle=False,       # Disables shuffling during training
        max_iter=n_epochs,   # Set to n_epochs
        early_stopping=True,   
        batch_size='auto',   # Automatically determine batch size
        random_state=config['random_state']
    )

    writer.add_text('Training', f"Starting training for {n_epochs} epochs.")
    writer.add_text('Training', f"Hidden Layers: {config['hidden_layer_sizes']}, Activation: {config['activation']}, Solver: {config['solver']}", 0)

    # Fit the model once for n_epochs
    mlp.fit(X_train, y_train)

    # Log the loss curve after training
    loss_curve = mlp.loss_curve_
    for epoch, loss in enumerate(loss_curve, 1):
        writer.add_scalar('Training Loss', loss, epoch)
        writer.add_text('Training', f"Epoch {epoch}/{n_epochs} - Loss: {loss}")

    writer.add_text('Training', "Model training completed.")
    return mlp

def plot_results(y_test, y_test_pred, target_variable, results_dir, writer):
    # Time Series Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_test_pred, label='Predicted')
    plt.legend()
    plt.title(f'Actual vs. Predicted {target_variable}')
    plt.xlabel('Sample Index')
    plt.ylabel(target_variable)
    plt.tight_layout()
    # Save the figure
    plot_filename = f"MLP_timeseries.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    # Log the plot to TensorBoard
    img = plt.imread(plot_path)
    writer.add_image("Time Series Plots", img, dataformats='HWC')

    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scatter Plot for {target_variable}')
    plt.tight_layout()
    scatter_plot_filename = f"MLP_scatter_{target_variable}.png"
    scatter_plot_path = os.path.join(results_dir, scatter_plot_filename)
    plt.savefig(scatter_plot_path)
    plt.close()
    # Read and log to TensorBoard
    img = plt.imread(scatter_plot_path)
    writer.add_image(f"Scatter Plots/{target_variable}", img, dataformats='HWC')

    # Residual Plot
    residuals = y_test - y_test_pred
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), colors='r', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {target_variable}')
    plt.tight_layout()
    residual_plot_filename = f"MLP_residual_{target_variable}.png"
    residual_plot_path = os.path.join(results_dir, residual_plot_filename)
    plt.savefig(residual_plot_path)
    plt.close()
    # Read and log to TensorBoard
    img = plt.imread(residual_plot_path)
    writer.add_image(f"Residual Plots/{target_variable}", img, dataformats='HWC')

    # Error Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'Error Histogram for {target_variable}')
    plt.tight_layout()
    histogram_filename = f"MLP_histogram_{target_variable}.png"
    histogram_path = os.path.join(results_dir, histogram_filename)
    plt.savefig(histogram_path)
    plt.close()
    # Read and log to TensorBoard
    img = plt.imread(histogram_path)
    writer.add_image(f"Error Histograms/{target_variable}", img, dataformats='HWC')

def evaluate_and_save_results(config, y_test, y_test_pred, target_variable, model_identifier, results_dir, writer):
    # Compute metrics
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = masked_mae(y_test_pred, y_test)
    mape = masked_mape(y_test_pred, y_test)
    smape = masked_smape(y_test_pred, y_test)
    mdape = masked_mdape(y_test_pred, y_test)
    r2 = masked_r2_score(y_test_pred, y_test)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'sMAPE': smape,
        'MdAPE': mdape,
        'R2': r2
    }

    # Log individual metrics to TensorBoard
    writer.add_scalar(f"Metrics/MSE", mse, 0)
    writer.add_scalar(f"Metrics/RMSE", rmse, 0)
    writer.add_scalar(f"Metrics/MAE", mae, 0)
    writer.add_scalar(f"Metrics/MAPE", mape, 0)
    writer.add_scalar(f"Metrics/sMAPE", smape, 0)
    writer.add_scalar(f"Metrics/MdAPE", mdape, 0)
    writer.add_scalar(f"Metrics/R2", r2, 0)

    # Save metrics to a file
    metrics_file = os.path.join(results_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Metrics for {target_variable}:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.6f}\n")

    # Log the metrics file as text in TensorBoard
    with open(metrics_file, 'r') as f:
        metrics_content = f.read()
    writer.add_text("Metrics/Text", metrics_content, 0)

    # Save detailed predictions
    predictions_file = os.path.join(results_dir, "predictions.csv")
    predictions_data = {
        f"{target_variable}_Actual": y_test,
        f"{target_variable}_Predicted": y_test_pred,
        f"{target_variable}_Residual": y_test - y_test_pred
    }
    df_predictions = pd.DataFrame(predictions_data)
    df_predictions.to_csv(predictions_file, index=False)

    # Log the predictions file as text in TensorBoard (optional)
    writer.add_text("Predictions/DataFrame", df_predictions.head().to_html(), 0)

    return metrics

def save_model(config, model, model_identifier, model_save_dir_run, writer):
    os.makedirs(model_save_dir_run, exist_ok=True)
    model_file = os.path.join(model_save_dir_run, f"{model_identifier}.joblib")
    joblib.dump(model, model_file)
    # Log the model save event in TensorBoard
    writer.add_text("Model Save", f"Model saved to {model_file}", 0)
    return model_file  # Return the path for logging

def setup_tensorboard(log_dir_run, config):
    os.makedirs(log_dir_run, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_run)
    # Log initial configuration as text
    with open('config_MLP_single.yaml', 'r') as f:  # Updated config file name
        config_text = f.read()
    writer.add_text("Config", config_text, 0)
    return writer

def main():
    # Load configuration from YAML
    with open('config_MLP_single.yaml', 'r') as file:  # Updated config file name
        config = yaml.safe_load(file)

    # Extract target variable from configuration
    target_variable = config['target_variable']  # e.g., 'x_dif'

    # Generate a unique run identifier with target variable name
    run_id = f"MLP_{target_variable}_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define base directories from the configuration
    base_log_dir = config['logging']['log_dir']
    base_model_save_dir = config['logging']['model_save_dir']
    base_results_dir = config['logging']['results_dir']

    # Create run-specific directories
    log_dir_run = os.path.join(base_log_dir, run_id)
    model_save_dir_run = os.path.join(base_model_save_dir, run_id)
    results_dir_run = os.path.join(base_results_dir, run_id)

    os.makedirs(log_dir_run, exist_ok=True)
    os.makedirs(model_save_dir_run, exist_ok=True)
    os.makedirs(results_dir_run, exist_ok=True)

    # Set up TensorBoard
    writer = setup_tensorboard(log_dir_run, config)

    # Log the start of the training pipeline
    writer.add_text("Training Pipeline", "Starting the training pipeline.", 0)

    # Prepare data
    X, y = prepare_data(config)
    writer.add_text("Data", f"Data prepared with shape X: {X.shape}, y: {y.shape}", 0)

    # Train-test split without shuffling
    test_size = config['test_size']
    random_state = config['random_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=False  # Disables shuffling to maintain data order
    )
    writer.add_text("Data Split", f"Data split into train: {X_train.shape}, test: {X_test.shape}", 0)

    # Scale input and target data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit on training data only and flatten to 1D array
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()  # Flatten to 1D array

    # Transform test data and flatten to 1D array
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()  # Flatten to 1D array

    writer.add_text("Data Scaling", "Data scaling completed. Scalers fitted on training data only.", 0)

    # Train the model with TensorBoard logging
    n_epochs = config['epochs']
    model = train_model(config, X_train_scaled, y_train_scaled, writer, n_epochs)

    # Save the trained model
    model_identifier = "mlp_model"
    model_file = save_model(config, model, model_identifier, model_save_dir_run, writer)

    # Evaluate on test set
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred_scaled = y_test_pred_scaled.reshape(-1, 1)  # Reshape to 2D array for inverse_transform
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled).ravel()
    y_test_unscaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    writer.add_text("Evaluation", "Model evaluation on test set completed.", 0)

    # Plot results
    plot_results(y_test_unscaled, y_test_pred_unscaled, target_variable, results_dir_run, writer)

    # Save and evaluate results with TensorBoard logging
    evaluate_and_save_results(config, y_test_unscaled, y_test_pred_unscaled, target_variable, model_identifier, results_dir_run, writer)

    # Log the loss curve to TensorBoard
    for epoch, loss in enumerate(model.loss_curve_, 1):
        writer.add_scalar('Training/Loss_Curve', loss, epoch)

    writer.add_text("Training Pipeline", "Training pipeline completed successfully.", 0)

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
