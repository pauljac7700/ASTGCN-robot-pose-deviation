"""Multi-layer perceptron baseline, multi-target.

Takes the same features as the ASTGCN but discards the graph structure, so the
difference between the two measures what modelling the kinematic chain is worth.
"""

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
    # Load data
    with open(config['locate_data_file'], 'r') as r:
        locate_data = yaml.safe_load(r)

    print("Dataset Dimension:", config['dataset_dimension'])
    print("Dataset Name:", config['dataset_name'])
    print("Dataset Type:", config['dataset_type'])
    num_joints = config['num_joints']

    # Extract dataset file path from YAML
    grid_file = locate_data.get(config['dataset_dimension'], {}).get(config['dataset_name'], {}).get(config['dataset_type'], None)

    # Construct full dataset path
    datapath_to_dataset = f"data/{config['dataset_dimension']}_datasets/{grid_file}"
    df = pd.read_csv(datapath_to_dataset)
    df=df.loc[1:]

    # Drop unnecessary columns and check for missing values
    df = df.drop(columns=['step_order'], errors='ignore')
    if df.isnull().values.any():
        df = df.dropna()

    # Define input features and residual variables
    residual_variables = config.get('residual_variables', {}).get(config['dataset_dimension'], [])
    joint_features = [f'joint_{i}' for i in range(1, num_joints + 1)]
    target_pose_features = config.get('target_pose_variables', {}).get(config['dataset_dimension'], [])

    # Exclude residuals from input features
    input_features = joint_features + target_pose_features

    # Extract features and targets
    X = df[input_features].values
    y = df[residual_variables].values

    return X, y

def train_model(config, X_train, y_train, writer, n_epochs):
    # Initialize the MLPRegressor
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(config['hidden_layer_sizes']),
        activation=config['activation'],
        solver=config['solver'],
        shuffle=False,
        max_iter=n_epochs,  # Set to n_epochs
        warm_start=False,   # Not needed when calling fit() once
        batch_size='auto',  # Automatically determine batch size
        random_state=config['random_state']
    )

    writer.add_text('Training', f"Starting training for {n_epochs} epochs.")

    # Fit the model once for n_epochs
    mlp.fit(X_train, y_train)

    # Log the loss curve after training
    loss_curve = mlp.loss_curve_
    for epoch, loss in enumerate(loss_curve, 1):
        writer.add_scalar('Training Loss', loss, epoch)
        writer.add_text('Training', f"Epoch {epoch}/{n_epochs} - Loss: {loss}")

    writer.add_text('Training', "Model training completed.")
    return mlp

def plot_results(y_test, y_test_pred, residual_variables, results_dir, writer):
    num_targets = len(residual_variables)

    # Time Series Plot
    plt.figure(figsize=(12, 6 * num_targets))
    for idx, residual_var in enumerate(residual_variables):
        plt.subplot(num_targets, 1, idx + 1)
        plt.plot(y_test[:, idx], label='Actual')
        plt.plot(y_test_pred[:, idx], label='Predicted')
        plt.legend()
        plt.title(f'Actual vs. Predicted {residual_var}')
        plt.xlabel('Sample Index')
        plt.ylabel(residual_var)
    plt.tight_layout()
    # Save the figure
    plot_filename = f"MLP_timeseries.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    # Log the plot to TensorBoard
    img = plt.imread(plot_path)
    writer.add_image("Time Series Plots", img, dataformats='HWC')

    # Scatter Plots, Residual Plots, and Error Histograms
    for idx, residual_var in enumerate(residual_variables):
        # Scatter Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test[:, idx], y_test_pred[:, idx], alpha=0.5)
        plt.plot([y_test[:, idx].min(), y_test[:, idx].max()],
                 [y_test[:, idx].min(), y_test[:, idx].max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Scatter Plot for {residual_var}')
        plt.tight_layout()
        scatter_plot_filename = f"MLP_scatter_{residual_var}.png"
        scatter_plot_path = os.path.join(results_dir, scatter_plot_filename)
        plt.savefig(scatter_plot_path)
        plt.close()
        # Read and log to TensorBoard
        img = plt.imread(scatter_plot_path)
        writer.add_image(f"Scatter Plots/{residual_var}", img, dataformats='HWC')

        # Residual Plot
        residuals = y_test[:, idx] - y_test_pred[:, idx]
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test_pred[:, idx], residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_test_pred[:, idx].min(), xmax=y_test_pred[:, idx].max(), colors='r', linestyles='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for {residual_var}')
        plt.tight_layout()
        residual_plot_filename = f"MLP_residual_{residual_var}.png"
        residual_plot_path = os.path.join(results_dir, residual_plot_filename)
        plt.savefig(residual_plot_path)
        plt.close()
        # Read and log to TensorBoard
        img = plt.imread(residual_plot_path)
        writer.add_image(f"Residual Plots/{residual_var}", img, dataformats='HWC')

        # Error Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title(f'Error Histogram for {residual_var}')
        plt.tight_layout()
        histogram_filename = f"MLP_histogram_{residual_var}.png"
        histogram_path = os.path.join(results_dir, histogram_filename)
        plt.savefig(histogram_path)
        plt.close()
        # Read and log to TensorBoard
        img = plt.imread(histogram_path)
        writer.add_image(f"Error Histograms/{residual_var}", img, dataformats='HWC')

def evaluate_and_save_results(config, y_test, y_test_pred, residual_variables, model_identifier, results_dir, writer):
    # Initialize metric lists
    metrics = {}
    mse_list = []
    rmse_list = []
    mae_list = []
    mape_list = []
    smape_list = []
    mdape_list = []
    r2_list = []
    residuals_dict = {}

    # Compute metrics for each residual variable
    for idx, residual_var in enumerate(residual_variables):
        mse = mean_squared_error(y_test[:, idx], y_test_pred[:, idx])
        rmse = np.sqrt(mse)
        mape = masked_mape(y_test_pred[:, idx], y_test[:, idx])
        smape = masked_smape(y_test_pred[:, idx], y_test[:, idx])
        mdape = masked_mdape(y_test_pred[:, idx], y_test[:, idx])
        mae = masked_mae(y_test_pred[:, idx], y_test[:, idx])
        r2 = masked_r2_score(y_test_pred[:, idx], y_test[:, idx])

        metrics[residual_var] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'sMAPE': smape,
            'MdAPE': mdape,
            'R2': r2
        }

        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        smape_list.append(smape)
        mdape_list.append(mdape)
        r2_list.append(r2)

        residuals_dict[residual_var] = y_test[:, idx] - y_test_pred[:, idx]

        # Log individual metrics to TensorBoard
        writer.add_scalar(f"Metrics/{residual_var}/MSE", mse, 0)
        writer.add_scalar(f"Metrics/{residual_var}/RMSE", rmse, 0)
        writer.add_scalar(f"Metrics/{residual_var}/MAE", mae, 0)
        writer.add_scalar(f"Metrics/{residual_var}/MAPE", mape, 0)
        writer.add_scalar(f"Metrics/{residual_var}/sMAPE", smape, 0)
        writer.add_scalar(f"Metrics/{residual_var}/MdAPE", mdape, 0)
        writer.add_scalar(f"Metrics/{residual_var}/R2", r2, 0)

    # Compute mean metrics over all residual variables
    mean_metrics = {
        'MSE': np.mean(mse_list),
        'RMSE': np.mean(rmse_list),
        'MAE': np.mean(mae_list),
        'MAPE': np.mean(mape_list),
        'sMAPE': np.mean(smape_list),
        'MdAPE': np.median(mdape_list),
        'R2': np.mean(r2_list)
    }

    # Log mean metrics to TensorBoard
    for key, value in mean_metrics.items():
        writer.add_scalar(f"Metrics/Mean/{key}", value, 0)

    # Save metrics to a file
    metrics_file = os.path.join(results_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Test Results:\n")
        for residual_var, metric in metrics.items():
            f.write(f"Metrics for {residual_var}:\n")
            for key, value in metric.items():
                f.write(f"  {key}: {value:.6f}\n")
        f.write("\nMean Metrics over all residual variables:\n")
        for key, value in mean_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")

    # Log the metrics file as text in TensorBoard
    with open(metrics_file, 'r') as f:
        metrics_content = f.read()
    writer.add_text("Metrics/Text", metrics_content, 0)

    # Save detailed predictions
    predictions_file = os.path.join(results_dir, "predictions.csv")
    predictions_data = {}
    for i, var in enumerate(residual_variables):
        predictions_data[f"{var}_Actual"] = y_test[:, i]
        predictions_data[f"{var}_Predicted"] = y_test_pred[:, i]
        predictions_data[f"{var}_Residual"] = residuals_dict[var]
    df_predictions = pd.DataFrame(predictions_data)
    df_predictions.to_csv(predictions_file, index=False)

    # Log the predictions file as text in TensorBoard (optional)
    writer.add_text("Predictions/DataFrame", df_predictions.head().to_html(), 0)

    # Identify indices for x, y, z values
    xyz_indices = [idx for idx, residual_var in enumerate(residual_variables) if residual_var in ['x_dif', 'y_dif', 'z_dif']]

    # Calculate Euclidean distance for each sample using only x, y, z values
    euclidean_distances = np.linalg.norm(y_test[:, xyz_indices] - y_test_pred[:, xyz_indices], axis=1)

    # Calculate Euclidean distance for actual residuals
    actual_euclidean_distances = np.linalg.norm(y_test[:, xyz_indices], axis=1)
    
    # Calculate mean values
    mean_euclidean = np.mean(euclidean_distances)
    mean_actual_euclidean = np.mean(actual_euclidean_distances)

    # Create a line plot for Euclidean distances with mean values indicated in the legend
    plt.figure(figsize=(12, 6))
    plt.plot(actual_euclidean_distances,
             label=f'Actual Residuals (Euclidean Distance) - Mean: {mean_actual_euclidean:.2f}',
             color='green')
    plt.plot(euclidean_distances,
             label=f'Difference (Actual - Predicted) - Mean: {mean_euclidean:.2f}',
             color='blue')
    plt.title('Euclidean Distance Over Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    euclidean_plot_filename = f"{model_identifier}_euclidean_distance_plot.png"
    euclidean_plot_path = os.path.join(results_dir, euclidean_plot_filename)
    plt.savefig(euclidean_plot_path)
    plt.close()
    print(f"Euclidean distance plot saved to {euclidean_plot_path}")

    return metrics, mean_metrics

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
    with open('config_MLP_multi.yaml', 'r') as f:
        config_text = f.read()
    writer.add_text("Config", config_text, 0)
    return writer

def main():
    # Load configuration from YAML
    with open('config_MLP_multi.yaml', 'r') as file:
        config = yaml.safe_load(file)

    dataset_dimension = config['dataset_dimension']
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    model_name = config['model_name']['multi']

    # Generate a unique run identifier
    run_id = "MLP_" +datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define base directories from the configuration
    base_dir = os.path.join(dataset_dimension,dataset_name,dataset_type,model_name)
    log_dir = os.path.join('logs', base_dir)
    model_save_dir = os.path.join('saved_models', base_dir)
    results_dir = os.path.join('results', base_dir)

    # Create run-specific directories
    log_dir_run = os.path.join(log_dir, run_id)
    model_save_dir_run = os.path.join(model_save_dir, run_id)
    results_dir_run = os.path.join(results_dir, run_id)

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

    # Train-validation-test split (validation removed)
    test_size = config['test_size']
    random_state = config['random_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=False
    )
    writer.add_text("Data Split", f"Data split into train: {X_train.shape}, test: {X_test.shape}", 0)

    # Scale input and target data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Transform test data
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    writer.add_text("Data Scaling", "Data scaling completed. Scalers fitted on training data only.", 0)

    # Train the model with TensorBoard logging
    n_epochs = config['epochs']
    model = train_model(config, X_train_scaled, y_train_scaled, writer, n_epochs)

    # Save the trained model
    model_identifier = "MLP"
    model_file = save_model(config, model, model_identifier, model_save_dir_run, writer)

    # Evaluate on test set
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)

    writer.add_text("Evaluation", "Model evaluation on test set completed.", 0)

    # Plot results
    plot_results(y_test_unscaled, y_test_pred_unscaled, config['residual_variables'][dataset_dimension], results_dir_run, writer)

    # Save and evaluate results with TensorBoard logging
    evaluate_and_save_results(config, y_test_unscaled, y_test_pred_unscaled, config['residual_variables'][dataset_dimension], model_identifier, results_dir_run, writer)

    # Log the loss curve to TensorBoard
    for epoch, loss in enumerate(model.loss_curve_, 1):
        writer.add_scalar('Training/Loss_Curve', loss, epoch)

    writer.add_text("Training Pipeline", "Training pipeline completed successfully.", 0)

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
