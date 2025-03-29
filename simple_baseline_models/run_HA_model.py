import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def handle_missing_values(df):
    initial_shape = df.shape
    df_clean = df.dropna().reset_index(drop=True)
    final_shape = df_clean.shape
    if initial_shape != final_shape:
        print(f"Dropped {initial_shape[0] - final_shape[0]} rows containing missing values.")
    else:
        print("No missing values found.")
    return df_clean

def split_data(df, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1."

    total = len(df)
    train_end = int(total * train_frac)
    val_end = train_end + int(total * val_frac)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    print(f"Data split into:")
    print(f" - Training set: {train_df.shape[0]} rows")
    print(f" - Validation set: {val_df.shape[0]} rows")
    print(f" - Test set: {test_df.shape[0]} rows")

    return train_df, val_df, test_df

def compute_history_average(train_df, val_df, test_df, target_features, history_length=5):
    combined_train_val = pd.concat([train_df, val_df], ignore_index=True)
    predictions = {feature: [] for feature in target_features}

    print("Generating predictions using the History Average model...")
    for idx in tqdm(range(len(test_df)), desc="Predicting"):
        for feature in target_features:
            start_idx = len(combined_train_val) - history_length
            if start_idx < 0:
                start_idx = 0
            historical_values = combined_train_val[feature].iloc[start_idx:]
            avg = historical_values.mean()
            predictions[feature].append(avg)
        combined_train_val = pd.concat([combined_train_val, test_df.iloc[[idx]]], ignore_index=True)

    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def calculate_mape(y_true, y_pred, threshold=1e-5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > threshold
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    valid_mask = denominator != 0
    if not np.any(valid_mask):
        return np.nan
    smape_vals = 2.0 * np.abs(y_pred[valid_mask] - y_true[valid_mask]) / denominator[valid_mask]
    return np.mean(smape_vals) * 100

def calculate_mdape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_mask = y_true != 0
    if not np.any(valid_mask):
        return np.nan
    ape = np.abs((y_pred[valid_mask] - y_true[valid_mask]) / y_true[valid_mask]) * 100
    return np.median(ape)

def evaluate_predictions(test_df, predictions_df, target_features):
    # We'll compute MSE, RMSE, MAE, MAPE, sMAPE, MdAPE, R2 for each target and for the mean
    metrics_dict = {}

    for feature in target_features:
        actual = test_df[feature].values
        predicted = predictions_df[feature].values
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = calculate_mape(actual, predicted)
        smape = calculate_smape(actual, predicted)
        mdape = calculate_mdape(actual, predicted)
        r2 = r2_score(actual, predicted)

        metrics_dict[feature] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "sMAPE": smape,
            "MdAPE": mdape,
            "R2": r2
        }

    # Compute mean metrics
    mean_metrics = {}
    for metric in ["MSE","RMSE","MAE","MAPE","sMAPE","MdAPE","R2"]:
        vals = [metrics_dict[f][metric] for f in target_features if not pd.isna(metrics_dict[f][metric])]
        mean_val = np.mean(vals) if len(vals) > 0 else np.nan
        mean_metrics[metric] = mean_val
    metrics_dict["mean"] = mean_metrics

    return metrics_dict

def save_predictions(test_df, predictions_df, results_dir, target_features):
    residuals_df = test_df.copy()
    for feature in target_features:
        residuals_df[f"{feature}_pred"] = predictions_df[feature]
        residuals_df[f"{feature}_residual"] = residuals_df[feature] - residuals_df[f"{feature}_pred"]
    
    excel_path = os.path.join(results_dir, 'predictions_with_residuals.xlsx')
    residuals_df.to_excel(excel_path, index=False)
    print(f"Predictions and residuals saved to {excel_path}")

def plot_actual_vs_predicted(test_df, predictions_df, target_features, results_dir):
    for feature in target_features:
        plt.figure(figsize=(15,5))
        plt.plot(test_df[feature].values, label='Actual', marker='o')
        plt.plot(predictions_df[feature].values, label='Predicted (HA)', marker='x')
        plt.title(f'Actual vs Predicted for {feature}')
        plt.xlabel('Time Steps')
        plt.ylabel('Pose Deviation')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{feature}_Actual_vs_Predicted.png"))
        plt.close()
        print(f"Actual vs Predicted plot for {feature} saved.")

def save_metrics_txt(metrics_dict, target_features, results_dir):
    metrics_txt = os.path.join(results_dir, 'HA_metrics.txt')
    with open(metrics_txt, 'w') as f:
        f.write("Test Results:\n")
        for feature in target_features:
            f.write(f"Metrics for {feature}:\n")
            f.write(f"  Mean Squared Error (MSE): {metrics_dict[feature]['MSE']:.6f}\n")
            f.write(f"  Root Mean Squared Error (RMSE): {metrics_dict[feature]['RMSE']:.6f}\n")
            f.write(f"  Mean Absolute Error (MAE): {metrics_dict[feature]['MAE']:.6f}\n")
            f.write(f"  Mean Absolute Percentage Error (MAPE): {metrics_dict[feature]['MAPE']:.6f}%\n")
            f.write(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {metrics_dict[feature]['sMAPE']:.6f}%\n")
            f.write(f"  Median Absolute Percentage Error (MdAPE): {metrics_dict[feature]['MdAPE']:.6f}%\n")
            f.write(f"  R-squared (R²): {metrics_dict[feature]['R2']:.6f}\n\n")

        f.write("Mean Metrics over all target variables:\n")
        f.write(f"  Mean Squared Error (MSE): {metrics_dict['mean']['MSE']:.6f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {metrics_dict['mean']['RMSE']:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {metrics_dict['mean']['MAE']:.6f}\n")
        f.write(f"  Mean Absolute Percentage Error (MAPE): {metrics_dict['mean']['MAPE']:.6f}%\n")
        f.write(f"  Symmetric Mean Absolute Percentage Error (sMAPE): {metrics_dict['mean']['sMAPE']:.6f}%\n")
        f.write(f"  Median Absolute Percentage Error (MdAPE): {metrics_dict['mean']['MdAPE']:.6f}%\n")
        f.write(f"  R-squared (R²): {metrics_dict['mean']['R2']:.6f}\n")

    print(f"Metrics saved to {metrics_txt}")

def main():
    csv_file = 'data/measurements_random_poses_cleaned.csv'  # Replace with your actual CSV file
    results_dir = 'results/HA/HA_results_random_poses'
    history_length = 5
    os.makedirs(results_dir, exist_ok=True)
    
    target_features = ['x_dif', 'y_dif', 'z_dif', 'rx_dif', 'ry_dif', 'rz_dif']
    
    print("Loading data...")
    df = load_data(csv_file)
    df = df[target_features]

    print("Handling missing values...")
    df = handle_missing_values(df)
    
    print("Splitting data into training, validation, and test sets...")
    train_df, val_df, test_df = split_data(df)
    
    predictions_df = compute_history_average(train_df, val_df, test_df, target_features, history_length=history_length)
    aligned_test_df = test_df.iloc[-len(predictions_df):].reset_index(drop=True)

    print("Evaluating model performance...")
    metrics_dict = evaluate_predictions(aligned_test_df, predictions_df, target_features)

    save_metrics_txt(metrics_dict, target_features, results_dir)
    save_predictions(aligned_test_df, predictions_df, results_dir, target_features)

    print("Plotting Actual vs Predicted for each feature...")
    plot_actual_vs_predicted(aligned_test_df, predictions_df, target_features, results_dir)
    
    print("History Average (HA) modeling completed successfully.")

if __name__ == "__main__":
    main()
