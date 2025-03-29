import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.varmax import VARMAX

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def handle_missing_values(df):
    if df.isnull().values.any():
        df = df.dropna()
        print("Dropped rows with missing values.")
    else:
        print("No missing values found.")
    return df

def split_data(df, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    total = len(df)
    train_end = int(total * train_frac)
    val_end = train_end + int(total * val_frac)
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    print(f"Training set size: {train_df.shape[0]}")
    print(f"Validation set size: {val_df.shape[0]}")
    print(f"Test set size: {test_df.shape[0]}")
    
    return train_df, val_df, test_df

def scale_data(train_df, val_df, test_df, exog_features, target_features, scalers_path='scalers_VARX.joblib'):
    exog_scalers = {}
    target_scalers = {}
    
    for feature in exog_features:
        scaler = StandardScaler()
        train_df[feature] = scaler.fit_transform(train_df[[feature]])
        val_df[feature] = scaler.transform(val_df[[feature]])
        test_df[feature] = scaler.transform(test_df[[feature]])
        exog_scalers[feature] = scaler
    
    for feature in target_features:
        scaler = StandardScaler()
        train_df[feature] = scaler.fit_transform(train_df[[feature]])
        val_df[feature] = scaler.transform(val_df[[feature]])
        test_df[feature] = scaler.transform(test_df[[feature]])
        target_scalers[feature] = scaler
    
    scalers = {'exog_scalers': exog_scalers, 'target_scalers': target_scalers}
    joblib.dump(scalers, scalers_path)
    print(f"Scalers saved to {scalers_path}")
    
    return train_df, val_df, test_df, scalers

def fit_varmax_model(endog, exog, max_lag=2):
    try:
        model = VARMAX(endog, exog=exog, order=(max_lag, 0))
        fitted_model = model.fit(disp=False, maxiter=1000, method='bfgs')
        print(f"Fitted VARMAX model with lag order: {max_lag}")
        return fitted_model
    except Exception as e:
        print(f"Failed to fit VARMAX model: {e}")
        return None

def inverse_transform(df, scalers, target_features):
    df_inv = df.copy()
    for feature in target_features:
        scaler = scalers['target_scalers'][feature]
        df_inv[feature] = scaler.inverse_transform(df[[feature]])
    return df_inv

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

def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = calculate_mape(actual, predicted)
    smape = calculate_smape(actual, predicted)
    mdape = calculate_mdape(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mae, mape, smape, mdape, r2

def plot_predictions(actual, predicted, feature, results_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='x')
    plt.title(f'Actual vs Predicted: {feature}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{feature}_actual_vs_predicted.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved: {plot_path}")

def save_metrics_txt(metrics_dict, target_features, results_dir):
    metrics_txt_path = os.path.join(results_dir, 'VARX_metrics.txt')
    with open(metrics_txt_path, 'w') as f:
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
    print(f"Metrics saved to {metrics_txt_path}")

def main():
    # Configuration
    csv_file = 'data/measurements_random_poses_cleaned.csv'
    scalers_path = 'data/scalers_VARX.joblib'
    how_much_data = 500
    results_dir = f'results/VARX/VARX_results_for_this_specific_dataset_with_{how_much_data}'
    os.makedirs(results_dir, exist_ok=True)
    
    target_features = ['x_dif', 'y_dif', 'z_dif', 'rx_dif', 'ry_dif', 'rz_dif']
    exog_features = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                     'x_set', 'y_set', 'z_set', 'rx_set', 'ry_set', 'rz_set']
    
    print("Loading data...")
    df = load_data(csv_file)
    # Optionally restrict data
    df = df.loc[0:how_much_data]
    print("Handling missing values...")
    df = handle_missing_values(df)
    print("Splitting data...")
    train_df, val_df, test_df = split_data(df)
    print("Scaling data...")
    train_df, val_df, test_df, scalers = scale_data(train_df, val_df, test_df, exog_features, target_features, scalers_path=scalers_path)
    
    combined_train_df = pd.concat([train_df, val_df], ignore_index=True)
    endog_train = combined_train_df[target_features]
    exog_train = combined_train_df[exog_features]
    
    print("Fitting VARMAX model...")
    varmax_model = fit_varmax_model(endog_train, exog_train, max_lag=5)
    if varmax_model is None:
        print("Model fitting failed. Exiting.")
        return
    
    if len(test_df) < 1:
        print("Test set is too small. Exiting.")
        return
    
    test_exog = test_df[exog_features].copy()
    predicted_list = []
    actual_list = []
    endog_names = varmax_model.model.endog_names
    
    # One-step-ahead forecasting
    for i in range(len(test_exog)):
        exog_one_step = test_exog.iloc[[i]]
        forecast_df = varmax_model.forecast(steps=1, exog=exog_one_step)
        forecast_df.columns = endog_names
        
        forecast_inverse = inverse_transform(forecast_df, scalers, target_features)
        actual_scaled = test_df[target_features].iloc[[i]].reset_index(drop=True)
        actual_inverse = inverse_transform(actual_scaled, scalers, target_features)
        
        predicted_list.append(forecast_inverse.iloc[0])
        actual_list.append(actual_inverse.iloc[0])
    
    predicted_all = pd.DataFrame(predicted_list, columns=target_features)
    actual_all = pd.DataFrame(actual_list, columns=target_features)

    # Evaluate metrics for each target
    metrics_dict = {}
    for feature in target_features:
        mse, rmse, mae, mape, smape, mdape, r2 = evaluate_model(actual_all[feature], predicted_all[feature])
        metrics_dict[feature] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "sMAPE": smape,
            "MdAPE": mdape,
            "R2": r2
        }
        # Plot
        plot_predictions(actual_all[feature].values, predicted_all[feature].values, feature, results_dir)

    # Compute mean metrics
    mean_metrics = {}
    for metric in ["MSE","RMSE","MAE","MAPE","sMAPE","MdAPE","R2"]:
        vals = [metrics_dict[f][metric] for f in target_features if not pd.isna(metrics_dict[f][metric])]
        mean_val = np.mean(vals) if len(vals) > 0 else np.nan
        mean_metrics[metric] = mean_val
    metrics_dict["mean"] = mean_metrics

    # Save metrics
    save_metrics_txt(metrics_dict, target_features, results_dir)

    print("VARX modeling completed successfully.")

if __name__ == "__main__":
    main()
