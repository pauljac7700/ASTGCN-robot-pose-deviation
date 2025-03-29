# evaluation_metrics.py

import numpy as np

def masked_mse(preds, labels, null_val=np.nan):
    '''
    Compute Mean Squared Error with masking.
    If null_val is provided, positions with that value are masked out.
    '''
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(np.float32)

    mask /= np.mean(mask)
    mse = np.square(preds - labels).astype('float32')
    mse = np.nan_to_num(mse * mask)
    return np.mean(mse)

def masked_mape(preds, labels, null_val=np.nan):
    '''
    Compute Mean Absolute Percentage Error with masking.
    If null_val is provided, positions with that value are masked out.
    '''
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    
    # Avoid division by zero by masking out zero labels
    non_zero = labels != 0
    mask = mask & non_zero  # Logical AND on boolean arrays
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    
    mape = np.abs((preds - labels) / labels)
    mape = np.nan_to_num(mape * mask)
    return np.mean(mape) * 100  # Return as percentage



def masked_mae(preds, labels, null_val=np.nan):
    '''
    Compute Mean Absolute Error with masking.
    If null_val is provided, positions with that value are masked out.
    '''
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(np.float32)

    mask /= np.mean(mask)
    mae = np.abs(preds - labels)
    mae = np.nan_to_num(mae * mask)
    return np.mean(mae)

def masked_r2_score(preds, labels, null_val=np.nan):
    '''
    Compute R-squared (coefficient of determination) with masking.
    If null_val is provided, positions with that value are masked out.
    '''
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(np.float32)

    mask /= np.mean(mask)

    # Calculate the mean of the masked labels
    labels_mean = np.sum(labels * mask) / np.sum(mask)

    # Total sum of squares (proportional to variance of the ground truth)
    ss_tot = np.sum(np.square((labels - labels_mean) * mask))

    # Residual sum of squares
    ss_res = np.sum(np.square((labels - preds) * mask))

    # Handle case where ss_tot is zero
    if ss_tot == 0:
        return 0.0  # Undefined R2, return 0.0 or appropriate value

    # R2 calculation
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score

def masked_smape(preds, labels, null_val=np.nan):
    '''
    Compute Symmetric Mean Absolute Percentage Error with masking.
    If null_val is provided, positions with that value are masked out.
    '''
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    
    # Avoid division by zero by ensuring labels and preds are not both zero
    non_zero = (labels != 0) | (preds != 0)
    mask = mask & non_zero  # Logical AND on boolean arrays
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    
    denominator = (np.abs(labels) + np.abs(preds)) / 2.0
    # To avoid division by zero, set denominator to 1 where it's zero (since mask already handles non-zero)
    denominator = np.where(denominator == 0, 1, denominator)
    
    smape = np.abs(preds - labels) / denominator
    smape = np.nan_to_num(smape * mask)
    return np.mean(smape) * 100  # Return as percentage

def masked_mdape(preds, labels, null_val=np.nan):
    '''
    Compute Median Absolute Percentage Error with masking.
    If null_val is provided, positions with that value are masked out.
    '''
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    
    # Avoid division by zero by masking out zero labels
    non_zero = labels != 0
    mask = mask & non_zero  # Logical AND on boolean arrays
    
    # Convert mask to boolean for indexing
    mask = mask.astype(bool)
    
    # Compute absolute percentage errors
    ape = np.abs((preds - labels) / labels)
    
    # Apply mask
    ape_masked = ape[mask]
    
    # Replace NaN and infinite values with zero
    ape_masked = np.nan_to_num(ape_masked)
    
    # Compute median
    mdape = np.median(ape_masked) * 100  # Convert to percentage
    
    return mdape