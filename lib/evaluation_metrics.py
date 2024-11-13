# metrics.py

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

    # R2 calculation
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score
