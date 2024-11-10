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
