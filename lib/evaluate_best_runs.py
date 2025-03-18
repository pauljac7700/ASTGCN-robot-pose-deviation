import os
import re
import numpy as np
import pandas as pd

# Define the base folder and excluded models
base_folder = "/Users/paulj/Documents/Double_Degree_Tsinghua/Master_thesis/code/my_code/ASTGCN-2019-pytorch/results"
excluded_models = ["ASTGCN_single", "ASTGCN_new_graph", "MLP_single"]

# Path to the best runs file
best_runs_file_path = "results/best_runs.txt"
model_run_map = {}

# Read the best runs file and populate the model_run_map
with open(best_runs_file_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue  # Skip malformed lines
        model_part = parts[0].split(":", 1)[1].strip()
        run_part = parts[1].split(":", 1)[1].strip()
        model_run_map[model_part] = run_part

def extract_short_metric_name(metric_str):
    """
    Extracts the short metric name from the metric string.
    E.g., "Mean Squared Error (MSE)" -> "MSE"
    If no parentheses are present, returns the original string.
    Also unifies 'R²' to 'R2'.
    """
    match = re.search(r"\(([^)]+)\)", metric_str)
    if match:
        short_name = match.group(1).strip()
    else:
        short_name = metric_str.strip()
    
    # Unify R² to R2
    short_name = short_name.replace("R²", "R2")
    return short_name

all_data = []

# Iterate through each model and its best run to extract metrics
for model_name, run_name in model_run_map.items():
    if model_name in excluded_models:
        continue
    run_path = os.path.join(base_folder, model_name, run_name)
    if not os.path.isdir(run_path):
        continue

    # Find the metrics file
    metrics_file = None
    for f in os.listdir(run_path):
        if f.endswith("metrics.txt"):
            metrics_file = os.path.join(run_path, f)
            break

    if not metrics_file:
        continue

    with open(metrics_file, 'r') as mf:
        lines = [l.rstrip() for l in mf.readlines()]

    model_data = {"Model": model_name, "Best Run": run_name}
    current_dimension = None

    for line in lines:
        if line.startswith("Metrics for "):
            # Example: "Metrics for x_dif:"
            dim = line.split()[-1].replace(":", "")
            current_dimension = dim
        elif "Mean Metrics over all target variables:" in line:
            current_dimension = "mean"
        elif current_dimension is not None and ":" in line:
            metric_part, val_part = line.split(":", 1)
            metric_part = metric_part.strip()
            val_str = val_part.strip().replace("%", "")
            try:
                val = float(val_str)
            except ValueError:
                continue

            short_name = extract_short_metric_name(metric_part)
            key = f"{current_dimension}_{short_name}"
            model_data[key] = val

    all_data.append(model_data)

# Create DataFrame from the collected data
df = pd.DataFrame(all_data)
save_excel_path = "results/all_models_metrics_compared.xlsx"

# Set 'Model' as index and transpose the DataFrame
df.set_index('Model', inplace=True)
df = df.transpose()

# Define the desired order of models
model_order = ["HA", "VARX", "MLP_multi", "ConvLSTM", "TGCN", "ASTGCN_multi", 
               "ASTGCN_multi_no_attention", "ASTGCN_multi_no_spatial", "ASTGCN_multi_no_temporal"]

# Reorder the columns based on the specified model order
# Handle cases where some models might not be present
existing_models = [model for model in model_order if model in df.columns]
df = df[existing_models]

# Initialize the Excel writer with xlsxwriter engine
with pd.ExcelWriter(save_excel_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Direct Model Comparison', startrow=1, header=False)
    workbook = writer.book
    worksheet = writer.sheets['Direct Model Comparison']

    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'middle',
        'fg_color': '#D7E4BC',
        'border': 1
    })

    center_align_format = workbook.add_format({
        'align': 'center',
        'valign': 'middle',
        'border': 1
    })

    right_border_format = workbook.add_format({
        'align': 'center',
        'valign': 'middle',
        'border': 1,
        'right': 2,  # Thicker right border
        'left': 2,  # Thicker left border
    })

    highlight_format = workbook.add_format({
        'bg_color': '#FFFF00',
        'align': 'center',
        'valign': 'middle',
        'border': 1
    })

    # Write the header
    for col_num, value in enumerate(['Metric'] + list(df.columns)):
        worksheet.write(0, col_num, value, header_format)

    # Apply formatting to all cells
    (max_row, max_col) = df.shape

    for row_num in range(1, max_row + 1):  # Include header row (row 0)
        # Apply right border format to the first column ('Metric' column)
        worksheet.write(row_num, 0, df.index[row_num - 1], right_border_format)

        # Apply center alignment to the remaining cells in each row
        for col_num in range(1, max_col + 1):
            worksheet.write(row_num, col_num, df.iloc[row_num - 1, col_num - 1], center_align_format)

    # Set column widths
    worksheet.set_column(0, 0, 20)  # Set wider width for the first column
    for col_num in range(1, max_col + 1):
        worksheet.set_column(col_num, col_num, 25)

    # Freeze the header row
    worksheet.freeze_panes(1, 0)

    # Convert the DataFrame index to a list for iteration
    metrics = df.index.tolist()
    
    # Iterate through each metric to apply conditional formatting
    for i, metric in enumerate(metrics):
        # Skip the second row (Excel row index 2)
        if i == 0:
            continue
        
        row_num = i + 1  # Excel rows are 1-indexed and header is row 1
        # Determine if the metric is an R2 metric
        is_r2 = 'R2' in metric
        
        row_values = df.iloc[i].values
        
        if is_r2:
            # Highlight the cell with the highest value
            best_idx = np.argmax(row_values)
        else:
            # Highlight the cell with the lowest value
            best_idx = np.argmin(row_values)
        
        # Calculate the Excel column (0-indexed)
        col_num = best_idx + 1  # +1 because first column is Metric
        
        # Get the cell value
        cell_value = df.iloc[i, best_idx]
        
        # Write the cell with the highlight format
        worksheet.write(row_num, col_num, cell_value, highlight_format)
    
    # Add table without additional formatting since header is already formatted
    worksheet.add_table(0, 0, max_row, max_col, {
        'columns': [{'header': 'Metric'}] + [{'header': model} for model in df.columns],
        'style': 'Table Style Light 9',  # Choose a light table style
        'banded_rows': False,

    })

print(f"Aggregated metrics saved and formatted in {save_excel_path}")
