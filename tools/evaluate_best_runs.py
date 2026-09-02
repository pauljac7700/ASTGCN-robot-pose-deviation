"""Standalone: summarise the best checkpoints across training runs.

Reads the saved runs, ranks them by validation loss and writes a comparison. Run
directly; nothing imports it.
"""

import os
import re
import numpy as np
import pandas as pd
from get_folder_structure import print_folder_structure
from typing import List, Dict, Union
from pandas import DataFrame

base_folder: str = "results"
max_depth: int = 3  # define the depth you care about

print(f"Folder structure of: {base_folder}")
print_folder_structure(base_folder, max_depth=max_depth)

excluded_models: List[str] = ["ASTGCN_single", "MLP_single"]

def extract_short_metric_name(metric_str: str) -> str:
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
    return short_name.replace("R²", "R2")

def process_best_run_file(best_run_file_path: str, excluded: List[str]) -> DataFrame:
    """
    Reads a best_runs.txt file and extracts metrics for each model-run.
    Returns a DataFrame with metrics as rows and models as columns.
    """
    parent_dir = os.path.dirname(best_run_file_path)
    model_run_map: Dict[str, str] = {}
    with open(best_run_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            model_part = parts[0].split(":", 1)[1].strip()
            run_part = parts[1].split(":", 1)[1].strip()
            model_run_map[model_part] = run_part

    all_data = []
    for model_name, run_name in model_run_map.items():
        if model_name in excluded:
            continue
        run_path = os.path.join(parent_dir, model_name, run_name)
        if not os.path.isdir(run_path):
            continue

        # Find the metrics file in the run directory
        metrics_file = None
        for fname in os.listdir(run_path):
            if fname.endswith("metrics.txt"):
                metrics_file = os.path.join(run_path, fname)
                break
        if not metrics_file:
            continue

        with open(metrics_file, "r") as mf:
            lines = [l.rstrip() for l in mf.readlines()]

        model_data: Dict[str, Union[float, str]] = {"Model": model_name, "Best Run": run_name}
        current_dimension = None
        for line in lines:
            if line.startswith("Metrics for "):
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
    df = pd.DataFrame(all_data)
    if not df.empty:
        df.set_index("Model", inplace=True)
        df = df.transpose()
    return df

def write_df_to_sheet(df: DataFrame, sheet_name: str, writer: pd.ExcelWriter, model_order: List[str]) -> None:
    """
    Writes the given DataFrame to a sheet in the Excel writer with formatting.
    The DataFrame is assumed to have metrics as rows and models as columns.
    """
    # Reorder columns if possible
    existing_models = [model for model in model_order if model in df.columns]
    df = df[existing_models]
    df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

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
        'right': 2,
        'left': 2,
    })
    highlight_format = workbook.add_format({
        'bg_color': '#FFFF00',
        'align': 'center',
        'valign': 'middle',
        'border': 1
    })

    # Write header row
    col_headers = ['Metric'] + list(df.columns)
    for col_num, value in enumerate(col_headers):
        worksheet.write(0, col_num, value, header_format)

    max_row, max_col = df.shape  # max_row: number of metric rows, max_col: number of models
    for row_num in range(1, max_row + 1):
        # Write the metric name in the first column
        worksheet.write(row_num, 0, df.index[row_num - 1], right_border_format)
        # Write data cells
        for col_num in range(1, max_col + 1):
            worksheet.write(row_num, col_num, df.iloc[row_num - 1, col_num - 1], center_align_format)

    # Set column widths
    worksheet.set_column(0, 0, 20)
    for col_num in range(1, max_col + 1):
        worksheet.set_column(col_num, col_num, 25)

    # Freeze header row
    worksheet.freeze_panes(1, 0)

    # Conditional formatting: highlight best values per metric row
    metrics = df.index.tolist()
    for i, metric in enumerate(metrics):
        if i == 0:
            continue
        row_num = i + 1
        is_r2 = 'R2' in metric
        row_values = df.iloc[i].values
        best_idx = np.argmax(row_values) if is_r2 else np.argmin(row_values)
        col_num = best_idx + 1  # offset for metric column
        cell_value = df.iloc[i, best_idx]
        worksheet.write(row_num, col_num, cell_value, highlight_format)

    worksheet.add_table(0, 0, max_row, max_col, {
        'columns': [{'header': 'Metric'}] + [{'header': model} for model in df.columns],
        'style': 'Table Style Light 9',
        'banded_rows': False,
    })

def safe_sheet_name(name: str) -> str:
    """
    Returns a safe Excel sheet name (max 31 chars, no forbidden characters).
    """
    forbidden = r'[]:*?/\\'
    for ch in forbidden:
        name = name.replace(ch, "_")
    return name[:31]

# Find all files ending with 'best_runs.txt' under the base folder.
best_run_files: List[str] = []
for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith("best_runs.txt"):
            best_run_files.append(os.path.join(root, file))

# Define desired model order for columns.
model_order: List[str] = [
    "HA", "VARX", "MLP_multi", "ConvLSTM", "TGCN", "ASTGCN_multi",
    "ASTGCN_multi_no_attention", "ASTGCN_multi_no_spatial", "ASTGCN_multi_no_temporal"
]

save_excel_path = os.path.join(base_folder, "all_models_metrics_compared.xlsx")
with pd.ExcelWriter(save_excel_path, engine="xlsxwriter") as writer:
    for best_run_file in best_run_files:
        df = process_best_run_file(best_run_file, excluded_models)
        # Skip if no data was extracted.
        if df.empty:
            continue
        # Generate a sheet name from the relative path of the best_run file.
        rel_dir = os.path.relpath(os.path.dirname(best_run_file), base_folder)
        sheet_name = safe_sheet_name(rel_dir.replace(os.sep, "_") or "root")
        write_df_to_sheet(df, sheet_name, writer, model_order)

print(f"Aggregated metrics saved and formatted in {save_excel_path}")
