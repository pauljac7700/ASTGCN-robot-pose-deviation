"""Standalone: locate the best checkpoint for each configuration.

Run directly; nothing imports it.
"""

import os
from typing import Dict, List, Optional, Tuple
from get_folder_structure import print_folder_structure

def folder_depth(path: str, base: str) -> int:
    """Return the depth of a folder relative to the base folder.
    The base folder is considered depth 0.
    """
    rel = os.path.relpath(path, base)
    if rel == ".":
        return 0
    return rel.count(os.sep) + 1

def extract_r2_from_file(metrics_file: str) -> Optional[float]:
    """Extract the R2 value from the given metrics file.

    Looks for the line following the header 
    "Mean Metrics over all residual variables:" that contains "R2:" or "R-squared".
    Returns the float value if successfully extracted; otherwise, returns None.
    """
    with open(metrics_file, 'r') as mf:
        lines = mf.readlines()

    capture = False
    for line in lines:
        if "Mean Metrics over all residual variables:" in line:
            capture = True
            continue
        if capture and ("R2:" in line or "R-squared" in line):
            parts = line.split(":")
            if len(parts) == 2:
                val_str = parts[1].strip().replace("%", "")
                try:
                    return float(val_str)
                except ValueError:
                    return None
            break
    return None

def process_model_directory(model_path: str, base_folder: str) -> Tuple[Optional[str], float]:
    """Process a model directory to determine its best run based on the R2 metric.

    Iterates over run subdirectories in `model_path`, collects R2 scores, writes a text file
    listing each run and its R2 score (sorted best first) with a filename based on the folder structure,
    and returns (best_run, best_r2).
    """
    best_r2: float = float('-inf')
    best_run: Optional[str] = None
    run_results: List[Tuple[str, float]] = []

    for run_name in os.listdir(model_path):
        run_path = os.path.join(model_path, run_name)
        if not os.path.isdir(run_path):
            continue

        # Look for a metrics.txt file in the run directory.
        for file_name in os.listdir(run_path):
            if file_name.endswith("metrics.txt"):
                metrics_file = os.path.join(run_path, file_name)
                r2_val = extract_r2_from_file(metrics_file)
                if r2_val is not None:
                    run_results.append((run_name, r2_val))
                    if r2_val > best_r2:
                        best_r2 = r2_val
                        best_run = run_name
                break  # Only process the first metrics file found

    # Sort run_results descending by R2 value (best first)
    run_results.sort(key=lambda x: x[1], reverse=True)

    # Compute a filename prefix based on the folder structure relative to base_folder.
    # This will include the model name.
    relative_path = os.path.relpath(model_path, base_folder)
    filename_prefix = relative_path.replace(os.sep, "_")
    output_txt = os.path.join(model_path, f"{filename_prefix}_run_r2_scores.txt")

    with open(output_txt, "w") as out_f:
        for run, r2 in run_results:
            out_f.write(f"Run: {run}, R2: {r2}\n")

    return best_run, best_r2

def main() -> None:
    """Main function to display folder structure, process models, and save best runs."""
    base_folder: str = "results" #'results_with_different_test_data' or 'results'
    max_depth: int = 3  # define the depth you care about

    print(f"Folder structure of: {base_folder}")
    print_folder_structure(base_folder, max_depth=max_depth)

    excluded_models: List[str] = ["ASTGCN_single", "MLP_single"]

    # Traverse subdirectories at the specified max_depth.
    for root, dirs, files in os.walk(base_folder):
        if folder_depth(root, base_folder) != max_depth:
            continue

        models_best: Dict[str, Tuple[str, float]] = {}

        for model_name in os.listdir(root):
            if model_name in excluded_models:
                continue

            model_path = os.path.join(root, model_name)
            if not os.path.isdir(model_path):
                continue

            best_run, best_r2 = process_model_directory(model_path, base_folder)
            if best_run is not None:
                models_best[model_name] = (best_run, best_r2)

        # Create a filename prefix from the folder structure relative to base_folder.
        relative_folder = os.path.relpath(root, base_folder)
        filename_prefix = relative_folder.replace(os.sep, "_")
        output_txt = os.path.join(root, f"{filename_prefix}_best_runs.txt")

        with open(output_txt, "w") as out_f:
            for model, (run, r2) in models_best.items():
                out_f.write(f"Model: {model}, Best Run: {run}, R2: {r2}\n")

        print(f"Results saved to: {output_txt}")

if __name__ == "__main__":
    main()
