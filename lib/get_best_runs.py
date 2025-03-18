import os
import pandas as pd

base_folder = "/Users/paulj/Documents/Double_Degree_Tsinghua/Master_thesis/code/my_code/ASTGCN-2019-pytorch/results"

# Models to exclude
excluded_models = ["ASTGCN_single", "ASTGCN_new_graph","MLP_single"]

models_best = {}

for model_name in os.listdir(base_folder):
    if model_name in excluded_models:
        continue

    model_path = os.path.join(base_folder, model_name)
    if not os.path.isdir(model_path):
        continue

    best_r2 = float('-inf')
    best_run = None

    for run_name in os.listdir(model_path):
        run_path = os.path.join(model_path, run_name)
        if not os.path.isdir(run_path):
            continue

        # Find metrics file
        for f in os.listdir(run_path):
            if f.endswith("metrics.txt"):
                metrics_file = os.path.join(run_path, f)
                with open(metrics_file, 'r') as mf:
                    lines = mf.readlines()

                capture = False
                for line in lines:
                    if "Mean Metrics over all target variables:" in line:
                        capture = True
                        continue
                    if capture and ("R2:" in line or "R-squared" in line):
                        parts = line.split(":")
                        if len(parts) == 2:
                            val_str = parts[1].strip().replace("%", "")
                            try:
                                r2_val = float(val_str)
                                if r2_val > best_r2:
                                    best_r2 = r2_val
                                    best_run = run_name
                            except ValueError:
                                pass
                        break

    if best_run is not None:
        models_best[model_name] = (best_run, best_r2)

# Save results to a text file in the base folder
output_txt = os.path.join(base_folder, "best_runs.txt")
with open(output_txt, "w") as out_f:
    for model, (run, r2) in models_best.items():
        out_f.write(f"Model: {model}, Best Run: {run}, R2: {r2}\n")

print(f"Results saved to: {output_txt}")

