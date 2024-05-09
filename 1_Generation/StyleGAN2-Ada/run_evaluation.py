# %%
from glob import glob
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# %%
import platform
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()

# %%
experiment_name = "00005-stylegan2_ada_images-mirror-auto2-kimg5000-resumeffhq512"
experiment_path = f"{DATA_PATH}/Models/Stylegan2_Ada/Experiments/" + experiment_name
experiment_path

# %%
snapshots = glob(f"{experiment_path}/*.pkl")
snapshots.sort(key=os.path.getmtime)
snapshots = snapshots[0::5] # Evaluate only every fifth value


# %%
os.chdir(f"{ROOT_PATH}/stylegan2-ada-pytorch/")

# %%
import subprocess

# Assuming `experiment_path` is defined somewhere in your script
results_save_path = f"{experiment_path}/evaluation_metrics.txt"

# Base command setup
base_cmd = "python calc_metrics.py --metrics={metrics} --network={network} --verbose=False --gpus=2"
networks = snapshots  # Assuming `snapshots` is defined as a list of network paths
metrics_list = ["kid50k_full", "is50k", "pr50k3_full"]  # Add more metrics as needed

# Function to execute command
def run_command(metrics, network):
    cmd = base_cmd.format(metrics=metrics, network=network)
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

# Loop through combinations and execute commands
for network in networks:
    for metrics in metrics_list:
        stdout, stderr = run_command(metrics, network)

        # Save each result to the file immediately after it's generated
        with open(results_save_path, "a") as file:  # Open in append mode
            file.write(f"Network: {network}, Metrics: {metrics}\n{stdout}\n")
            if stderr:
                file.write(f"Errors: {stderr}\n")




