import os 
import submitit
import pathlib
from tqdm import tqdm

from cs336_data.tokenize_cluster import tokenize_single_file

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default="/data/c-salzhu/tokenizedCC_strict_0522_4/")
parser.add_argument('--CC_filtered', type=str, default='/data/c-salzhu/filteredCC_strict_0522_5/')
args = parser.parse_args()

CC_filtered_path = args.CC_filtered
output_directory_path = args.output 

if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

# Set up the submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs")
max_simultaneous_jobs = 16
filteredCC_filepaths = []
for root, _, files in os.walk(CC_filtered_path):
    for file in files:
        file_path = os.path.join(root, file)
        filteredCC_filepaths.append(file_path)
print(len(filteredCC_filepaths))

# Configure parameters of each job launched by submitit
executor.update_parameters(
    slurm_array_parallelism=max_simultaneous_jobs,
    timeout_min=15,
    mem_gb=2,
    cpus_per_task=1,
    slurm_account="student",
    slurm_partition="a4-cpu",
    slurm_qos="a4-cpu-qos",
)
futures = []
# Use exector.batch() context manager to group all of the jobs in a Slurm array
with executor.batch():
    for filteredCC_filepath in filteredCC_filepaths:
        # For each WARC filepath, submit a job to the executor and get a future back
        wet_filename = str(pathlib.Path(filteredCC_filepath).name)
        future = executor.submit(
            tokenize_single_file,
            filteredCC_filepath,
            output_directory_path
        )
        # Store the futures
        futures.append(future)

# Use tqdm to display progress
for future in tqdm(
    submitit.helpers.as_completed(futures),
    total=len(filteredCC_filepaths),
    ):
    output_file = future.result()
    print(f"Output file written: {output_file}")
