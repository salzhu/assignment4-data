import os 
import submitit
import pathlib
from tqdm import tqdm

from cs336_data.filter_cc_batch_cluster import process_single_wet_file

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default="/data/c-salzhu/filteredCC_strict_0522/")
parser.add_argument('--CC', type=str, default='/data/CC/')
args = parser.parse_args()

CC_wets_path = args.CC
output_directory_path = args.output 

if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

# Set up the submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs")
max_simultaneous_jobs = 16
# wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]
wet_filepaths = []
for root, _, files in os.walk(CC_wets_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file[:2] != 'CC': continue
        wet_filepaths.append(file_path)
print(len(wet_filepaths))

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
    for wet_filepath in wet_filepaths:
        # For each WARC filepath, submit a job to the executor and get a future back
        wet_filename = str(pathlib.Path(wet_filepath).name)
        future = executor.submit(
            process_single_wet_file,
            wet_filepath,
            output_directory_path
        )
        # Store the futures
        futures.append(future)

# Use tqdm to display progress
for future in tqdm(
    submitit.helpers.as_completed(futures),
    total=len(wet_filepaths),
    ):
    output_file = future.result()
    print(f"Output file written: {output_file}")
