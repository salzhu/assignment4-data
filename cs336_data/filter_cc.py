import os
import pathlib
import fasttext
import gzip 
import re 
from fastwarc.stream_io import *
from fastwarc.warc import ArchiveIterator, WarcRecordType

import submitit
from tqdm import tqdm

from harmful_content import detect_nsfw_content, detect_hate_speech
from language_identification import identify_language
from gopher_quality_filters import gopher_quality_filter

output_directory_path = "/data/c-salzhu/filteredCC/"
CC_wets_path = '/data/CC/'
# CC_wets_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/CC/CC/'

paloma_model_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/paloma_classifier.bin'
paloma_model_path = '/home/c-salzhu/paloma_classifier.bin'
paloma_classifier = fasttext.load_model(paloma_model_path)

def cleanup(text):
    text = re.sub(r'\s+', ' ', text)
    return text.replace('\n', ' ').strip()

def parse_content_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as file: #rb?
        content = file.read()

    pattern = r'Content-Length: .*?\n\n(.*?)WARC/1\.0' # Content-Length HTTP/1\.1 
    matches = re.findall(pattern, content, re.DOTALL)

    return matches

def passes_filters(text):
    # check language --> 0.7 english score 
    language, score = identify_language(text)
    if language != 'en' or score < 0.5: 
        # print(f'language {language} {score}', end=' ')
        return False
    
    # paloma classifier 
    quality, score = paloma_classifier.predict(text)
    if quality[0] != '__label__paloma': 
        # print(f'paloma {quality}', end=' ', flush=True)
        # print(text[:100])
        return False 

    # gopher quality classifier 
    gopher_quality = gopher_quality_filter(text)
    if gopher_quality == False: 
        # print('gopher', end=' ', flush=True)
        return False 

    # harmful content 
    if detect_nsfw_content(text)[0] == 'nsfw': 
        # print('nsfw', end=' ', flush=True)
        return False
    if detect_hate_speech(text)[0] == 'toxic': 
        # print('toxic', end=' ', flush=True)
        return False

    return True 

def process_single_wet_file(input_path: str, output_dir_path: str):
    # TODO: read input path, process the input, and write the output to output_path
    # texts = parse_content_gz(input_path)
    
    wet_filename = str(pathlib.Path(input_path).name)
    wet_filename = wet_filename[:wet_filename.find('.')]
    output_file_path = os.path.join(output_dir_path, f'{wet_filename}.txt')
    with open(output_file_path, 'a', encoding='utf-8') as file:
        count = 0 
        stream = GZipStream(open(input_path, 'rb'))
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
            text = record.reader.read().decode("utf-8")
            # print(text)
        # for text in tqdm(texts): 
            temp = cleanup(text)
            if passes_filters(temp):
                file.write(f"{text}<|endoftext|>")
                print(count, flush=True)
                count += 1

    return output_file_path

if __name__ == '__main__':
    wet_filepaths = []
    for root, _, files in os.walk(CC_wets_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file[:2] != 'CC': continue
            wet_filepaths.append(file_path)
    print(len(wet_filepaths))
    for filepath in wet_filepaths: 
        # process_single_wet_file(filepath, '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/CC/filtered')
        process_single_wet_file(filepath, '/data/c-salzhu/CC_filtered/')
        print(f'processed {filepath}')

"""
# Set up the submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs")
max_simultaneous_jobs = 8
# wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]
wet_filepaths = []
for root, _, files in os.walk(CC_wets_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file[:2] != 'CC': continue
        wet_filepaths.append(file_path)

# Configure parameters of each job launched by submitit
executor.update_parameters(
    slurm_array_parallelism=max_simultaneous_jobs,
    timeout_min=15,
    mem_gb=2,
    cpus_per_task=2,
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
            os.path.join(output_directory_path, wet_filename)
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
"""