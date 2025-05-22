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

output_directory_path = "/data/c-salzhu/filteredCC_strict_0522_test/"
CC_wets_path = '/data/CC/'

paloma_model_path = '/home/c-salzhu/paloma_classifier_strict.bin'
paloma_classifier = fasttext.load_model(paloma_model_path)

lid_model_path = '/data/classifiers/lid.176.bin'
lid_model = fasttext.load_model(lid_model_path)

nsfw_model_path = '/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin'
nsfw_model = fasttext.load_model(nsfw_model_path)
toxic_model_path = '/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin'
toxic_model = fasttext.load_model(toxic_model_path)

BATCH_SIZE = 512

def cleanup(text):
    text = re.sub(r'\s+', ' ', text)
    return text.replace('\n', ' ').strip()

def process_batch(raw_texts, file):
    texts = [cleanup(text) for text in raw_texts]
    # check language --> 0.5 english score 
    languages, scores = lid_model.predict(texts)
    
    english_raw_texts = []
    count = 0 

    for i in range(len(texts)):
        if languages[i][0] != '__label__en' or scores[i][0] < 0.5: 
            continue 
        english_raw_texts.append(raw_texts[i])
    # print(f'after english {len(english_raw_texts)}', end=' ', flush=True)

    english_texts = [cleanup(text) for text in english_raw_texts]

    qualities, scores = paloma_classifier.predict(english_texts)
    paloma_raw_texts = []
    for i in range(len(english_texts)):
        text = english_texts[i]
        if qualities[i][0] != '__label__paloma': 
            continue 
        paloma_raw_texts.append(english_raw_texts[i])
    # print(f'after paloma {len(paloma_raw_texts)}', end=' ', flush=True)

    paloma_texts = [cleanup(text) for text in paloma_raw_texts]

    nsfw_label, scores = nsfw_model.predict(paloma_texts)
    toxic_label, scores = toxic_model.predict(paloma_texts)
    nsfw_skip = 0 
    toxic_skip = 0
    for i in range(len(paloma_texts)):
        text = paloma_texts[i]
        if nsfw_label[i][0] == '__label__nsfw': 
            # nsfw_skip += 1
            continue 
        if toxic_label[i][0] == '__label__toxic': 
            # toxic_skip += 1
            continue 
        if gopher_quality_filter(text) == False: 
            continue 
        file.write(f"{paloma_raw_texts[i]}<|endoftext|>")
        count += 1
    # print(f'nsfw skip {nsfw_skip} toxic skip {toxic_skip} after gopher {count}', flush=True)
    return count

def process_single_wet_file(input_path: str, output_dir_path: str):
    # TODO: read input path, process the input, and write the output to output_path
    # texts = parse_content_gz(input_path)
    
    wet_filename = str(pathlib.Path(input_path).name)
    wet_filename = wet_filename[:wet_filename.find('.')]
    output_file_path = os.path.join(output_dir_path, f'{wet_filename}.txt')

    buffer_texts: list[str] = []
    total_docs = 0
    good_docs  = 0
    stream = GZipStream(open(input_path, 'rb'))
    with open(output_file_path, 'a', encoding='utf-8') as file:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):          # WET is fine TODO removed func_filter=is_http
            text = record.reader.read().decode("utf-8")

            buffer_texts.append(text)
            total_docs += 1

            # periodic progress
            if total_docs % (BATCH_SIZE * 5) == 0:
                print(f"Processed {total_docs:,} docs "
                    f"| accepted {good_docs:,} ")

            # process full batch
            if len(buffer_texts) >= BATCH_SIZE:
                good_docs += process_batch(buffer_texts, file)
                buffer_texts.clear()

    return output_file_path

# if __name__ == '__main__':
#     wet_filepaths = []
#     for root, _, files in os.walk(CC_wets_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             if file[:2] != 'CC': continue
#             wet_filepaths.append(file_path)
#     print(len(wet_filepaths))
#     for filepath in wet_filepaths: 
#         process_single_wet_file(filepath, output_directory_path)
#         # process_single_wet_file(filepath, '/data/c-salzhu/CC_filtered/')
#         print(f'processed {filepath}')


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
wet_filepaths = wet_filepaths[:3]
print(len(wet_filepaths))

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
