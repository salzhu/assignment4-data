import os
import pathlib
import fasttext
import gzip 
import re 
from fastwarc.stream_io import *
from fastwarc.warc import ArchiveIterator, WarcRecordType

import submitit
from tqdm import tqdm

# from harmful_content import detect_nsfw_content, detect_hate_speech
# from language_identification import identify_language
# from gopher_quality_filters import gopher_quality_filter

# output_directory_path = "/data/c-salzhu/filteredCC/"
# # CC_wets_path = '/data/CC/'
# CC_wets_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/CC/CC/'

# paloma_model_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/paloma_classifier.bin'
# # paloma_model_path = '/home/c-salzhu/paloma_classifier.bin'
# paloma_classifier = fasttext.load_model(paloma_model_path)

# def parse_content_gz(file_path):
#     with gzip.open(file_path, 'rt', encoding='utf-8') as file: #rb?
#         content = file.read()

#     pattern = r'Content-Length: .*?\n\n(.*?)WARC/1\.0' # Content-Length HTTP/1\.1 
#     matches = re.findall(pattern, content, re.DOTALL)

#     return matches

# def cleanup(text):
#     text = re.sub(r'\s+', ' ', text)
#     return text.replace('\n', ' ').strip()

# def passes_filters(text):
#     # check language --> 0.5 english score 
#     language, score = identify_language(text)
#     if language != 'en' or score < 0.5: 
#         # print(f'language {language} {score}', end=' ')
#         return False
    
#     # paloma classifier 
#     quality, score = paloma_classifier.predict(text)
#     print(quality, score, flush=True)
#     if quality[0] != '__label__paloma': 
#         # print(f'paloma {quality}', end=' ', flush=True)
#         # print(text[:100])
#         return False 

#     # gopher quality classifier 
#     gopher_quality = gopher_quality_filter(text)
#     if gopher_quality == False: 
#         # print('gopher', end=' ', flush=True)
#         return False 

#     # harmful content 
#     if detect_nsfw_content(text)[0] == 'nsfw': 
#         # print('nsfw', end=' ', flush=True)
#         return False
#     if detect_hate_speech(text)[0] == 'toxic': 
#         # print('toxic', end=' ', flush=True)
#         return False

#     return True 

# def process_single_wet_file(input_path: str, output_dir_path: str):
#     # TODO: read input path, process the input, and write the output to output_path
#     # texts = parse_content_gz(input_path)
    
#     wet_filename = str(pathlib.Path(input_path).name)
#     wet_filename = wet_filename[:wet_filename.find('.')]
#     output_file_path = os.path.join(output_dir_path, f'{wet_filename}.txt')
#     with open(output_file_path, 'a', encoding='utf-8') as file:
#         count = 0 
#         stream = GZipStream(open(input_path, 'rb'))
#         for record in tqdm(ArchiveIterator(stream, record_types=WarcRecordType.conversion)):
#             text = record.reader.read().decode("utf-8")
#             # print(text)
#         # for text in tqdm(texts): 
#             temp = cleanup(text)
#             if passes_filters(temp):
#                 file.write(f"{text}<|endoftext|>")
#                 print(count, flush=True)
#                 count += 1

#     return output_file_path

if __name__ == '__main__':
    wet_filepath = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/CC/CC/CC-MAIN-20250430220529-20250501010529-00961.warc.wet.gz'
    stream = GZipStream(open(wet_filepath, 'rb'))
    count = 0 
    for record in tqdm(ArchiveIterator(stream, record_types=WarcRecordType.conversion)):
        text = record.reader.read().decode("utf-8")
        print(text)
        print('-----------------------------------')
        count += 1 
        if count == 10: break 