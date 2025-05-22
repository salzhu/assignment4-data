import os
import pathlib
import re 
import fasttext
from fastwarc.stream_io import *
from fastwarc.warc import ArchiveIterator, WarcRecordType

from tqdm import tqdm

from harmful_content import detect_nsfw_content, detect_hate_speech
from language_identification import identify_language
from gopher_quality_filters import gopher_quality_filter

output_directory_path = "/data/c-salzhu/filteredCC/"
CC_wets_path = '/data/CC/'

paloma_model_path = '/home/c-salzhu/paloma_classifier.bin'
paloma_classifier = fasttext.load_model(paloma_model_path)

def cleanup(text):
    text = re.sub(r'\s+', ' ', text)
    return text.replace('\n', ' ').strip()

def passes_filters(text):
    # check language --> 0.5 english score 
    language, score = identify_language(text)
    if language != 'en' or score < 0.5: 
        return False
    
    # paloma classifier 
    quality, score = paloma_classifier.predict(text)
    if quality[0] != '__label__paloma': 
        return False 

    # gopher quality classifier 
    gopher_quality = gopher_quality_filter(text)
    if gopher_quality == False: 
        return False 

    # harmful content 
    if detect_nsfw_content(text)[0] == 'nsfw': 
        return False
    if detect_hate_speech(text)[0] == 'toxic': 
        return False

    return True 

def process_single_wet_file(input_path: str, output_dir_path: str):
    # TODO: read input path, process the input, and write the output to output_path
    
    wet_filename = str(pathlib.Path(input_path).name)
    wet_filename = wet_filename[:wet_filename.find('.')]
    output_file_path = os.path.join(output_dir_path, f'{wet_filename}.txt')
    with open(output_file_path, 'a', encoding='utf-8') as file:
        count = 0 
        stream = GZipStream(open(input_path, 'rb'))
        for record in tqdm(ArchiveIterator(stream, record_types=WarcRecordType.conversion)):
            text = record.reader.read().decode("utf-8")
            temp = cleanup(text)
            if passes_filters(temp):
                file.write(f"{text}<|endoftext|>")
                print(count, flush=True)
                count += 1

    return output_file_path