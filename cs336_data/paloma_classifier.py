import fasttext
import re 
import os 
from tqdm import tqdm
import numpy as np
import random 
import gzip 
import re 
from fastwarc.stream_io import *
from fastwarc.warc import ArchiveIterator, WarcRecordType

from transformers import AutoTokenizer

from extract_text import extract_text 
from quality_classifier import parse_content

# model_path = '/home/c-salzhu/paloma_classifier.bin'
model_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/paloma_classifier_strict.bin'
# paloma_path = '/data/paloma/tokenized_paloma_c4_100_domains_validation.bin'
paloma_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/tokenized_paloma_c4_100_domains_validation.bin'
# cc_path = '/home/c-salzhu/cc_scrape.txt'
cc_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/cc_scrape_small.txt'
train_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/paloma_classifier_train.train'
# train_path = '/data/c-salzhu/paloma_classifier_train.train'
tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/CC/CC/CC-MAIN-20250430220529-20250501010529-00961.warc.wet.gz'
wiki_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/wiki_english_positives.txt'

def cleanup(text):
    text = re.sub(r'\s+', ' ', text)
    return text.replace('\n', ' ').strip()

if __name__ == '__main__':

    data = np.fromfile(
        paloma_path,
        dtype=np.uint16
    )

    if os.path.exists(train_path):
        os.remove(train_path)

    with open(train_path, 'a', encoding='utf-8') as file:

        texts = tokenizer.decode(data)
        texts = texts.split('<|endoftext|>')
        for text in tqdm(texts): 
            text = cleanup(text)
            file.write(f"__label__paloma {text}\n")

        # texts = np.arange(5000)

        # wiki_texts = []
        # with open(wiki_path, 'r') as wiki_file:
        #     lines = wiki_file.readlines()
        #     for line in lines:
        #         wiki_texts.append(line.strip())
    
        # for text in tqdm(wiki_texts[:2*len(texts)]): 
        #     # print(found, end=' ', flush=True)
        #     # extracted_text = extract_text(raw_text.encode('utf-8'))
        #     text = cleanup(text)
        #     file.write(f"__label__paloma {text}\n")

        count = 0 
        stream = GZipStream(open(input_path, 'rb'))
        for record in tqdm(ArchiveIterator(stream, record_types=WarcRecordType.conversion)):
            text = record.reader.read().decode("utf-8")
            # print(text)
            text = cleanup(text)
            file.write(f"__label__cc {text}\n")
            count += 1
            if count == len(texts): break

        # lowq_texts = parse_content(cc_path)
        # random.shuffle(lowq_texts)
        # for raw_text in tqdm(lowq_texts[:len(texts)]):
        #     extracted_text = extract_text(raw_text.encode('utf-8'))
        #     extracted_text = cleanup(extracted_text)
        #     file.write(f"__label__cc {extracted_text}\n")

    lines = open(train_path).readlines()
    random.shuffle(lines)
    open(train_path, 'w').writelines(lines)

    model = fasttext.train_supervised(input=train_path, lr=0.1, epoch=5)
    model.save_model(model_path)
    print('classifier saved!')