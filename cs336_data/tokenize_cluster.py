import multiprocessing
import pathlib
import os 

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# input_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/CC/filtered/CC-MAIN-20250430220529-20250501010529-00961.txt'
# output_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/CC/tokenized/test.bin'

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_line(line):
    return tokenizer.encode(line) #+ [tokenizer.eos_token_id] # there already is <|endoftext|> token 

def tokenize_single_file(input_path: str, output_dir_path: str):

    CC_filename = str(pathlib.Path(input_path).name)
    CC_filename = CC_filename[:CC_filename.find('.')]
    output_file_path = os.path.join(output_dir_path, f'{CC_filename}.txt')

    with open(input_path) as f:
        lines = f.readlines()

    results = []

    for line in lines: 
        results.append(tokenize_line(line))

    # Flatten the list of ids and convert to numpy array
    all_ids = [token_id for sublist in results for token_id in sublist]
    print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_file_path)

    return output_file_path

# if __name__ == '__main__':

#     with open(input_path) as f:
#         lines = f.readlines()

#     results = []

#     for line in lines: 
#         results.append(tokenize_line(line))

#     # Flatten the list of ids and convert to numpy array
#     all_ids = [token_id for sublist in results for token_id in sublist]
#     print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
#     ids_array = np.array(all_ids, dtype=np.uint16)
#     ids_array.tofile(output_path)

#     print(all_ids[-30:])