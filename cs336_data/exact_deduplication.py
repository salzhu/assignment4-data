import hashlib
import os 

algorithm = hashlib.sha256

"""
3.1 Exact line deduplication
helper function for exact_deduplication
- makes one pass through the corpus to count how many occurrences of each line we observe
"""
def build_line_count(file_path, lines):
    with open(file_path, "r") as file:
        for line in file:
            hashed_line = algorithm(line.strip().encode('utf-8')).hexdigest()
            if hashed_line not in lines: 
                lines[hashed_line] = 0 
            lines[hashed_line] += 1

"""
3.1 Exact line deduplication
helper function for exact_deduplication
- second pass: rewrite each document by preserving only its unique lines
"""
def rewrite_file_unique(file_path, lines, output_dir): 
    new_file_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(new_file_path, "w") as file_new:
        with open(file_path, "r") as file:
            for line in file:
                hashed_line = algorithm(line.strip().encode('utf-8')).hexdigest()
                assert lines[hashed_line] > 0
                if lines[hashed_line] == 1: 
                    file_new.write(line.strip() + "\n")

"""
3.1 Exact line deduplication
- takes a list of paths to input files and performs exact line deduplication on them
- first counts the frequency of each line in the corpus, using a hash to reduce memory,
- then rewrites each file by only keeping its unique lines
"""
def exact_deduplication(file_paths, output_dir): 
    # build lines count 
    lines = {}
    for file_path in file_paths: 
        build_line_count(file_path, lines)

    # remove duplicated lines 
    for file_path in file_paths: 
        rewrite_file_unique(file_path, lines, output_dir)