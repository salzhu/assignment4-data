import random
import os 
import mmh3
import string 
import re 
import unicodedata
import shutil

"""
3.2 MinHash + LSH document deduplication
(helper function) normalizes text by: 
- lowercasing, removing punctuation, normalizing whitespaces, and removing accents
- and applying NFD unicode normalization
"""
def normalize_text(text): 
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text)
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([char for char in nfkd_form if not unicodedata.combining(char)])
    text = unicodedata.normalize('NFD', text)
    return text

"""
3.2 MinHash + LSH document deduplication
(helper function) gets all n-grams in a document 
"""
def get_ngrams(filepath, n): 
    ngrams = []
    with open(filepath, "r") as file:
        text = file.read()
    text = normalize_text(text)
    for i in range(len(text) - n): 
        ngrams.append(text[i: i+n])
    return ngrams

"""
3.2 MinHash + LSH document deduplication
(helper function) computes the minhash for a file with k hash functions
"""
def minhash(filepath, n, k): 
    ngrams = get_ngrams(filepath, n)
    minhashes = []
    for i in range(k): 
        temp_ngram_hashed = []
        for ngram in ngrams:
            temp_ngram_hashed.append(mmh3.hash(ngram, i))
        minhashes.append(min(temp_ngram_hashed))
    return minhashes

"""
3.2 MinHash + LSH document deduplication
(helper function) returns whether two filepaths match in LSH over n_bands
"""
def lsh(filepath1, filepath2, n, k, n_bands): 
    minhash1 = minhash(filepath1, n, k)
    minhash2 = minhash(filepath2, n, k)
    for i in range(0, k, k // n_bands): 
        if minhash1[i : i + k // n_bands] == minhash2[i : i + k // n_bands]:
            return True 
    return False 

"""
3.2 MinHash + LSH document deduplication
(helper function) computes the jaccard similarity of two filepaths 
"""
def jaccard_similarity(filepath1, filepath2, n): 
    ngrams1 = get_ngrams(filepath1, n)
    ngrams2 = get_ngrams(filepath2, n)

    union = len(list(set(ngrams1) | set(ngrams2)))
    intersection = len(list(set(ngrams1) & set(ngrams2)))
    return intersection / union

"""
3.2 MinHash + LSH document deduplication
(helper function) returns if two filepaths are candidate duplicates
"""
def is_candidate_duplicate(filepath1, filepath2, n, k, n_bands, jaccard_threshold): 
    return lsh(filepath1, filepath2, n, k, n_bands) and jaccard_similarity(filepath1, filepath2, n) > jaccard_threshold

"""
3.2 MinHash + LSH document deduplication
(helper function) builds all candidate duplicates in a list of files via clustering
"""
def build_candidates(filepaths, n, k, n_bands, jaccard_threshold): 
    clusters = {}
    count = 0 
    for i in range(len(filepaths)):
        filepath1 = filepaths[i]
        for j in range(0, i):
            filepath2 = filepaths[j]
            if not is_candidate_duplicate(filepath1, filepath2, n, k, n_bands, jaccard_threshold): 
                continue 
            count_of_duplicate = clusters[filepath2]
            if filepath1 not in clusters: 
                clusters[filepath1] = count_of_duplicate
                continue 
            if clusters[filepath1] != count_of_duplicate:
                for filepath_sweep in filepaths: 
                    if filepath_sweep in clusters and clusters[filepath_sweep] == clusters[filepath1]:
                        clusters[filepath_sweep] = count_of_duplicate
            clusters[filepath1] = count_of_duplicate

        if filepath1 not in clusters: 
            clusters[filepath1] = count
            count += 1

    return clusters 

"""
3.2 MinHash + LSH document deduplication 
- takes a list of paths to input files and performs fuzzy document deduplication with minhash and LSH
- computes minhash signatures for each document in the provided list of paths
- uses LSH with the provided number of bands to identify candidate duplicates
- compute the true ngram Jaccard similarity between candidate duplicates
    - removes those that exceed a given threshold. 
- normalizes text to improve recall 
"""
def minhash_deduplication(filepaths, n, k, n_bands, jaccard_threshold, output_dir): 
    clusters = build_candidates(filepaths, n, k, n_bands, jaccard_threshold)
    max_count = max(list(clusters.values()))
    for i in range(max_count + 1): 
        filepaths_with_label_i = []
        for filepath in filepaths: 
            if clusters[filepath] == i: 
                filepaths_with_label_i.append(filepath)
        if len(filepaths_with_label_i) == 0: continue 
        keep_filepath = random.choice(filepaths_with_label_i)
        new_filepath = os.path.join(output_dir, os.path.basename(keep_filepath))
        shutil.copy2(keep_filepath, new_filepath)