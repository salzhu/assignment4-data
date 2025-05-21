import fasttext
import re 
import os 
from tqdm import tqdm

from extract_text import extract_text 
from mask_pii import mask_emails, mask_phone_numbers, mask_ip_addresses
from harmful_content import detect_nsfw_content, detect_hate_speech
from language_identification import identify_language
from gopher_quality_filters import gopher_quality_filter

wiki_data_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/wiki_sample_100.txt'
wiki_data_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/wiki_english_positives.txt'
cc_data_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/example_warcs_many.txt'
cc_data_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/cc_scrape_small.txt'
model_path = 'quality_classifier.bin'

low_quality_cc_example = '/Users/sallyzhu/Desktop/cs336/assignment4-data/tests/fixtures/low_quality_cc.txt'
high_quality_wiki_example = '/Users/sallyzhu/Desktop/cs336/assignment4-data/tests/fixtures/high_quality_wiki_reference.txt'

def parse_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file: # , encoding='utf-8'
        content = file.read()

    pattern = r'Content-Length: .*?\n\n(.*?)WARC/1\.0' # Content-Length HTTP/1\.1 
    matches = re.findall(pattern, content, re.DOTALL)

    return matches

def mask_and_filter(text): 
     # mask pii 
    text = text.lower()
    text = mask_emails(text)[0]
    text = mask_phone_numbers(text)[0]
    text = mask_ip_addresses(text)[0]

    # if harmful content --> bad 
    if detect_nsfw_content(text)[0] == 'nsfw': return False, text
    if detect_hate_speech(text)[0] == 'toxic': return False, text

    # if text not english --> bad 
    if identify_language(text)[0] != 'en': return False, text

    text = re.sub(r'\s+', ' ', text)

    # if text not filter --> bad 
    return gopher_quality_filter(text), text.replace('\n', ' ').strip()

def mask(text):
    text = text.lower()
    text = mask_emails(text)[0]
    text = mask_phone_numbers(text)[0]
    text = mask_ip_addresses(text)[0]
    text = re.sub(r'\s+', ' ', text)
    return text.replace('\n', ' ').strip()

"""
2.7 Quality Classifier / train
- train a quality classifier that, given text, returns a numeric quality score
"""
def train_quality_classifier(high_quality_path, low_quality_path, n_high, n_low, text_path='quality.train', model_path=model_path):

    if os.path.exists(text_path):
        os.remove(text_path)

    with open(text_path, 'a', encoding='utf-8') as file:

        highq_texts = []
        with open(high_quality_path, 'r') as wiki_file:
            lines = wiki_file.readlines()
            for line in lines:
                highq_texts.append(line.strip())
    
        # highq_texts = parse_content(high_quality_path)
        found = 0
        for raw_text in tqdm(highq_texts): 
            # print(found, end=' ', flush=True)
            extracted_text = extract_text(raw_text.encode('utf-8'))

            # quality, text = mask_and_filter(extracted_text)
            # print(extracted_text[:50])

            if True: # quality == 
                text = mask(extracted_text)
                file.write(f"__label__wiki {text}\n")
                found += 1
            if found >= n_high: 
                break
        
        lowq_texts = parse_content(low_quality_path)
        
        n_low = found
        found = 0
        for raw_text in tqdm(lowq_texts): 
            # print(found, end=' ', flush=True)
            extracted_text = extract_text(raw_text.encode('utf-8'))

            text = mask(extracted_text)

            # if quality == True or quality == False: 
            file.write(f"__label__cc {text}\n")
            found += 1
            if found >= n_low: 
                break

    model = fasttext.train_supervised(input=text_path, lr=0.1, epoch=10)
    model.save_model(model_path)

"""
2.7 Quality Classifier / label 
- labels a page as high or low-quality;
- provides a confidence score in the label
"""
def classify_quality(text):

    quality, text = mask_and_filter(text)
    if quality == False: 
        return 'cc', 1.0

    model = fasttext.load_model(model_path)
    label, prob = model.predict(text)
    if label[0] == '__label__wiki': label = 'wiki'
    if label[0] == '__label__cc': label = 'cc'

    return label, prob[0]

if __name__ == '__main__': 
    train_quality_classifier(wiki_data_path, cc_data_path, 5000, 5000, model_path=model_path)

    with open(low_quality_cc_example) as f:
        low_quality_cc = f.read()
    low_pred = classify_quality(low_quality_cc)
    print(f'cc example: {low_pred}')

    with open(high_quality_wiki_example) as f:
        high_quality_wiki = f.read()
    high_pred = classify_quality(high_quality_wiki)
    print(f'wiki example: {high_pred}')
