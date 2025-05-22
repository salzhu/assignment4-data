import fasttext
from extract_text import extract_text

nsfw_model_path = '/Users/sallyzhu/Downloads/jigsaw_fasttext_bigrams_nsfw_final.bin'
toxic_model_path = '/Users/sallyzhu/Downloads/jigsaw_fasttext_bigrams_hatespeech_final.bin'
# nsfw_model_path = '/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin'
# toxic_model_path = '/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin'
n_texts = 20
n_total_texts = 100

"""
2.5 Harmful Content / NSFW content
- labels a given string as containing NSFW content or not
- returns a pair containing both the label and a confidence score
"""
def detect_nsfw_content(unicode_string):
    model = fasttext.load_model(nsfw_model_path)
    unicode_string = unicode_string.replace('\n', ' ')
    label, score = model.predict(unicode_string)
    if label[0] == '__label__nsfw': label = 'nsfw'
    if label[0] == '__label__non-nsfw': label = 'non-nsfw'
    return label, score[0]

"""
2.5 Harmful Content / Hate speech
- labels a given string as containing NSFW content or not
- returns a pair containing both the label and a confidence score
"""
def detect_hate_speech(unicode_string):
    model = fasttext.load_model(toxic_model_path)
    unicode_string = unicode_string.replace('\n', ' ')
    label, score = model.predict(unicode_string)
    if label[0] == '__label__toxic': label = 'toxic'
    if label[0] == '__label__non-toxic': label = 'non-toxic'
    return label, score[0]

"""
run 'uv run python cs336_data/harmful_content.py'
(change import to from extract_text import extract_text)
(prints 10 nsfw, 10 hate speech, 10 not harmful content)
"""
if __name__ == '__main__':
    with open('/Users/sallyzhu/Desktop/cs336/assignment4-data/data/cs336_data/example_warcs_many.txt', 'r') as file:
        file_content = file.read()

    split_files = file_content.split('WARC-Type: response')
    split_files = split_files[1:-1]
    total_detected = 0 
    print(len(split_files))
    for i in range(len(split_files)):
        raw_text = split_files[i]
        raw_text = raw_text[:raw_text.find('WARC/1.0')]
        
        raw_text = raw_text[raw_text.find('WARC-Identified-Payload-Type'):]
        raw_text = raw_text[raw_text.find('Content-Length') + 20:]

        extracted_text = extract_text(raw_text.encode('utf-8'))

        # Try seeing if any harmful content is detected 
        nsfw_label, nsfw_score = detect_nsfw_content(extracted_text)
        hate_label, hate_score = detect_hate_speech(extracted_text)

        if (nsfw_label == 'nsfw' or hate_label == 'toxic'):
            total_detected += 1 
            print(extracted_text)
            # print(f'{total_detected}/{i}  | nsfw {nsfw_label} {nsfw_score}  |  hate {hate_label} {hate_score}')
            print(f'{nsfw_label} & {nsfw_score:.4} & {hate_label} & {hate_score:.4}')

        if total_detected >= n_texts: break
