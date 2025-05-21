import fasttext
from extract_text import extract_text

lid_model_path = '/Users/sallyzhu/Downloads/lid.176.bin'
lid_model_path = '/data/classifiers/lid.176.bin '
n_texts = 20

"""
2.3 Language Identification
- takes a Unicode string and identifies the main language that is present in this string
- returns a pair, containing an identifier of the language and a score between 0 and 1 representing 
    its confidence in that prediction
"""
def identify_language(unicode_string):
    model = fasttext.load_model(lid_model_path)
    unicode_string = unicode_string.replace('\n', ' ')
    languages, scores = model.predict(unicode_string)
    language = languages[0]
    if language == '__label__zh':
        language = 'zh'
    elif language == '__label__en':
        language = 'en'
    return language, scores[0]

"""
run 'uv run cs336_data/language_identification.py'
(change import to from extract_text import extract_text)
"""
if __name__ == '__main__':
    with open('/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/example_warcs_many.txt', 'r') as file:
        file_content = file.read()

    split_files = file_content.split('WARC-Type: response')
    split_files = split_files[1:-1]
    for i in range(n_texts):
        raw_text = split_files[i]
        # print(raw_text)
        # print(raw_text.find('WARC/1.0'))
        raw_text = raw_text[:raw_text.find('WARC/1.0')]
        # print(raw_text.find('WARC/1.0'))
        
        raw_text = raw_text[raw_text.find('WARC-Identified-Payload-Type'):]
        raw_text = raw_text[raw_text.find('Content-Length') + 20:]
        # print(raw_text)

        extracted_text = extract_text(raw_text.encode('utf-8'))
        print(i, identify_language(extracted_text))