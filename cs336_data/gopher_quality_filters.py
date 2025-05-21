import nltk 
from extract_text import extract_text

nltk.download()
n_texts = 20

"""
2.6 Quality Rules 
- takes a string as its only argument; 
- returns a boolean indicating whether the text passes the Gopher quality filters

filters: if the document
- contains less than 50 or more than 100,000 words
- has a mean word length outside the range of 3 to 10 characters
- has more than 30% of lines ending with an ellipsis (“...”)
- contains less than 80% of words with at least one alphabetic character
"""
def gopher_quality_filter(text):
    words = nltk.tokenize.word_tokenize(text)

    # contains less than 50 or more than 100,000 words
    if len(words) < 50 or len(words) > 100000: 
        # print('failed word count')
        return False 

    # has a mean word length outside the range of 3 to 10 characters
    word_lengths = 0
    for word in words: word_lengths += len(word)
    word_lengths /= len(words)
    if word_lengths < 3 or word_lengths > 10: 
        # print('failed word length')
        return False 

    # contains less than 80% of words with at least one alphabetic character
    has_alph = 0 
    for word in words: has_alph += int(word.upper().isupper())
    has_alph /= len(words) 
    if has_alph < 0.8: 
        # print('failed alphabetic character')
        return False 

    # has more than 30% of lines ending with an ellipsis (“...”)
    lines = text.splitlines()
    ends_ellipsis = 0
    for line in lines: 
        ends_ellipsis += int(line.endswith('...'))
    ends_ellipsis /= len(lines) 
    if ends_ellipsis > 0.3: 
        # print('failed ellipsis')
        return False 

    return True 

"""
run 'uv run python cs336_data/gopher_quality_filters.py'
(change import to from extract_text import extract_text)
"""
if __name__ == '__main__':
    with open('/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/example_warcs_many.txt', 'r') as file:
        file_content = file.read()

    split_files = file_content.split('WARC-Type: response')
    split_files = split_files[1:-1]
    total_detected = 0 
    for i in range(n_texts):
        raw_text = split_files[i]
        raw_text = raw_text[:raw_text.find('WARC/1.0')]
        
        raw_text = raw_text[raw_text.find('WARC-Identified-Payload-Type'):]
        raw_text = raw_text[raw_text.find('Content-Length') + 20:]

        extracted_text = extract_text(raw_text.encode('utf-8'))

        # get gopher quality label 
        quality_label = gopher_quality_filter(extracted_text)

        print('----------------------------------------------------------------------------------------')
        print(extracted_text)
        print('----------------------------------------------------------------------------------------')
        print(f'quality {quality_label}')
