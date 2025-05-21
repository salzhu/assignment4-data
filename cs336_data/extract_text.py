from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding, bytes_to_str

"""
2.2 HTML to text conversion extract_text
extracts text from a byte string containing raw HTML
"""
def extract_text(bytestr):
    # convert byte string to unicode
    decoded = bytes_to_str(bytestr, detect_encoding(bytestr))
    return extract_plain_text(decoded)

if __name__ == '__main__':
    with open('/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/example_warc.txt', 'r') as file:
        file_content = file.read()
        print(extract_text(file_content.encode('utf-8')))
        