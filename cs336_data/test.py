import re

def parse_content(file_path):
    with open(file_path, 'r', encoding='iso-8859-1') as file: # , encoding='utf-8'
        content = file.read()

    pattern = r'HTTP/1\.1 .*?\n\n(.*?)WARC/1\.0'
    matches = re.findall(pattern, content, re.DOTALL)

    return matches

# Usage
file_path = '/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/data/wiki_sample_100.txt'
parsed_content = parse_content(file_path)
print('parsed')
print(parsed_content[0])
print(len(parsed_content))
# for i, content in enumerate(parsed_content, 1):
#     print(f"Match {i}:")
#     print(content.strip())
#     print("-------------------")