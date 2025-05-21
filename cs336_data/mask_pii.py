import re
from extract_text import extract_text

email_replacement = "|||EMAIL_ADDRESS|||"
phone_number_replacement = "|||PHONE_NUMBER|||"
ip_address_replacement = "|||IP_ADDRESS|||"

n_texts_find = 20
n_texts = 100

"""
2.4 Personal identifiable information 
- masks out emails
- takes a string as input, and replaces all instances of email addresses with the string "|||EMAIL_ADDRESS|||
"""
def mask_emails(input):
    start_count = input.count(email_replacement)
    email_regex = r"\b[\w.-]+@[\w.-]+\.\w+\b" 
    emails_masked = re.sub(email_regex, email_replacement, input)
    return emails_masked, emails_masked.count(email_replacement) - start_count

"""
2.4 Personal identifiable information 
- masks out phone numbers
- takes a string as input, and replaces all instances of phone numbers with the string "|||PHONE_NUMBER|||
"""
def mask_phone_numbers(input):
    start_count = input.count(phone_number_replacement)
    phone_number_regex = r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'
    phone_numbers_masked = re.sub(phone_number_regex, phone_number_replacement, input)
    return phone_numbers_masked, phone_numbers_masked.count(phone_number_replacement) - start_count

"""
2.4 Personal identifiable information 
- masks out IP addresses
- takes a string as input, and replaces all instances of IP addresses with the string "|||IP_ADDRESS|||
"""
def mask_ip_addresses(input):
    start_count = input.count(ip_address_replacement)
    ip_address_regex = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    ip_addresses_masked = re.sub(ip_address_regex, ip_address_replacement, input)
    return ip_addresses_masked, ip_addresses_masked.count(ip_address_replacement) - start_count

"""
run 'uv run python cs336_data/mask_pii.py'
(change import to from extract_text import extract_text)
"""
if __name__ == '__main__':
    with open('/Users/sallyzhu/Desktop/cs336/assignment4-data/cs336_data/example_warcs_many.txt', 'r') as file:
        file_content = file.read()

    split_files = file_content.split('WARC-Type: response')
    split_files = split_files[1:-1]
    total_masked = 0 
    for i in range(n_texts):
        raw_text = split_files[i]
        raw_text = raw_text[:raw_text.find('WARC/1.0')]
        
        raw_text = raw_text[raw_text.find('WARC-Identified-Payload-Type'):]
        raw_text = raw_text[raw_text.find('Content-Length') + 20:]

        extracted_text = extract_text(raw_text.encode('utf-8'))

        # Try seeing if anything is masked 
        masked_emails = mask_emails(extracted_text)
        masked_phone = mask_phone_numbers(masked_emails[0])
        masked_ip = mask_ip_addresses(masked_phone[0])

        if masked_emails[1] > 0 or masked_phone[1] > 0 or masked_ip[1] > 0:
            total_masked += 1
            print(extracted_text)
            print(masked_ip)
            print(f'{total_masked}/{i}  | emails {masked_emails[1]}  |  phones {masked_phone[1]}  |  ip {masked_ip[1]}')
        if total_masked >= 20: 
            break 
