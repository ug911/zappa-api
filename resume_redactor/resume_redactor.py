import fitz
import re


def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]


def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)


def extract_links(code):
    pattern = r'\b(?:https?://)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    links = re.findall(pattern, code)
    return links


def redact_pdf(input_file, regex_list, output_file):
    doc = fitz.Document(input_file)
    # Loop for regex
    regex_words = []
    # for item in regex_list:
    #     # loop for pages in current document
    #     for page in doc:
    #         # Get text from page separated by new line
    #         redact_data = page.get_text("text").split('\n')
    #
    #         # loop for searching regex within each line and adding word to a list
    #         for line in redact_data:
    #             if re.search(item, line, re.IGNORECASE):
    #                 reg_search = str(re.search(item, line, re.IGNORECASE))
    #                 reg_str_individual = re.findall(r"'(.*?)'", reg_search)[0]
    #                 regex_words.append(reg_str_individual)
    #     # print(redact_data) - look at if it's not stored as text how can it read text in images.
    # # combine added words, added regex to pre-determined lists
    # words = [x for x in manual_words + regex_words if x.strip() != '']
    words = regex_list

    # redaction method - search for each word within each page by document
    for page in doc:
        for word in words:
            for instance in page.search_for(word, quads=True):
                areas = page.search_for(word)

                # fill area around the word and colour
                [page.add_redact_annot(area, fill=(0, 0, 0)) for area in areas]
                page.apply_redactions()

    # print file names
    print("\nRedacted: " + str(input_file))

    doc.save(output_file, True)  # manually entered output location
