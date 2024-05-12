from PyPDF2 import PdfReader
from flask import Flask, request, send_file
from resume_redactor import extract_phone_numbers, extract_email_addresses, extract_links, redact_pdf

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # Get the value of the 'additional_words' parameter from the POST request
    additional_words = request.form.get('additional_words')

    # Check if a PDF file was sent in the request
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    # Check if the file has a PDF extension
    if file.filename.split('.')[-1].lower() != 'pdf':
        return 'File must be a PDF', 400

    # Check if additional_words parameter is a list
    if additional_words and isinstance(additional_words, list):
        return 'Additional words must be a List', 400

    filename_input = '/tmp/input_file.pdf'
    filename_output = '/tmp/output_file.pdf'

    with open(filename_input, 'wb') as f:
        f.write(file.read())

    print("Processed PDF saved successfully.")

    # Process the PDF file
    process_pdf_file(filename_input, filename_output, additional_words)

    return send_file(
        filename_output,
        mimetype='application/pdf',
        as_attachment=True
    )


def process_pdf_file(input_filename, output_filename, additional_words):
    document = PdfReader(input_filename)
    text = document.pages[0].extract_text()
    phone_numbers = extract_phone_numbers(text)
    emails = extract_email_addresses(text)
    links = extract_links(text)
    list_of_regex_words = phone_numbers + emails + links + additional_words
    words_to_redact = [x for x in list_of_regex_words if x != '']

    print(text)
    print("Phone: {}\nEmails: {}\nLinks: {}\nFinal: {}".format(phone_numbers, emails, links, words_to_redact))

    redact_pdf(input_file=input_filename, regex_list=words_to_redact, output_file=output_filename)

    return True


if __name__ == '__main__':
    app.run()
