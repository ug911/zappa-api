from PyPDF2 import PdfReader
from flask import Flask, request, send_file
from resume_redactor import extract_phone_numbers, extract_email_addresses, extract_links, redact_pdf

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/redact_resume', methods=['POST'])
def redact_resume():
    # Get the value of the 'additional_words' parameter from the POST request
    candidate_name = request.form.get('candidate_name')

    # Check if a PDF file was sent in the request
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    # Check if the file has a PDF extension
    if file.filename.split('.')[-1].lower() != 'pdf':
        return 'File must be a PDF', 400

    # Check if additional_words parameter is a list
    if candidate_name and not isinstance(candidate_name, str):
        return 'Candidate Name must be a String', 400

    filename_input = '/tmp/input_file.pdf'
    filename_output = '/tmp/output_file.pdf'

    with open(filename_input, 'wb') as f:
        f.write(file.read())

    print("Processed PDF saved successfully.")

    # Process the PDF file
    process_pdf_file(filename_input, filename_output, candidate_name)

    return send_file(
        filename_output,
        mimetype='application/pdf',
        as_attachment=True
    )


def process_pdf_file(input_filename, output_filename, candidate_name):
    document = PdfReader(input_filename)
    text = document.pages[0].extract_text()
    phone_numbers = extract_phone_numbers(text)
    emails = extract_email_addresses(text)
    links = extract_links(text)
    print("Phone: {}\nEmails: {}\nLinks: {}\nInput Words: {}".format(phone_numbers, emails, links, candidate_name))
    list_of_regex_words = phone_numbers + emails + links + [candidate_name]
    words_to_redact = [x for x in list_of_regex_words if x != '']

    print(text)
    print("Phone: {}\nEmails: {}\nLinks: {}\nFinal: {}".format(phone_numbers, emails, links, words_to_redact))

    redact_pdf(input_file=input_filename, regex_list=words_to_redact, output_file=output_filename)

    return True


if __name__ == '__main__':
    app.run()
