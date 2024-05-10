from PyPDF2 import PdfReader
from flask import Flask, request, send_file
from resume_redactor import extract_phone_numbers, extract_email_addresses, redact_pdf

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # Check if a PDF file was sent in the request
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    # Check if the file has a PDF extension
    if file.filename.split('.')[-1].lower() != 'pdf':
        return 'File must be a PDF', 400

    filename_input = '/tmp/input_file.pdf'
    filename_output = '/tmp/output_file.pdf'

    with open(filename_input, 'wb') as f:
        f.write(file.read())

    print("Processed PDF saved successfully.")

    # Process the PDF file
    process_pdf_file(file, filename_input, filename_output)

    return send_file(
        filename_output,
        mimetype='application/pdf',
        as_attachment=True
    )


def process_pdf_file(pdf_file, input_filename, output_filename):
    # Your PDF processing code goes here
    # For example, you can use libraries like PyPDF2 or pdfplumber
    # to extract text, manipulate pages, etc.
    # This is just a placeholder function.
    processed_pdf_data = pdf_file.read()  # Placeholder processing, just returning the file as is

    document = PdfReader(input_filename)
    text = document.pages[0].extract_text()
    print(text.title())
    phone_numbers = extract_phone_numbers(text.title())
    print(phone_numbers)
    emails = extract_email_addresses(text.title())
    print(emails)

    list_of_regex_words = phone_numbers + emails
    print(list_of_regex_words)
    redact_pdf(input_file=input_filename, regex_list=list_of_regex_words, manual_words=[], output_file=output_filename)

    return processed_pdf_data


if __name__ == '__main__':
    app.run()
