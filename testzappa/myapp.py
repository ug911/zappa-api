from PyPDF2 import PdfReader
from flask import Flask, request, send_file
from resume_redactor import extract_phone_numbers, extract_email_addresses, extract_links, redact_pdf
import boto3
import json

OUTPUT_FILE_BUCKET = 'tech-japan-resumes'

s3 = boto3.client('s3')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

def read_pdf_from_s3(bucket_name, file_name):
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    pdf_data = obj['Body'].read()
    return pdf_data

def write_pdf_to_s3(pdf_data, bucket_name, file_name):
    s3.put_object(Body=pdf_data, Bucket=bucket_name, Key=file_name)
    return True

@app.route('/redact_resume_s3', methods=['POST'])
def redact_resume():
    # Get the value of the 'additional_words' parameter from the POST request
    candidate_name = request.form.get('candidate_name')
    input_file_bucket = request.form.get('bucket')
    input_file_path = request.form.get('filepath')

    file = read_pdf_from_s3(input_file_bucket, input_file_path)

    # Check if additional_words parameter is a list
    if candidate_name and not isinstance(candidate_name, str):
        return 'Candidate Name must be a String', 400

    filename_input = '/tmp/input_file.pdf'
    filename_output = '/tmp/output_file.pdf'

    with open(filename_input, 'wb') as f:
        f.write(file)

    print("Processed PDF saved successfully.")

    # Process the PDF file
    process_pdf_file(filename_input, filename_output, candidate_name)

    with open(filename_output, 'rb') as f:
        output_file = f.read()

    write_pdf_to_s3(output_file, OUTPUT_FILE_BUCKET, input_file_path)

    response = app.response_class(
        response=json.dumps({'output_file_bucket': OUTPUT_FILE_BUCKET, 'output_file_name': input_file_name}),
        status=200,
        mimetype='application/json'
    )
    return response


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
