from PyPDF2 import PdfReader
from flask import Flask, request, send_file
from resume_redactor import extract_phone_numbers, extract_email_addresses, extract_links, redact_pdf
import boto3
from urllib.parse import urlparse
import requests
import json

OUTPUT_FILE_BUCKET = 'techjapan-hub-filesXX'
OUTPUT_FILE_BUCKET_TEST = 'tj-zappa-serverless'

s3 = boto3.client('s3')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


def download_pdf_from_url(file_url, local_filepath):
    # Download the file
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(local_filepath, 'wb') as file:
            file.write(response.content)
        success = True
        print("Download successful")
    else:
        success = False
        print("Failed to download file")
    return success


def write_pdf_to_s3(pdf_data, bucket_name, file_name):
    res = s3.put_object(Body=pdf_data, Bucket=bucket_name, Key=file_name)
    success = False
    presigned_url = None
    if res.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:
        success = True
        presigned_url = s3.generate_presigned_url('get_object',
                                                  Params={'Bucket': 'tj-zappa-serverless',
                                                            'Key': 'candidates/xyz/redacted/MD-AFROZ-Resume.pdf'},
                                                  ExpiresIn=3600)
    return success, presigned_url


@app.route('/redact_resume', methods=['POST'])
def redact_resume():
    # Get the value of the 'additional_words' parameter from the POST request
    candidate_name = request.form.get('candidate_name')
    candidate_resume_url = request.form.get('candidate_resume_url')
    debug = request.form.get('debug')

    output_file_bucket = OUTPUT_FILE_BUCKET_TEST if debug else OUTPUT_FILE_BUCKET

    parsed_url = urlparse(candidate_resume_url)
    path = parsed_url.path
    filename = path.split('/')[-1]
    final_s3path = '/'.join(x for x in path.split('/')[:-1] + ['redacted', filename] if x != '')

    local_input_filename = '/tmp/{}'.format(filename)
    local_output_filename = '/tmp/redacted_{}'.format(filename)

    download_success = download_pdf_from_url(candidate_resume_url, local_input_filename)

    # Check if resume file is successfully downloaded
    if not download_success:
        return app.response_class(
            response=json.dumps({'message': 'Unable to download URL'}),
            status=400
        )

    # Check if candidate_name is available and is a string
    if candidate_name and not isinstance(candidate_name, str):
        return app.response_class(
            response=json.dumps({'message': 'Candidate Name must be a String'}),
            status=400
        )

    # Process the PDF file
    process_pdf_file(local_input_filename, local_output_filename, candidate_name)

    with open(local_output_filename, 'rb') as f:
        output_file = f.read()

    write_success, output_file_url = write_pdf_to_s3(output_file, output_file_bucket, final_s3path)

    # Check if redacted resume file is successfully uploaded
    if not write_success:
        return app.response_class(
            response=json.dumps({'message': 'Unable to upload redacted file'}),
            status=400
        )

    response = app.response_class(
        response=json.dumps({'output_file': output_file_url}),
        status=200
    )
    return response


@app.route('/redact_resume_from_file', methods=['POST'])
def redact_resume_from_file():
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
