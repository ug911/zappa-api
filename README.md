# Deploying Code on AWS Lambda (using Zappa)

## Zappa (https://github.com/zappa/Zappa)
- Deploying on Zappa is prety simple and straight-forward
- Create a simple Flask app and use the ```zappa init``` and ```zappa deploy dev``` to deplot Zappa
- Everything should be inside a virtual environment only - ```source testzappa/venv/activate``` for activating and ```deactivate``` for deactivating
- Install all dependencies inside the venv only - once installed, they are in ```./lib/python3.8/site-packages/```
- If your Flask app is in file xyz.py then the path to your application in ```zappa_settings.json``` should be ```xyz.app```
- The 502 error is usually due to a missing package
- Use ```zappa status``` to get details of deployment
- Use ```zappa tail dev --since 5m``` to see the logs only for the last 5 mins

## About AWS Lambda
- The total size of the deployment can not be more than 250 MB

## Deploying PDF Redactor code on Zappa
- Installing PyMuPDF was a pain!
- PyMuPDF library is the main (and probably the only) library for successfully redacting PDFs
- Created a Layer in AWS lambda with the PyMUPDF and Fitz libraries in a zip folder (Install in Linux inside a venv and then pick folders from site-packages)
- Need to attach the layer with the Lambda function after each ```zappa update dev``` invocation
- Saving the files in lambda function needs to be done in ```/tmp``` folder only

