from flask import Flask, request, send_file
from flask_cors import CORS
import boto3
from urllib.parse import urlparse
import requests
import json
import os, joblib
import numpy as np
from pathlib import Path
import onnxruntime as rt

s3 = boto3.client('s3')

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
sess = rt.InferenceSession(str(BASE_DIR / "models" / "model_rf_v1.onnx"),
                           providers=["CPUExecutionProvider"])

def predict_proba(X: np.ndarray):
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    outputs = sess.run(None, {"input": X})
    # names vary by converter; typical: ['output_probability','output_label']
    proba = outputs[0]
    return proba


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "base_dir": BASE_DIR})


