# Deploying ONNX Models with Zappa on AWS Lambda

This repository contains the steps and configurations required to deploy machine learning models (converted to ONNX format) on AWS Lambda using Zappa with Python 3.11.

---

## Prerequisites

* Python **3.11** installed locally
* AWS CLI configured with an IAM role that has permissions to create Lambda functions and related resources (in this project, Utkarsh’s IAM role was used)
* Docker installed for building layers
* Zappa installed in the Python 3.11 environment

---

## Workflow Overview

1. **Install Zappa (Python 3.11 required)**

   ```bash
   pip install zappa
   ```

2. **Configure AWS profile with IAM role**
   Ensure the IAM role used has sufficient permissions for Lambda, API Gateway, CloudFormation, and IAM.

3. **Convert scikit-learn models to ONNX**

   * Models trained in scikit-learn must be converted to ONNX format.
   * A detailed Jupyter notebook with conversion steps is committed in this repo (`convert_to_onnx.ipynb`).

4. **Create Lambda layers for dependencies**
   Some libraries (numpy, scipy, sklearn, onnxruntime) are too large for direct deployment and must be installed as Lambda layers.

---

## Lambda Layers

The following pre-created layers are used in this project:

```json
[
  "arn:aws:lambda:ap-northeast-1:501843401636:layer:numpy-py311:2",
  "arn:aws:lambda:ap-northeast-1:501843401636:layer:scipy-py311:1",
  "arn:aws:lambda:ap-northeast-1:501843401636:layer:sklearn-py311:1",
  "arn:aws:lambda:ap-northeast-1:501843401636:layer:onnxruntime-py311:1"
]
```

These ARNs should be added to your `zappa_settings.json` under the `layers` key.

---

### Creating a Numpy Layer (Example)

```bash
sudo rm -rf ~/lambda-layer-np && mkdir -p ~/lambda-layer-np/python
sudo docker run --rm -v ~/lambda-layer-np:/var/task public.ecr.aws/sam/build-python3.11 \
  /bin/sh -lc 'pip install --no-cache-dir -t /var/task/python numpy==1.26.4 && \
               find /var/task/python -type d -regex ".*/\(tests\|test\|benchmarks\|__pycache__\)" -prune -exec rm -rf {} + && \
               find /var/task/python -name "*.pyc" -delete && \
               find /var/task/python -name "*.so" -exec strip --strip-unneeded {} + || true'

cd ~/lambda-layer-np && zip -r9 layer_onlynumpy311.zip python
aws lambda publish-layer-version --layer-name numpy-py311 --region ap-northeast-1 \
  --compatible-runtimes python3.11 --zip-file fileb://layer_onlynumpy311.zip
```

---

### Creating an ONNX Runtime Layer (Example)

```bash
sudo docker run --rm -v ~/layer-ort:/var/task public.ecr.aws/sam/build-python3.11 \
  /bin/sh -lc '
    python -m pip install --upgrade pip &&
    pip install -t /var/task/python onnxruntime==1.16.0 &&
    find /var/task/python -type d -regex ".*/\(tests\|test\|benchmarks\|__pycache__\)" -prune -exec rm -rf {} + &&
    find /var/task/python -name "*.pyc" -delete &&
    (find /var/task/python -name "*.so" -exec strip --strip-unneeded {} + || true)
  '
```

---

## Deployment Notes

* **Size Limits**

  * Lambda layer size must be < **66 MB**.
  * Total Lambda deployment package must be < **250 MB**.
  * Including scikit-learn often pulls in `scipy`, which bloats the size. Consider trimming dependencies.

* **Library Clashes**

  * Conflicts may occur if the same library is bundled both in a Lambda layer and in the Zappa deployment package.
  * To avoid clashes, keep heavy dependencies in layers and exclude them from your `requirements.txt`.

---

## Learnings

1. Keep each layer focused (e.g., numpy-only, onnxruntime-only).
2. Test locally before deploying, as layer + package conflicts are common.
3. Track ARNs for versioned layers in a central place (e.g., this README or `zappa_settings.json`).


