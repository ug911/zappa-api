# Embedding-Based Model Training and ONNX Export

This repository contains scripts to train and evaluate binary classifiers on embedding data (e.g., resume and position embeddings), select the best model, and export it for deployment in both **Joblib** and **ONNX** formats.  

The workflow is designed for portability: scikit-learn pipelines are serialized with Joblib for Python environments, while ONNX enables deployment in serverless environments (e.g., AWS Lambda with `onnxruntime`).

---

## 📦 Requirements

One-time installs:

```bash
pip install numpy pandas scikit-learn joblib xgboost
pip install skl2onnx onnx onnxruntime
````

---

## 📂 Data Format

The script expects a JSON file named trainningDataFinal.json.
This file must be downloaded from the tj-zappa-serverless S3 bucket before running the script:

```bash
aws s3 cp s3://tj-zappa-serverless/trainningDataFinal.json .
```

The script expects a JSON file (`trainningDataFinal.json`) structured as a list of rows with the following fields:

* `resumeEmbedding` → array or stringified list of floats
* `positionEmbedding` → array or stringified list of floats
* `status` → binary target label (0 or 1)

Example:

```json
[
  {
    "resumeEmbedding": [0.12, 0.98, -0.33, ...],
    "positionEmbedding": [0.44, -0.21, 0.10, ...],
    "status": 1
  },
  ...
]
```

---

## 🚀 Usage

Run the script directly:

```bash
python train_and_export.py
```

Steps performed:

1. **Load Data** – Reads JSON and builds a feature matrix by concatenating resume + position embeddings.
2. **Preprocess & Split** – Converts targets to integers and performs an 80/20 stratified split.
3. **Train Models** – Fits multiple candidate classifiers (e.g., Logistic Regression, Random Forest).
4. **Evaluate Models** – Computes accuracy, precision, recall, F1, and ROC-AUC on the holdout set.
5. **Select Best Model** – Picks the top performer (by ROC-AUC).
6. **Refit on Full Dataset** – Trains the chosen model on all samples for final export.
7. **Export Artifacts**:

   * `.joblib` → for Python environments
   * `.onnx` → for cross-platform inference
   * `manifest.json` → metadata (model name, timestamp, metrics, paths)

All artifacts are saved in the `artifacts/` folder.

---

## 📊 Outputs

Example output directory:

```
artifacts/
  ├── RandomForest_20250908-123456.joblib
  ├── RandomForest_20250908-123456.onnx
  └── RandomForest_20250908-123456_manifest.json
```

The manifest contains:

```json
{
  "best_model_name": "RandomForest",
  "created_at": "20250908-123456",
  "metrics_on_holdout": {
    "model": "RandomForest",
    "accuracy": 0.83,
    "precision": 0.81,
    "recall": 0.79,
    "f1": 0.80,
    "roc_auc": 0.88
  },
  "paths": {
    "joblib": "artifacts/RandomForest_20250908-123456.joblib",
    "onnx": "artifacts/RandomForest_20250908-123456.onnx"
  },
  "n_features": 768
}
```

---

## ✅ ONNX Sanity Check

After export, the script runs a quick `onnxruntime` inference on a small sample batch:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("artifacts/RandomForest_20250908-123456.onnx")
onnx_input_name = sess.get_inputs()[0].name
preds = sess.run(None, {onnx_input_name: np.random.rand(5, n_features).astype(np.float32)})
```

---

## 🌐 Deployment Notes

* **AWS Lambda**: Use Python 3.11 runtime and add Lambda layers for `numpy` and `onnxruntime`.
* **Scaling**: XGBoost and CatBoost models can be added with proper imbalance handling (`scale_pos_weight`, `class_weight`).
* **Traceability**: The manifest ensures reproducibility and links metrics with model artifacts.

---

## 🔧 Extending

* Add more classifiers to the `models` dict.
* Tune hyperparameters or replace with GridSearchCV/Optuna.
* Replace embeddings with any numeric features.
* Integrate with MLOps pipelines for automated retraining and deployment.


