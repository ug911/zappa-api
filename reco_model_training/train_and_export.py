# =============================================================================
# Train binary classifiers on embedding features and export the best model
# to both .joblib and ONNX. Includes quick ONNX sanity check.
#
# Usage:
#   - Place your training data in 'trainningDataFinal.json'.
#   - Ensure the JSON yields a table with:
#       * 'resumeEmbedding'   -> list/array or stringified list of floats
#       * 'positionEmbedding' -> list/array or stringified list of floats
#       * 'status'            -> binary target {0,1}
#   - Run this script with Python 3.11.
#
# One-time installs:
#   pip install numpy pandas scikit-learn joblib xgboost
#   pip install skl2onnx onnx onnxruntime
#
# Notes:
#   - We compute a simple feature vector by concatenating the two embeddings.
#   - We evaluate models on an 80/20 stratified split using ROC-AUC and other metrics.
#   - The best model (by ROC-AUC) is refit on ALL data, then exported as .joblib and ONNX.
#   - ONNX export here uses skl2onnx (works for sklearn pipelines/estimators).
# =============================================================================

import numpy as np
import pandas as pd
import json
import ast
import joblib, os, time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.base import clone

# Persist models
from joblib import dump

# Optional additional models (uncomment if needed)
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

# ONNX tooling (for sklearn models/pipelines)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_vector(x):
    """
    Normalize an input value into a 1D float numpy array.

    Accepts:
      - list / np.ndarray               -> returns as float array
      - stringified list (e.g. "[1,2]") -> parsed via ast.literal_eval
      - comma-separated string "1,2,3"  -> parsed by splitting on commas

    Returns:
      np.ndarray(dtype=float, shape=(n_features,))

    If input is invalid or empty, returns an empty float array (shape=(0,)).
    """
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)
    if isinstance(x, str):
        try:
            return np.asarray(ast.literal_eval(x), dtype=float)
        except Exception:
            # Fallback for raw comma-separated strings like "1, 2, 3"
            return np.asarray([float(t) for t in x.strip("[]").split(",")], dtype=float)
    # Null/NaN/other types -> empty vector
    return np.asarray([], dtype=float)


def build_feature_matrix(df, resume_col="resumeEmbedding", position_col="positionEmbedding"):
    """
    Build an (n_samples, n_features) feature matrix by concatenating:
      [resumeEmbedding || positionEmbedding]

    Raises:
      ValueError if any row has invalid/missing embeddings.

    Returns:
      X: np.ndarray of shape (n_samples, n_resume + n_position)
    """
    rows, bad_ix = [], []
    for i, (rvec, pvec) in enumerate(zip(df[resume_col], df[position_col])):
        v1 = to_vector(rvec)
        v2 = to_vector(pvec)
        if v1.ndim != 1 or v2.ndim != 1 or v1.size == 0 or v2.size == 0:
            bad_ix.append(i)
            continue
        rows.append(np.concatenate([v1, v2], axis=0))

    if bad_ix:
        # Show first few bad indices for quick debugging
        raise ValueError(
            f"Found {len(bad_ix)} rows with invalid embeddings (indices: {bad_ix[:10]} ...). "
            "Clean them first."
        )

    return np.vstack(rows)


# -----------------------------------------------------------------------------
# Data loading & preprocessing
# -----------------------------------------------------------------------------
data = json.load(open('trainningDataFinal.json', 'r'))
df = pd.DataFrame(data)

# Build features/labels
X = build_feature_matrix(df, "resumeEmbedding", "positionEmbedding")
y = df["status"].astype(int).to_numpy()

print(f"Samples: {X.shape[0]}, feature_dim: {X.shape[1]}")
print(pd.Series(y).value_counts().rename("class_counts"))

# -----------------------------------------------------------------------------
# Train/validation split (80/20 stratified to preserve class balance)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Class imbalance helpers (optional utilities)
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = (neg / pos) if pos > 0 else 1.0  # For XGBoost/etc.
class_weights_list = [neg / (neg + pos), pos / (neg + pos)] if (neg + pos) > 0 else None

# -----------------------------------------------------------------------------
# Model definitions
#   Add/remove models as needed. The leaderboard will pick the best via ROC-AUC.
# -----------------------------------------------------------------------------
models = {}

# 1) Logistic Regression with standardization
models["LogReg"] = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(
        max_iter=2000, solver="saga", class_weight="balanced",
        n_jobs=-1, random_state=42
    )),
])

# 2) Random Forest (class_weight='balanced' to mitigate imbalance)
models["RandomForest"] = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    n_jobs=-1, random_state=42, class_weight="balanced"
)

# Example: enable XGBoost if you want to compare
# models["XGBoost"] = XGBClassifier(
#     n_estimators=400, max_depth=6, learning_rate=0.05,
#     subsample=0.9, colsample_bytree=0.9, random_state=42,
#     scale_pos_weight=scale_pos_weight, tree_method="hist"
# )


# -----------------------------------------------------------------------------
# Train, evaluate, and rank models
# -----------------------------------------------------------------------------
rows = []
pred_store = {}  # Useful if you want to inspect predictions later

for name, model in models.items():
    model.fit(X_train, y_train)

    # Prefer predict_proba for ROC-AUC; fall back to decision_function or hard preds
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Some linear models expose decision_function; rescale to [0,1] for AUC
        from sklearn.preprocessing import MinMaxScaler
        scores = model.decision_function(X_test).reshape(-1, 1)
        y_proba = MinMaxScaler().fit_transform(scores).ravel()
    else:
        # Worst-case fallback: treat hard predictions as pseudo-probabilities
        y_proba = model.predict(X_test).astype(float)

    # Threshold at 0.5 for classification metrics
    y_pred = (y_proba >= 0.5).astype(int)

    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    })
    pred_store[name] = {"y_pred": y_pred, "y_proba": y_proba}

results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
print("\n=== Model leaderboard (80/20 split) ===")
print(results)

# Select the best by ROC-AUC
best_name = results.iloc[0]["model"]
best_model = models[best_name]
print(f"\nBest model: {best_name}")

# -----------------------------------------------------------------------------
# Refit the best model on ALL data and export artifacts
# -----------------------------------------------------------------------------
final_model = clone(best_model).fit(X, y)

os.makedirs("artifacts", exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")

# Save as .joblib (for Python inference)
joblib_path = f"artifacts/{best_name}_{stamp}.joblib"
joblib.dump(final_model, joblib_path)

# Convert to ONNX (for portable, cross-language inference)
# Assumes X is purely numeric features (float32/float64).
n_features = X.shape[1]
initial_types = [("input", FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(
    final_model,
    initial_types=initial_types,
    target_opset=17,  # 17+ recommended for recent runtimes
)

onnx_path = f"artifacts/{best_name}_{stamp}.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# -----------------------------------------------------------------------------
# Quick ONNX sanity check (optional but recommended)
#   Ensures the exported graph runs and produces outputs for a small batch.
# -----------------------------------------------------------------------------
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# Use a small sample from your real feature matrix
X_sample = X[:10].astype(np.float32, copy=False)

# ONNX input names can vary by converter; fetch programmatically
onnx_input_name = sess.get_inputs()[0].name
onnx_outputs = sess.run(None, {onnx_input_name: X_sample})
# 'onnx_outputs' will contain model-dependent arrays (e.g., probabilities, labels)

# -----------------------------------------------------------------------------
# Save a lightweight manifest for traceability and deployment pipelines
# -----------------------------------------------------------------------------
manifest = {
    "best_model_name": best_name,
    "created_at": stamp,
    "metrics_on_holdout": results.loc[0].to_dict(),  # top row corresponds to best model
    "paths": {"joblib": joblib_path, "onnx": onnx_path},
    "n_features": int(n_features)
}
with open(f"artifacts/{best_name}_{stamp}_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

