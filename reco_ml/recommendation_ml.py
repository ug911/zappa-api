# app.py
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort

# Optional: pull model from S3 if MODEL_S3_URI is set (zappa has boto3 already)
import boto3

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# You can package the model with your code (e.g., models/model_rf_v1.onnx)
# or set MODEL_S3_URI=s3://bucket/path/to/model.onnx to download at cold start.
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", str(BASE_DIR / "models" / "model_rf_v1.onnx"))
MODEL_S3_URI = os.environ.get("MODEL_S3_URI", "").strip()

# Prediction threshold for class 1
PRED_THRESHOLD = float(os.environ.get("PRED_THRESHOLD", "0.5"))

# ONNXRuntime session performance knobs (safe defaults for Lambda)
INTRA_OP = int(os.environ.get("ORT_INTRA_OP", "1"))
INTER_OP = int(os.environ.get("ORT_INTER_OP", "1"))

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _maybe_download_from_s3(local_path: str, s3_uri: str) -> str:
    """
    If s3_uri is provided, download to /tmp/model.onnx (Lambda writeable).
    Otherwise return local_path unchanged.
    """
    if not s3_uri:
        return local_path

    # Lambda can only write to /tmp
    target = "/tmp/model.onnx"
    if os.path.exists(target):
        return target

    if not s3_uri.startswith("s3://"):
        raise ValueError("MODEL_S3_URI must start with s3://")

    s3 = boto3.client("s3")
    # s3://bucket/key...
    _, _, bucket_and_key = s3_uri.partition("s3://")
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError("Invalid MODEL_S3_URI format. Expected s3://bucket/key")

    s3.download_file(bucket, key, target)
    return target


def _ensure_2d_float32(X: np.ndarray, expected_n_features: int = None) -> np.ndarray:
    """
    Ensure input is 2D (batch, n_features) float32.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    if expected_n_features is not None and X.shape[1] != expected_n_features:
        raise ValueError(f"Expected {expected_n_features} features, got {X.shape[1]}")
    return X


def _extract_pos_proba(onnx_outputs: List[Any], positive_class=1) -> np.ndarray:
    """
    Supports skl2onnx classifier outputs:
      - outputs[0] = labels (ints)
      - outputs[1] = probabilities as ZipMap (list/array of dicts) OR ndarray

    Returns float array of shape (N,) with P(class=positive_class).
    """
    if not onnx_outputs:
        raise RuntimeError("ONNX session returned no outputs")

    # Try the 2nd output first (usually probabilities)
    probs = onnx_outputs[1] if len(onnx_outputs) > 1 else onnx_outputs[0]

    # Case A: probs is already an ndarray, e.g. shape (N,2)
    if isinstance(probs, np.ndarray) and np.issubdtype(probs.dtype, np.number):
        # If 2 columns, take column for positive class (assume class order [0,1])
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1].astype(np.float32).ravel()
        # If 1-D or (N,1), flatten
        return probs.astype(np.float32).ravel()

    # Case B: probs is ZipMap -> list/array of dicts: [{0: p0, 1: p1}, ...]
    if isinstance(probs, (list, tuple)) or (isinstance(probs, np.ndarray) and probs.dtype == object):
        out = []
        for row in probs:
            if isinstance(row, dict):
                # keys might be int or str
                if positive_class in row:
                    out.append(float(row[positive_class]))
                elif str(positive_class) in row:
                    out.append(float(row[str(positive_class)]))
                else:
                    # Fallback: pick the highest-prob class
                    out.append(float(max(row.values())))
            else:
                raise RuntimeError(f"Unexpected ZipMap row type: {type(row)}")
        return np.asarray(out, dtype=np.float32)

    # Last resort: scan all outputs for a 2-col ndarray
    for out in onnx_outputs:
        if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] == 2:
            return out[:, 1].astype(np.float32).ravel()

    raise RuntimeError("Could not find probability-like output in ONNX results")


def _concat_if_present(body: Dict[str, Any]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Build a batch X where each row is [candidateEmbedding || positionEmbedding].
    Returns:
      X: np.ndarray of shape (batch_size, n_features), dtype float32
      pair_ids: list of (candidateId, positionId) tuples aligned to X rows

    Supports:
      A) {"positionEmbedding": [...], "candidates": [{"candidateEmbedding": [...]}...]}
      B) {"candidateEmbedding": [...], "positions": [{"positionEmbedding": [...]}...]}
      Legacy:
         - {"features": [[...], [...]]}   -> ids become ("row_i","n/a")
         - {"resumeEmbedding": [...], "positionEmbedding": [...]} -> single row
    """
    pair_ids: List[Tuple[str, str]] = []

    # --- New schema: A) single position, many candidates
    if "positionEmbedding" in body and "candidates" in body:
        pos_id = str(body.get("positionId", ""))  # optional
        p = np.asarray(body["positionEmbedding"], dtype=np.float32)
        if p.ndim != 1:
            raise ValueError("positionEmbedding must be a 1D array")

        candidates = body.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) == 0:
            raise ValueError("'candidates' must be a non-empty list")

        rows = []
        for i, cand in enumerate(candidates):
            cid = str(cand.get("candidateId", f"candidate_{i}"))
            # Accept either "candidateEmbedding" (preferred) or legacy "resumeEmbedding"
            if "candidateEmbedding" in cand:
                c = np.asarray(cand["candidateEmbedding"], dtype=np.float32)
            elif "resumeEmbedding" in cand:  # soft fallback
                c = np.asarray(cand["resumeEmbedding"], dtype=np.float32)
            else:
                raise ValueError(f"Missing candidateEmbedding for candidate index {i}")
            if c.ndim != 1:
                raise ValueError(f"candidateEmbedding at index {i} must be 1D")

            rows.append(np.concatenate([c, p], axis=0))
            pair_ids.append((cid, pos_id))
        X = np.stack(rows, axis=0)
        return X, pair_ids

    # --- New schema: B) single candidate, many positions
    if "candidateEmbedding" in body and "positions" in body:
        cand_id = str(body.get("candidateId", ""))  # optional
        c = np.asarray(body["candidateEmbedding"], dtype=np.float32)
        if c.ndim != 1:
            raise ValueError("candidateEmbedding must be a 1D array")

        positions = body.get("positions", [])
        if not isinstance(positions, list) or len(positions) == 0:
            raise ValueError("'positions' must be a non-empty list")

        rows = []
        for i, pos in enumerate(positions):
            pid = str(pos.get("positionId", f"position_{i}"))
            # Accept "positionEmbedding" (preferred). Also tolerate accidental "positionIdEmbedding".
            emb_key = "positionEmbedding" if "positionEmbedding" in pos else (
                "positionIdEmbedding" if "positionIdEmbedding" in pos else None
            )
            if emb_key is None:
                raise ValueError(f"Missing positionEmbedding for position index {i}")
            p = np.asarray(pos[emb_key], dtype=np.float32)
            if p.ndim != 1:
                raise ValueError(f"{emb_key} at index {i} must be 1D")

            rows.append(np.concatenate([c, p], axis=0))
            pair_ids.append((cand_id, pid))
        X = np.stack(rows, axis=0)
        return X, pair_ids

    # --- Legacy: explicit features matrix
    if "features" in body:
        X = np.asarray(body["features"], dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # fabricate ids
        pair_ids = [(f"row_{i}", "n/a") for i in range(X.shape[0])]
        return X, pair_ids

    # --- Legacy: single resume+position embedding pair
    if "resumeEmbedding" in body and "positionEmbedding" in body:
        r = np.asarray(body["resumeEmbedding"], dtype=np.float32)
        p = np.asarray(body["positionEmbedding"], dtype=np.float32)
        if r.ndim != 1 or p.ndim != 1:
            raise ValueError("resumeEmbedding and positionEmbedding must be 1D arrays")
        X = np.concatenate([r, p], axis=0).reshape(1, -1)
        pair_ids = [("candidate", "position")]
        return X, pair_ids

    raise ValueError(
        "Invalid request. Provide either:\n"
        "A) {'positionEmbedding': [...], 'candidates': [{'candidateEmbedding': [...]}...]}\n"
        "B) {'candidateEmbedding': [...], 'positions': [{'positionEmbedding': [...]}...]}\n"
        "Or legacy forms: {'features': [[...]]} or {'resumeEmbedding': [...], 'positionEmbedding': [...]}."
    )


# -----------------------------------------------------------------------------
# Model session (global, created at import/cold start)
# -----------------------------------------------------------------------------
def _build_session() -> Tuple[ort.InferenceSession, str, int]:
    """
    Create ONNXRuntime session, detect input name and n_features.
    """
    model_path = _maybe_download_from_s3(MODEL_LOCAL_PATH, MODEL_S3_URI)

    so = ort.SessionOptions()
    so.intra_op_num_threads = INTRA_OP
    so.inter_op_num_threads = INTER_OP
    # Disable optimizations that sometimes increase cold start
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    n_features = int(sess.get_inputs()[0].shape[1]) if len(sess.get_inputs()[0].shape) == 2 else None
    return sess, input_name, n_features


SESSION, INPUT_NAME, N_FEATURES = _build_session()

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/")
def root():
    return "OK"


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_S3_URI or MODEL_LOCAL_PATH,
        "input_name": INPUT_NAME,
        "n_features": N_FEATURES,
        "threshold": PRED_THRESHOLD
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepted request payloads:

    A) Single position, many candidates
       {
         "positionId": "...",                     # optional but recommended
         "positionEmbedding": [...],              # 1D float list
         "candidates": [
           {"candidateId": "...", "candidateEmbedding": [...]},
           {"candidateId": "...", "candidateEmbedding": [...]}
         ]
       }

    B) Single candidate, many positions
       {
         "candidateId": "...",                    # optional but recommended
         "candidateEmbedding": [...],             # 1D float list
         "positions": [
           {"positionId": "...", "positionEmbedding": [...]},
           {"positionId": "...", "positionEmbedding": [...]}
         ]
       }

    Legacy (still supported for compatibility):
       - {"features": [[...], [...]]}
       - {"resumeEmbedding": [...], "positionEmbedding": [...]}

    Returns (JSON array at top level):
       [
         {"candidateId": "c1", "positionId": "pA", "probability": 0.8732},
         {"candidateId": "c2", "positionId": "pA", "probability": 0.4411},
         ...
       ]
    """
    try:
        body = request.get_json(force=True, silent=False) or {}

        # Build the model input batch and keep track of (candidateId, positionId) per row
        X_raw, pair_ids = _concat_if_present(body)

        # Enforce expected feature size if available from model
        X = _ensure_2d_float32(X_raw, expected_n_features=N_FEATURES)

        # Inference
        outputs = SESSION.run(None, {INPUT_NAME: X})
        pos_proba = _extract_pos_proba(outputs)

        # Normalize to 1D list of floats
        probas = [float(p) for p in np.asarray(pos_proba).reshape(-1)]

        # Map back to (candidateId, positionId)
        results = []
        for (cid, pid), p in zip(pair_ids, probas):
            results.append({
                "candidateId": cid,
                "positionId": pid,
                "probability": p
            })

        return jsonify(results)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Avoid leaking internals in prod; keep it simple
        return jsonify({"error": "inference_failed", "detail": str(e)}), 500


# Optional: simple metadata endpoint
@app.route("/metadata", methods=["GET"])
def metadata():
    try:
        outs = [o.name for o in SESSION.get_outputs()]
        return jsonify({
            "inputs": [{"name": i.name, "shape": i.shape, "type": str(i.type)} for i in SESSION.get_inputs()],
            "outputs": [{"name": n} for n in outs]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
